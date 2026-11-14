"""Train bootstrap experts and routers on separate source roles."""

from collections import defaultdict
import random

import torch
from torch.utils.data import DataLoader, Subset

from .blocks.router import candidate_context
from .blocks.targets import patch_occupancy, local_losses, gain_targets, gain_mse
from .model import expert_logits, TASK_VARIANTS
from .io import write_json


def bootstrap_indices(rows, seed):
    buckets = defaultdict(list)
    for index, row in enumerate(rows):
        buckets[(row['product'], row['defect'])].append(index)
    rng = random.Random(seed)
    groups = sorted(buckets)
    # Exactly N updates' worth of samples, approximately balanced by class.
    result = [rng.choice(buckets[groups[i % len(groups)]]) for i in range(len(rows))]
    rng.shuffle(result)
    return result


def train_experts(baseline, bank, dataset, config, seed, output, objective, device):
    bank.requires_grad_(True).train()
    histories = []
    for expert_index, expert in enumerate(bank.experts):
        sampling_seed = seed * 1000 + expert_index + 1
        indices = bootstrap_indices(dataset.rows, sampling_seed)
        write_json(output / f'bootstrap_{expert_index + 1}.json', dict(
            sampling_seed=sampling_seed,
            initialization_seed=bank.initialization_seeds[expert_index],
            image_ids=[dataset.rows[i]['image_id'] for i in indices]), exclusive=True)
        loader = DataLoader(Subset(dataset, indices), batch_size=config['batch_size'], shuffle=False)
        optimizer = torch.optim.Adam(expert.parameters(), lr=config['expert_lr'], betas=(0.5, 0.999))
        for epoch in range(config['expert_epochs']):
            total = 0.
            for batch in loader:
                features = baseline(batch['image'].to(device))
                probabilities = baseline.scorer(
                    bank.forward_expert(expert_index, features['h'], features['z0']),
                    features['text']).softmax(-1)
                loss = objective(probabilities, batch['training_mask'].to(device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
            histories.append(dict(expert=expert_index + 1, epoch=epoch, loss=total / len(loader)))
    bank.requires_grad_(False).eval()
    write_json(output / 'expert_training.json', histories)


def shuffled_donors(count, seed):
    if count < 2:
        raise ValueError('Shuffled semantic control requires at least two images')
    order = list(range(count))
    random.Random(seed).shuffle(order)
    # One fixed random cycle has no self-matches and preserves the sample pool.
    return {index: order[(i + 1) % count] for i, index in enumerate(order)}


@torch.no_grad()
def donor_context(baseline, dataset, indices, donors, device, text_indices=None, names=None):
    images = torch.stack([dataset[donors[int(i)]]['image'] for i in indices]).to(device)
    features = baseline(images)
    text = features['text'] if text_indices is None else features['text'][text_indices]
    logits = baseline.scorer(features['z0'], text)
    return candidate_context(logits, text, baseline.names if names is None else names)


def train_router_epoch(baseline, bank, router, dataset, config, variant, optimizer,
                       objective, device, generator, donors=None):
    router.train()
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, generator=generator)
    total = 0.
    for batch in loader:
        images = batch['image'].to(device)
        features = baseline(images)
        with torch.no_grad():
            logits = expert_logits(baseline, bank, features)
            occupancy, coverage = patch_occupancy(batch['mask'].to(device), logits.shape[2], logits.shape[-1])
            losses = local_losses(logits, occupancy)
            targets = gain_targets(losses)
            if variant == 'P2':
                # Canonical plus two mild, spatially aligned photometric views.
                # Brightness is applied in RGB space before CLIP normalization.
                mean = images.new_tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
                std = images.new_tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
                rgb = images * std + mean
                view_losses = []
                for factor in config['robust_brightness_factors']:
                    view = ((rgb * factor).clamp(0, 1) - mean) / std
                    view_losses.append(local_losses(expert_logits(baseline, bank, baseline(view)), occupancy))
                targets = gain_targets(losses, view_losses)
            context = candidate_context(logits[..., 0, :], features['text'], baseline.names)
            if donors is not None:
                donor = donor_context(baseline, dataset, batch['index'], donors, device)
                context = torch.cat((donor[..., :-3], context[..., -3:]), -1)
        prediction = router(features['h'].mean(0), context)
        if variant in TASK_VARIANTS:
            weights = prediction.softmax(-1)[None, ..., None]
            probabilities = (logits.softmax(-1) * weights).sum(-2)
            loss = objective(probabilities, batch['training_mask'].to(device))
        else:
            loss = gain_mse(prediction, targets, occupancy, coverage)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
    return total / len(loader)
