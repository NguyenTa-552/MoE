"""Source selection and locked target evaluation, with disk-backed scores."""

from collections import defaultdict
import csv
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from .blocks.scoring import dense_maps
from .blocks.targets import patch_occupancy, local_losses
from .diagnostics import routing_diagnostics
from .io import write_json
from .metrics import binary_metrics, pixel_aupro, multitype_metrics
from .model import expert_logits, route
from .training import donor_context


def primary_metrics(products, threshold):
    per_class = {}
    for product, record in products.items():
        truth, scores = record['truth'], record['scores']
        for index, name in enumerate(record['labels'][1:], 1):
            actual, predicted = truth == index, scores[..., index] >= threshold
            support = int(actual.sum())
            if not support:
                # Report absence, exclude from defect-only primary macro.
                f1 = None
            else:
                denominator = support + int(predicted.sum())
                f1 = float(2 * np.count_nonzero(actual & predicted) / denominator)
            per_class[f'{product}/{name}'] = dict(f1=f1, support=support)
    defined = [entry['f1'] for entry in per_class.values() if entry['f1'] is not None]
    return dict(macro_defect_f1=float(np.mean(defined)) if defined else None,
                threshold=threshold, per_class=per_class,
                averaging='Equal supported product/defect pairs; normal excluded; absent pairs null')


def compare_to_baseline(candidate, baseline):
    deltas = {}
    for name, entry in candidate['per_class'].items():
        before = baseline['per_class'].get(name, {}).get('f1')
        after = entry['f1']
        deltas[name] = None if before is None or after is None else after - before
    defined = [d for d in deltas.values() if d is not None]
    return dict(per_class_deltas=deltas, worst_class_delta=min(defined) if defined else None,
                macro_delta=None if candidate['macro_defect_f1'] is None or baseline['macro_defect_f1'] is None
                else candidate['macro_defect_f1'] - baseline['macro_defect_f1'])


@torch.no_grad()
def evaluate(baseline, bank, router, dataset, variant, device, thresholds,
             delta=0., scratch=None, export=None, donors=None, product_protocol=None,
             binary_names=None, coverage_target=None):
    started = time.monotonic()
    if router is not None:
        router.eval()
    if bank is not None:
        bank.eval()
    if export is not None:
        export = Path(export)
        export.mkdir(parents=True, exist_ok=True)
    losses_all, selections, predictions, validities = [], [], [], []
    class_memberships = defaultdict(list)
    per_group = defaultdict(list)
    image_predictions, group_diagnostics = [], {}
    with tempfile.TemporaryDirectory(prefix='moe-evaluation-', dir=scratch) as temporary:
        products = {}
        for index in range(len(dataset)):
            row, sample = dataset.rows[index], dataset[index]
            features = baseline(sample['image'][None].to(device))
            text = features['text']
            labels = ['normal'] + baseline.names
            mask = sample['mask'][None].to(device)
            # The post-hoc oracle is confined to typed diagnostics; its binary
            # localization is not a deployable prediction either.
            if product_protocol is not None:
                labels = product_protocol['products'][row['product']]
                indices = [0] + [baseline.names.index(name) + 1 for name in labels[1:]]
                remapped = torch.full_like(mask, -1)
                for local, global_id in enumerate(indices):
                    remapped[mask == global_id] = local
                if ((mask != -1) & (remapped == -1)).any():
                    raise ValueError('Ground truth contains a type outside the product vocabulary')
                mask = remapped
                text = text[indices]
            else:
                indices = None
            local_features = dict(features, text=text)
            logits = expert_logits(baseline, bank, features, text)
            occupancy, coverage = patch_occupancy(mask, logits.shape[2], logits.shape[-1])
            losses = local_losses(logits, occupancy)
            donor = None
            if donors is not None:
                donor = donor_context(baseline, dataset, [index], donors, device,
                                      text_indices=indices, names=labels[1:])
            routing = route(logits, local_features, labels[1:], variant, router, delta,
                            occupancy=occupancy, donor_context=donor)
            routed_loss = -(routing['probabilities'].clamp_min(1e-12).log()
                            * occupancy[None]).sum(-1).mean(0)
            maps = dense_maps(routing['probabilities'], mask.shape[-2:])
            scores = maps.mean(0)[0].permute(1, 2, 0).cpu().numpy()
            anomaly_map = maps[:, :, 1:].sum(2).sum(0)[0].cpu().numpy()
            image_score = features['image_logits'].softmax(-1)[0, 1].item()
            if binary_names is not None:
                binary_ids = [0] + [baseline.names.index(name) + 1 for name in binary_names]
                binary_text = features['text'][binary_ids]
                binary_features = dict(features, text=binary_text)
                binary_logits = expert_logits(baseline, bank, features, binary_text)
                binary_donor = None if donors is None else donor_context(
                    baseline, dataset, [index], donors, device, binary_ids, binary_names)
                binary_occupancy = None
                if variant == 'O1':
                    # Oracle local loss uses the available image type's binary
                    # candidate alias when present; refuse ambiguous mappings.
                    raise ValueError('O1 target binary evaluation is intentionally unavailable')
                binary_route = route(binary_logits, binary_features, binary_names, variant,
                                     router, delta, donor_context=binary_donor)
                binary_maps = dense_maps(binary_route['probabilities'], mask.shape[-2:])
                anomaly_map = binary_maps[:, :, 1:].sum(2).sum(0)[0].cpu().numpy()
                binary_anchor = torch.cat((binary_text[:1], F.normalize(binary_text[1:].mean(0, keepdim=True), dim=-1)))
                image_score = (F.normalize(features['image'], dim=-1) @ binary_anchor.T
                               / baseline.scorer.temperature).softmax(-1)[0, 1].item()
            product = row['product']
            if product not in products:
                count = sum(r['product'] == product for r in dataset.rows)
                height, width, classes = scores.shape
                prefix = Path(temporary) / str(len(products))
                products[product] = dict(
                    truth=np.lib.format.open_memmap(str(prefix) + '-truth.npy', mode='w+', dtype='int32', shape=(count, height, width)),
                    scores=np.lib.format.open_memmap(str(prefix) + '-scores.npy', mode='w+', dtype='float32', shape=(count, height, width, classes)),
                    binary=np.lib.format.open_memmap(str(prefix) + '-binary.npy', mode='w+', dtype='float32', shape=(count, height, width)),
                    labels=labels, count=0, image_truth=[], image_scores=[])
            record = products[product]
            offset = record['count']
            record['truth'][offset] = mask[0].cpu().numpy()
            record['scores'][offset], record['binary'][offset] = scores, anomaly_map
            record['image_truth'].append(row['anomaly'])
            record['image_scores'].append(image_score)
            record['count'] += 1
            valid = coverage > 0
            losses_all.append(losses.cpu())
            selections.append(routing['selected'].cpu())
            validities.append(valid.cpu())
            if routing['predicted'] is not None:
                predictions.append(routing['predicted'].cpu())
            unit_index = len(losses_all) - 1
            for class_id, class_name in enumerate(labels):
                membership = (occupancy[..., class_id] > 0).cpu() & valid.cpu()
                if membership.any():
                    class_memberships[f'{product}/{class_name}'].append((unit_index, membership[0]))
            per_group[f'product:{product}'].append(unit_index)
            per_group[f'defect:{row["defect"]}'].append(unit_index)
            image_predictions.append(dict(image_id=row['image_id'], product=product,
                                          prediction_file=f'{index:06d}.npz',
                                          defect=row['defect'], anomaly=row['anomaly'],
                                          baseline_loss=float(losses[..., 0][valid].mean()),
                                          selected_loss=float(routed_loss[valid].mean()),
                                          fallback_coverage=None if variant == 'B2' else float((routing['selected'][valid] == 0).float().mean())))
            if export is not None:
                np.savez_compressed(export / f'{index:06d}.npz',
                    truth=record['truth'][offset], scores=scores, binary_map=anomaly_map,
                    baseline_scores=dense_maps(logits[..., 0, :].softmax(-1), mask.shape[-2:]).mean(0)[0].permute(1, 2, 0).cpu().numpy(),
                    selected=(torch.full_like(routing['selected'], -1) if variant == 'B2' else routing['selected']).cpu().numpy(),
                    predicted_gains=np.array([]) if routing['predicted'] is None else routing['predicted'].cpu().numpy(),
                    context_margin=np.array([]) if routing['context'] is None else routing['context'][..., -3].cpu().numpy(),
                    fallback=((routing['selected'] == 0) & (variant != 'B2')).cpu().numpy())
        primary = [primary_metrics(products, threshold) for threshold in thresholds]
        legacy = {}
        for product, record in products.items():
            valid = record['truth'] != -1
            typed, per_defect = multitype_metrics(record['truth'][valid], record['scores'][valid], record['labels'])
            binary_truth = record['truth'] > 0
            binary = binary_metrics(binary_truth[valid], record['binary'][valid])
            binary['aupro'] = pixel_aupro(binary_truth, record['binary']) if valid.all() else None
            legacy[product] = dict(image=binary_metrics(record['image_truth'], record['image_scores']),
                                   binary_pixel=binary, multi_type=typed, per_defect=per_defect)
            actual = record['truth'][valid]
            predicted = record['scores'][valid].argmax(-1)
            classes = len(record['labels'])
            confusion = np.bincount(actual * classes + predicted, minlength=classes * classes).reshape(classes, classes)
            legacy[product]['confusion_matrix'] = dict(labels=record['labels'], counts=confusion.tolist())
        loss_tensor = torch.cat(losses_all)
        selected_tensor = torch.cat(selections)
        validity = torch.cat(validities)
        predicted_tensor = torch.cat(predictions) if predictions else None
        diagnostics = routing_diagnostics(loss_tensor, selected_tensor, predicted_tensor, validity)
        for group, ids in per_group.items():
            take = torch.tensor(ids)
            group_diagnostics[group] = routing_diagnostics(
                loss_tensor[take], selected_tensor[take],
                None if predicted_tensor is None else predicted_tensor[take], validity[take])
        class_diagnostics = {}
        for label, memberships in class_memberships.items():
            membership = torch.zeros_like(validity)
            for image_index, mask in memberships:
                membership[image_index] = mask
            class_diagnostics[label] = routing_diagnostics(loss_tensor, selected_tensor,
                                                           predicted_tensor, validity & membership)
        numeric_keys = ('harmful_routing_rate', 'missed_opportunity_rate', 'selection_regret', 'fallback_coverage')
        balanced = {}
        for key in numeric_keys:
            values = [r[key] for r in class_diagnostics.values() if r.get(key) is not None]
            balanced[key] = sum(values) / len(values) if values else None
        all_valid_losses = loss_tensor[validity]
        complementarity = dict(
            baseline_local_ce=float(all_valid_losses[:, 0].mean()),
            oracle_local_ce=float(all_valid_losses.min(-1).values.mean()),
            individual_local_ce=all_valid_losses.mean(0).tolist(),
            oracle_gain=float((all_valid_losses[:, 0] - all_valid_losses.min(-1).values).mean()))
        result = dict(primary=primary, legacy=legacy, routing=diagnostics,
                      routing_by_group=group_diagnostics, routing_by_class=class_diagnostics,
                      routing_class_balanced=balanced, complementarity=complementarity)
        if variant == 'B2':
            result.update(routing=None, routing_by_group=None, routing_by_class=None,
                          routing_class_balanced=None,
                          routing_reason='Uniform probability ensemble has no hard expert choice')
        if coverage_target is not None and predicted_tensor is not None:
            maxima = predicted_tensor[validity].max(-1).values
            if coverage_target == 0:
                calibrated = float(maxima.min()) - 1.
            elif coverage_target == 1:
                calibrated = float(maxima.max())
            else:
                calibrated = float(torch.quantile(maxima, coverage_target))
            result['coverage_calibration'] = dict(target=coverage_target, delta=calibrated,
                                                  achieved=float((maxima <= calibrated).float().mean()),
                                                  tie_rule='Fallback when gain <= delta; exact coverage may be impossible with ties')
        if export is not None:
            write_json(export / 'index.json', image_predictions)
            # Deterministic examples: largest local-loss improvements/regressions.
            ranked = sorted(image_predictions, key=lambda r: r['selected_loss'] - r['baseline_loss'])
            write_json(export / 'failure_examples.json', dict(
                sampling_rule='Three lowest and highest selected-minus-baseline mean local CE',
                improvements=ranked[:3], regressions=list(reversed(ranked[-3:]))))
            render_examples(export, dataset, ranked[:3] + ranked[-3:])
        # Close disk mappings before TemporaryDirectory cleanup (also on Windows).
        for record in products.values():
            for key in ('truth', 'scores', 'binary'):
                record[key]._mmap.close()
    result['evaluation_runtime_seconds'] = time.monotonic() - started
    return result


def render_examples(export, dataset, examples):
    """Export image/GT/baseline/selected/expert panels for sampled cases."""
    from PIL import Image, ImageDraw
    for example in examples:
        path = export / example['prediction_file']
        with np.load(path) as values:
            truth = values['truth']
            selected_map = values['selected'][0]
            side = int(selected_map.size ** 0.5)
            maps = [truth, values['baseline_scores'].argmax(-1), values['scores'].argmax(-1),
                    selected_map.reshape(side, side)]
            image = Image.open(dataset.root / example['image_id']).convert('RGB')
            size = 160
            panel = Image.new('RGB', (size * 5, size + 24), 'white')
            panel.paste(image.resize((size, size)), (0, 24))
            for i, array in enumerate(maps, 1):
                array = np.asarray(array)
                # Fixed ID palette; normal/fallback zero remains black.
                colors = np.stack(((array * 67) % 256, (array * 131) % 256,
                                   (array * 193) % 256), -1).astype('uint8')
                panel.paste(Image.fromarray(colors).resize((size, size), Image.Resampling.NEAREST), (i * size, 24))
            draw = ImageDraw.Draw(panel)
            for i, title in enumerate(('Image', 'Ground truth', 'Baseline argmax', 'Selected argmax', 'Expert index')):
                draw.text((i * size + 4, 4), title, fill='black')
            panel.save(export / (path.stem + '-example.png'))


def export_per_class(path, primary, comparison=None):
    with Path(path).open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['product_defect', 'f1', 'support', 'delta_to_B0'])
        for name, values in primary['per_class'].items():
            delta = None if comparison is None else comparison['per_class_deltas'].get(name)
            writer.writerow([name, values['f1'], values['support'], delta])
