"""Experiment variants over one fixed expert bank."""

import torch
from .blocks.router import candidate_context
from .blocks.scoring import select_probabilities
from .blocks.selection import Selector
from .blocks.targets import local_losses


VARIANTS = ('B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'P1', 'P2', 'O1',
            'P1-no-fallback', 'P1-no-context', 'P1-shuffle')
TASK_VARIANTS = ('B3', 'B5')
SEMANTIC_VARIANTS = ('B5', 'P1', 'P2', 'P1-no-fallback', 'P1-shuffle')


def expert_logits(baseline, bank, features, text=None):
    text = features['text'] if text is None else text
    zero = baseline.scorer(features['z0'], text).unsqueeze(-2)
    if bank is None:
        return zero
    return torch.cat((zero, baseline.scorer(bank(features['h'], features['z0']), text)), -2)


def route(logits, features, names, variant, router=None, delta=0.,
          occupancy=None, donor_context=None):
    probabilities = logits.softmax(-1)
    shape = logits.shape[1:3]
    selected = torch.zeros(shape, dtype=torch.long, device=logits.device)
    predicted, context = None, None
    if variant == 'B0':
        chosen = probabilities[..., 0, :]
    elif variant == 'B1':
        selected.fill_(1)
        chosen = probabilities[..., 1, :]
    elif variant == 'B2':
        # Uniform ensemble of the three residual experts, excluding baseline.
        chosen = probabilities[..., 1:, :].mean(-2)
    elif variant == 'O1':
        if occupancy is None:
            raise ValueError('Oracle is available only with evaluation labels')
        selected = local_losses(logits, occupancy).argmin(-1)
        chosen = select_probabilities(probabilities, selected)
    else:
        context = candidate_context(logits[..., 0, :], features['text'], names)
        if donor_context is not None:
            # Shuffle only competing semantic embeddings; preserve own margin,
            # abnormality, validity slot, and appearance features.
            context = torch.cat((donor_context[..., :-3], context[..., -3:]), -1)
        output = router(features['h'].mean(0), context)
        if variant in TASK_VARIANTS:
            selected = output.argmax(-1)
        else:
            predicted = output
            selected = Selector()(output, delta, fallback=variant != 'P1-no-fallback')
        chosen = select_probabilities(probabilities, selected)
    return dict(probabilities=chosen, selected=selected, predicted=predicted, context=context)
