"""Post-hoc routing diagnostics; labels do not enter inference."""

import torch


def routing_diagnostics(losses, selected, predicted=None, valid=None, epsilon=0.001):
    if valid is None:
        valid = torch.ones_like(selected, dtype=torch.bool)
    losses, selected = losses[valid], selected[valid]
    if not selected.numel():
        return {'units': 0, 'fallback_coverage': None, 'reason': 'No valid units'}
    gain = losses[:, :1] - losses
    chosen = gain.gather(1, selected[:, None]).squeeze(1)
    nonbaseline = selected != 0
    opportunities = (gain[:, 1:].max(-1).values > epsilon) if gain.shape[-1] > 1 else torch.zeros_like(nonbaseline)
    def ratio(numerator, denominator):
        return float(numerator.sum() / denominator.sum()) if denominator.sum() else None
    result = dict(
        units=selected.numel(), epsilon=epsilon,
        fallback_coverage=float((~nonbaseline).float().mean()),
        harmful_routing_rate=ratio((chosen < -epsilon) & nonbaseline, nonbaseline),
        missed_opportunity_rate=ratio((~nonbaseline) & opportunities, opportunities),
        selection_regret=float((losses.gather(1, selected[:, None]).squeeze(1)
                                - losses.min(-1).values).mean()),
        utilization=[float((selected == i).float().mean()) for i in range(losses.shape[-1])],
        mean_observed_gains=gain[:, 1:].mean(0).tolist(),
    )
    if predicted is not None:
        predicted = predicted[valid]
        result.update(gain_mae=float((predicted - gain[:, 1:]).abs().mean()),
                      gain_sign_accuracy=float((predicted.sign() == gain[:, 1:].sign()).float().mean()))
    return result
