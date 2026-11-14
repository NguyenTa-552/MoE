"""Detached appearance and competing-label semantic routing."""

import torch
from torch import nn
from torch.nn import functional as F


def candidate_context(baseline_logits, text, names):
    """Canonical-name tie breaks make the context invariant to label ordering.

    Logits [L,B,P,C], text [C,D], names excludes channel-zero normal.
    Abnormality sums layer and defect probabilities, matching the deployed
    baseline localization branch before spatial upsampling.
    """
    if len(names) != text.shape[0] - 1 or len(set(names)) != len(names) or not names:
        raise ValueError('Provide unique defect names aligned with text channels')
    layer_probabilities = baseline_logits.detach().softmax(-1)
    probabilities = layer_probabilities.mean(0)
    order = sorted(range(len(names)), key=lambda i: names[i])
    ids = torch.tensor(order, device=text.device) + 1
    scores = probabilities[..., ids]
    ranked = scores.argsort(dim=-1, descending=True, stable=True)
    first = ids[ranked[..., 0]]
    embeddings = F.normalize(text.detach(), dim=-1)
    one = embeddings[first]
    if len(names) > 1:
        second = ids[ranked[..., 1]]
        two = embeddings[second]
        margin = scores.gather(-1, ranked[..., :1]) - scores.gather(-1, ranked[..., 1:2])
        mask = torch.ones_like(margin)
    else:
        two = torch.zeros_like(one)
        margin = scores[..., :1]
        mask = torch.zeros_like(margin)
    abnormal = layer_probabilities[..., 1:].sum(-1).sum(0).unsqueeze(-1)
    return torch.cat((one, two, margin, abnormal, mask), -1).detach()


class GainRouter(nn.Module):
    def __init__(self, feature_dim, text_dim, experts=3, hidden_dim=128,
                 semantic=True, task_loss=False):
        super().__init__()
        self.semantic = semantic
        self.context_dim = 2 * text_dim + 3
        self.network = nn.Sequential(
            nn.Linear(feature_dim + self.context_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, experts + int(task_loss)),
        )

    def forward(self, features, context):
        # Zero context keeps appearance and semantic routers parameter matched.
        context = context.detach() if self.semantic else torch.zeros_like(context)
        return self.network(torch.cat((features.detach(), context), -1))
