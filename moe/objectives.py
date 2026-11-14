"""The upstream full-image objective for experts and task-loss routers."""

from .blocks.scoring import dense_maps


class DAPOObjective:
    def __init__(self):
        from loss import FocalLoss, BinaryDiceLoss
        self.focal, self.dice = FocalLoss(), BinaryDiceLoss()

    def __call__(self, probabilities, mask):
        # The frozen CLS CE is constant w.r.t. visual adapters and omitted
        # from optimization; it is unchanged for every expert/route.
        if (mask < 0).any():
            raise ValueError('Upstream Focal/Dice has no ignore-mask support; isolate a masked-objective ablation before training')
        maps = dense_maps(probabilities, mask.shape[-2:])
        typed = mask[:, None]
        binary = (typed > 0).to(probabilities.dtype)
        loss = probabilities.new_zeros(())
        for layer in maps:
            loss = loss + self.focal(layer, typed)
            loss = loss + self.dice(layer[:, :1].contiguous(), 1 - binary)
            loss = loss + self.dice(layer[:, 1:].sum(1, keepdim=True), binary)
        return 4 * loss
