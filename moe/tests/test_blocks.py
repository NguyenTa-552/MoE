import io
import unittest

import torch
from torch.nn import functional as F

from moe.blocks.experts import ExpertBank
from moe.blocks.router import GainRouter, candidate_context
from moe.blocks.scoring import BaselineScorer, dense_maps
from moe.blocks.selection import Selector
from moe.blocks.targets import patch_occupancy, local_losses, gain_targets, gain_mse
from moe.diagnostics import routing_diagnostics
from moe.model import expert_logits, route


class BlockTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)
        self.h = torch.randn(2, 2, 4, 6)
        self.z0 = torch.randn(2, 2, 4, 5)
        self.text = F.normalize(torch.randn(4, 5), dim=-1)
        self.scorer = BaselineScorer()
        self.bank = ExpertBank(6, 5, layers=2)
        self.features = dict(h=self.h, z0=self.z0, text=self.text)
        self.names = ['scratch', 'rust', 'hole']

    def test_zero_residual_and_forced_fallback_preserve_all_layer_scores(self):
        experts = self.bank(self.h, self.z0)
        expected = self.z0.unsqueeze(-2).expand_as(experts)
        torch.testing.assert_close(experts, expected, atol=0, rtol=0)
        class Baseline:
            scorer = self.scorer
        logits = expert_logits(Baseline(), self.bank, self.features)
        router = GainRouter(6, 5)
        routed = route(logits, self.features, self.names, 'P1', router, delta=1e10)
        torch.testing.assert_close(routed['probabilities'], logits[..., 0, :].softmax(-1), atol=0, rtol=0)
        maps = dense_maps(routed['probabilities'], (8, 8))
        upstream = []
        for patches in self.z0:
            probabilities = (F.normalize(patches, dim=-1) @ self.text.T / 0.07).softmax(-1)
            grid = probabilities.reshape(2, 2, 2, 4).permute(0, 3, 1, 2)
            upstream.append(F.interpolate(grid, (8, 8), mode='bilinear', align_corners=True))
        torch.testing.assert_close(maps, torch.stack(upstream))

    def test_semantic_context_invariant_to_label_order_even_with_ties(self):
        logits = torch.zeros(2, 2, 4, 4)
        before = candidate_context(logits, self.text, self.names)
        torch.testing.assert_close(before[..., -2], logits.softmax(-1)[..., 1:].sum(-1).sum(0))
        permutation = [0, 3, 1, 2]
        after = candidate_context(logits[..., permutation], self.text[permutation], ['hole', 'scratch', 'rust'])
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        torch.testing.assert_close(before[..., :5], self.text[3].expand(2, 4, 5))
        scorer_before = self.scorer(self.z0, self.text)
        scorer_after = self.scorer(self.z0, self.text[permutation])
        torch.testing.assert_close(scorer_after, scorer_before[..., permutation])

    def test_single_candidate_has_masked_second_slot_and_empty_defect_is_valid(self):
        context = candidate_context(torch.zeros(2, 2, 4, 2), self.text[:2], ['scratch'])
        self.assertTrue((context[..., 5:10] == 0).all())
        self.assertTrue((context[..., -1] == 0).all())
        occupancy, coverage = patch_occupancy(torch.zeros(2, 8, 8, dtype=torch.long), 4, 2)
        self.assertTrue((occupancy[..., 0] == 1).all())
        self.assertTrue((coverage == 1).all())

    def test_threshold_boundary_negative_gains_and_deterministic_expert_tie(self):
        gains = torch.tensor([[0.1, 0.1, 0.], [-1., -2., -3.], [0.2, 0.1, 0.1]])
        self.assertEqual(Selector()(gains, 0.1).tolist(), [0, 0, 1])
        self.assertEqual(Selector()(gains, 0.1, fallback=False).tolist(), [1, 1, 1])
        with self.assertRaises(ValueError):
            Selector()(torch.tensor([[float('nan')]]))

    def test_mixed_tiny_patch_and_ignored_pixels(self):
        mask = torch.zeros(1, 8, 8, dtype=torch.long)
        mask[0, 0, 0] = 1
        mask[0, :4, 4:] = -1
        occupancy, coverage = patch_occupancy(mask, 4, 3)
        self.assertAlmostEqual(occupancy[0, 0, 1].item(), 1 / 16)
        self.assertEqual(coverage[0, 1].item(), 0)
        self.assertAlmostEqual(occupancy[0, 0].sum().item(), 1.)
        all_ignored = torch.full_like(mask, -1)
        occupancy, coverage = patch_occupancy(all_ignored, 4, 3)
        predicted = torch.randn(1, 4, 3, requires_grad=True)
        loss = gain_mse(predicted, torch.randn_like(predicted), occupancy, coverage)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue((predicted.grad == 0).all())

    def test_gain_sign_robust_minimum_and_detachment(self):
        losses = torch.tensor([[[2., 1., 3.]]], requires_grad=True)
        gains = gain_targets(losses)
        self.assertEqual(gains.tolist(), [[[1., -1.]]])
        self.assertFalse(gains.requires_grad)
        robust = gain_targets(losses, [torch.tensor([[[2., 4., 1.]]])])
        self.assertEqual(robust.tolist(), [[[-2., -1.]]])

    def test_soft_ce_matches_exclusive_ce(self):
        logits = torch.randn(2, 2, 4, 3, 4)
        targets = torch.randint(0, 4, (2, 4))
        occupancy = F.one_hot(targets, 4).float()
        observed = local_losses(logits, occupancy)
        expected = -logits.log_softmax(-1).gather(-1, targets[None, ..., None, None].expand(2, 2, 4, 3, 1)).squeeze(-1).mean(0)
        torch.testing.assert_close(observed, expected)

    def test_expert_inputs_and_gain_targets_do_not_receive_gradients(self):
        h, z0 = self.h.requires_grad_(), self.z0.requires_grad_()
        self.bank(h, z0).sum().backward()
        self.assertIsNone(h.grad)
        self.assertIsNone(z0.grad)
        self.assertTrue(any(p.grad is not None for p in self.bank.parameters()))
        targets = torch.randn(2, 4, 3, requires_grad=True)
        prediction = torch.randn_like(targets, requires_grad=True)
        occupancy = F.one_hot(torch.zeros(2, 4, dtype=torch.long), 4).float()
        gain_mse(prediction, targets, occupancy, torch.ones(2, 4)).backward()
        self.assertIsNone(targets.grad)

    def test_router_checkpoint_roundtrip_reproduces_gains_and_selection(self):
        router = GainRouter(6, 5)
        context = candidate_context(self.scorer(self.z0, self.text), self.text, self.names)
        before = router(self.h.mean(0), context)
        memory = io.BytesIO()
        torch.save(router.state_dict(), memory)
        memory.seek(0)
        restored = GainRouter(6, 5)
        restored.load_state_dict(torch.load(memory, weights_only=True))
        after = restored(self.h.mean(0), context)
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        self.assertTrue(torch.equal(Selector()(before), Selector()(after)))

    def test_diagnostics_null_denominators_and_harmful_selection(self):
        losses = torch.tensor([[[1., 2., 0.5], [1., 2., 2.]]])
        selected = torch.tensor([[1, 0]])
        result = routing_diagnostics(losses, selected)
        self.assertEqual(result['harmful_routing_rate'], 1.)
        self.assertEqual(result['missed_opportunity_rate'], 0.)
        fallback = routing_diagnostics(losses, torch.zeros_like(selected))
        self.assertIsNone(fallback['harmful_routing_rate'])
        self.assertEqual(fallback['missed_opportunity_rate'], 1.)


if __name__ == '__main__':
    unittest.main()
