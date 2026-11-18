"""Actual upstream prompt/adapter/loss classes with a tiny synthetic encoder.

These checks exercise wrapper interfaces, not pretrained CLIP accuracy.
"""

from pathlib import Path
from types import ModuleType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from moe.dapo import FrozenDAPO
from moe.io import read_json
from moe.objectives import DAPOObjective


ROOT = Path(__file__).resolve().parents[2] / 'DAPO'


def reference_module(name):
    module = ModuleType(name)
    path = ROOT / f'{name}.py'
    # Compile directly to avoid creating bytecode in the reference checkout.
    exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
    return module


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Module()
        self.visual.patch_size = (2, 2)
        self.visual.conv1 = nn.Conv2d(3, 6, 2)
        self.ln_final = nn.LayerNorm(16)
        self.token_embedding = nn.Embedding(100, 16)

    def encode_image(self, inputs, layers):
        image, _ = inputs
        patches = F.adaptive_avg_pool2d(image, 2).flatten(2).transpose(1, 2)
        tokens = torch.cat((patches, patches.square()), -1)
        cls = torch.zeros(image.shape[0], 1, 6, device=image.device)
        image_features = F.pad(patches.mean(1), (0, 13), value=0.1)
        return image_features, [torch.cat((cls, tokens * (i + 1)), 1) for i in range(len(layers))], None

    def encode_learn_prompts(self, prompts, learned, template):
        return prompts.mean((1, 2)), None


def tokenizer(text):
    if isinstance(text, list):
        text = text[0]
    ids = [1] + [ord(c) % 90 + 2 for c in text[:75]] + [99]
    return torch.tensor([ids + [0] * (77 - len(ids))])


@unittest.skipUnless((ROOT / 'adapter.py').is_file(), 'Requires the read-only DAPO reference checkout')
class DAPOBoundaryTests(unittest.TestCase):
    def test_real_upstream_blocks_load_and_match_manual_forward(self):
        modules = {name: reference_module(name) for name in ('adapter', 'learnable_prompt', 'visual_prompt_tuning')}
        def create(*args, **kwargs):
            return TinyBackbone(), None, transforms.Compose([transforms.Resize(8), transforms.CenterCrop(8), transforms.ToTensor()])
        clip = SimpleNamespace(create_model_and_transforms=create,
                               get_tokenizer=lambda name: tokenizer,
                               get_model_config=lambda name: {'embed_dim': 16, 'vision_cfg': {'width': 6}})
        config = read_json(Path(__file__).resolve().parents[1] / 'configs/experiment.example.json')
        config['model'].update(image_size=8, feature_layers=[1, 2], depth=2, text_depth=2,
                               normal_token_cnt=2, abnormal_token_cnt=2, prompt_count=2, layer_token_cnt=2,
                               prefix_token_cnt=2)
        backbone = TinyBackbone()
        prompt = modules['learnable_prompt'].LearnablePrompt(backbone, ['scratch', 'rust'], 2, 2, 2, 2, 2, 'cpu', tokenizer)
        visual = modules['visual_prompt_tuning'].VisualPromptTuning(backbone, 2, 2, 'cpu')
        adapter = modules['adapter'].LinearAdapter(6, 16, 2)
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules', modules), patch('moe.dapo.upstream', return_value=clip):
            path = Path(tmp) / 'checkpoint.pt'
            torch.save(dict(prompt_learner=prompt.state_dict(), visual_prompt=visual.state_dict(),
                            linear_adapter=adapter.state_dict()), path)
            baseline = FrozenDAPO(config, path, ['scratch', 'rust'])
            baseline.train(True)
            self.assertFalse(baseline.training)
            self.assertFalse(any(p.requires_grad for p in baseline.parameters()))
            images = torch.randn(2, 3, 8, 8)
            actual = baseline(images)
            with torch.no_grad():
                image, tokens, _ = baseline.backbone.encode_image([images, baseline.visual_prompt()], [1, 2])
                raw = torch.stack([token[:, 1:].clone() for token in tokens])
                projected = torch.stack(baseline.adapter(tokens))
                prompts, learned, template = baseline.text_prompt(F.normalize(image, dim=-1))
                text, _ = baseline.backbone.encode_learn_prompts(prompts, learned, template)
                expected = F.normalize(projected, dim=-1) @ F.normalize(text, dim=-1).T / 0.07
            torch.testing.assert_close(actual['h'], raw)
            torch.testing.assert_close(actual['z0'], projected)
            torch.testing.assert_close(actual['baseline_logits'], expected)
            self.assertEqual(actual['h'].shape, (2, 2, 4, 6))
            self.assertEqual(actual['z0'].shape, (2, 2, 4, 16))
            # The alias checkpoint format must select exactly the same learned blocks.
            torch.save(dict(prompt_learner_state_dict=prompt.state_dict(), visual_prompt_state_dict=visual.state_dict(),
                            trainable_adapter_state_dict=adapter.state_dict()), path)
            alias = FrozenDAPO(config, path, ['scratch', 'rust'])
            for before, after in zip(baseline.adapter.parameters(), alias.adapter.parameters()):
                torch.testing.assert_close(before, after)
            # The archived clip_ad runner wraps plain DAPO blocks in a mapping.
            nested = dict(format_version=1, epoch=7, blocks=dict(
                visual_prompt=visual.state_dict(), text_prompt=prompt.state_dict(),
                spatial_adapter=adapter.state_dict()))
            torch.save(nested, path)
            restored = FrozenDAPO(config, path, ['scratch', 'rust'])
            self.assertEqual(restored.checkpoint_epoch, 7)
            for name in ('visual_prompt', 'text_prompt', 'adapter'):
                expected_state = getattr(baseline, name).state_dict()
                restored_state = getattr(restored, name).state_dict()
                self.assertEqual(set(expected_state), set(restored_state))
                for key in expected_state:
                    torch.testing.assert_close(restored_state[key], expected_state[key])
            # Dropping learned fusion would change the reproduced baseline.
            nested['blocks']['fusion'] = {'gates': torch.ones(2)}
            torch.save(nested, path)
            with self.assertRaisesRegex(ValueError, 'cannot be discarded'):
                FrozenDAPO(config, path, ['scratch', 'rust'])
            del nested['blocks']['fusion']
            nested['format_version'] = 99
            torch.save(nested, path)
            with self.assertRaisesRegex(ValueError, 'Unsupported clip_ad checkpoint format'):
                FrozenDAPO(config, path, ['scratch', 'rust'])

    def test_expert_objective_uses_actual_upstream_focal_and_dice(self):
        from moe.blocks.scoring import dense_maps
        reference = reference_module('loss')
        with patch.dict('sys.modules', {'loss': reference}):
            objective = DAPOObjective()
        logits = torch.randn(2, 2, 4, 3, requires_grad=True)
        probabilities = logits.softmax(-1)
        target = torch.randint(0, 3, (2, 8, 8))
        maps = dense_maps(probabilities, (8, 8))
        binary = (target[:, None] > 0).float()
        expected = 0.
        for layer in maps:
            expected += reference.FocalLoss()(layer, target[:, None])
            expected += reference.BinaryDiceLoss()(layer[:, :1].contiguous(), 1 - binary)
            expected += reference.BinaryDiceLoss()(layer[:, 1:].sum(1, keepdim=True), binary)
        actual = objective(probabilities, target)
        torch.testing.assert_close(actual, 4 * expected)
        actual.backward()
        self.assertIsNotNone(logits.grad)
        with self.assertRaisesRegex(ValueError, 'ignore-mask'):
            objective(probabilities, torch.full_like(target, -1))


if __name__ == '__main__':
    unittest.main()
