"""Read-only integration with the verified upstream DAPO blocks."""

import importlib
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from .blocks.scoring import BaselineScorer


def upstream(root):
    # Keep the reference checkout unchanged, including Python bytecode caches.
    sys.dont_write_bytecode = True
    root = Path(root).resolve()
    if not (root / 'open_clip/__init__.py').is_file():
        raise FileNotFoundError(f'Missing upstream DAPO/OpenCLIP at {root}')
    existing = sys.modules.get('open_clip')
    if existing is not None and Path(existing.__file__).resolve().parent != root / 'open_clip':
        raise RuntimeError('An unrelated open_clip package was imported before DAPO')
    sys.path.insert(0, str(root))
    return importlib.import_module('open_clip')


class FrozenDAPO(nn.Module):
    def __init__(self, config, checkpoint, names, device='cpu'):
        super().__init__()
        clip = upstream(config['dapo_root'])
        from adapter import LinearAdapter
        from learnable_prompt import LearnablePrompt
        from visual_prompt_tuning import VisualPromptTuning
        from torchvision import transforms
        model = config['model']
        self.backbone, _, self.preprocess = clip.create_model_and_transforms(
            model['name'], pretrained=model['pretrained'], force_image_size=model['image_size'])
        self.backbone.to(device)
        size = model['image_size']
        self.preprocess.transforms[0] = transforms.Resize(
            (size, size), interpolation=transforms.InterpolationMode.BICUBIC,
            max_size=None, antialias=None)
        self.preprocess.transforms[1] = transforms.CenterCrop((size, size))
        self.mask_transform = transforms.Compose([
            transforms.Resize((size, size)), transforms.CenterCrop(size), transforms.ToTensor()])
        self.text_prompt = LearnablePrompt(
            self.backbone, list(names), model['normal_token_cnt'], model['abnormal_token_cnt'],
            model['prompt_count'], model['text_depth'], model['layer_token_cnt'], device,
            clip.get_tokenizer(model['name']))
        # Upstream template tensors are plain attributes. Buffers allow transfers
        # while preserving the checkpoint's candidate-independent learned state.
        for key, value in list(vars(self.text_prompt).items()):
            if isinstance(value, torch.Tensor):
                delattr(self.text_prompt, key)
                self.text_prompt.register_buffer(key, value, persistent=False)
        self.visual_prompt = VisualPromptTuning(
            self.backbone, model['depth'], model['prefix_token_cnt'], device)
        architecture = clip.get_model_config(model['name'])
        self.input_dim = architecture['vision_cfg']['width']
        self.output_dim = architecture['embed_dim']
        self.layers = model['feature_layers']
        self.adapter = LinearAdapter(self.input_dim, self.output_dim, len(self.layers))
        self.scorer = BaselineScorer(model['temperature'])
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.checkpoint_epoch = saved.get('epoch')
        aliases = {
            'visual_prompt': ('visual_prompt', 'visual_prompt_state_dict'),
            'text_prompt': ('prompt_learner', 'prompt_learner_state_dict'),
            'adapter': ('linear_adapter', 'trainable_adapter_state_dict'),
        }
        for block, keys in aliases.items():
            weights = next((saved[key] for key in keys if saved.get(key)), None)
            if weights is None:
                raise ValueError(f'Missing original DAPO checkpoint block: {block}')
            getattr(self, block).load_state_dict(weights)
        self.to(device).requires_grad_(False).eval()
        self.names = list(names)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, images):
        image, tokens, _ = self.backbone.encode_image([images, self.visual_prompt()], self.layers)
        h = torch.stack([t[:, 1:] for t in tokens])
        z0 = torch.stack(self.adapter(list(tokens)))
        prompts, learned, template = self.text_prompt(F.normalize(image, dim=-1))
        text, _ = self.backbone.encode_learn_prompts(prompts, learned, template)
        text = F.normalize(text, dim=-1)
        binary_text = torch.cat([text[:1], F.normalize(text[1:].mean(0, keepdim=True), dim=-1)])
        image_logits = F.normalize(image, dim=-1) @ binary_text.T / self.scorer.temperature
        return dict(h=h.detach(), z0=z0.detach(), text=text.detach(), image=image.detach(),
                    image_logits=image_logits.detach(), baseline_logits=self.scorer(z0, text))
