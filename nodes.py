import os

import folder_paths
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from diffusers import DDIMScheduler
from torchvision.transforms import ToTensor

from .pipelines import (
    StableDiffusionAOVMatEstPipeline,
    X2RGBInpaintingPipeline,
    X2RGBPipeline,
)

# Register model directory under ComfyUI/models/rgbx/
RGBX_MODELS_DIR = os.path.join(folder_paths.models_dir, "rgbx")
os.makedirs(RGBX_MODELS_DIR, exist_ok=True)

# Maps pipeline key -> HuggingFace model name
MODEL_NAMES = {
    "rgb2x": "rgb-to-x",
    "x2rgb": "x-to-rgb",
    "x2rgb_inpainting": "x-to-rgb-inpainting",
}

PIPELINE_CLASSES = {
    "rgb2x": StableDiffusionAOVMatEstPipeline,
    "x2rgb": X2RGBPipeline,
    "x2rgb_inpainting": X2RGBInpaintingPipeline,
}

# Cache loaded pipelines to avoid reloading on each execution
_pipeline_cache = {}


def get_model_path(model_name):
    """Return local path if model is pre-downloaded, otherwise HuggingFace ID."""
    local_path = os.path.join(RGBX_MODELS_DIR, model_name)
    if os.path.isdir(local_path):
        return local_path
    # Fallback to HuggingFace ID (requires internet)
    return f"zheng95z/{model_name}"


def load_pipeline(pipeline_key):
    """Load and cache a diffusion pipeline."""
    if pipeline_key in _pipeline_cache:
        return _pipeline_cache[pipeline_key]

    model_name = MODEL_NAMES[pipeline_key]
    pipeline_class = PIPELINE_CLASSES[pipeline_key]
    model_path = get_model_path(model_name)

    pipe = pipeline_class.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
    )
    pipe.set_progress_bar_config(disable=True)

    _pipeline_cache[pipeline_key] = pipe
    return pipe


def prep_material_map(img, from_srgb=False, normalize=False, clamp=False):
    """Convert a ComfyUI IMAGE (BHWC) to pipeline format (CHW) with appropriate transforms."""
    if img is None:
        return None
    t = img[0].permute(2, 0, 1)  # (C, H, W)
    if from_srgb:
        t = t ** 2.2
    if clamp:
        t = torch.clamp(t, 0.0, 1.0)
    if normalize:
        t_hwc = t.permute(1, 2, 0)
        t_hwc = t_hwc * 2.0 - 1.0
        t_hwc = F.normalize(t_hwc, dim=-1, eps=1e-6)
        t = t_hwc.permute(2, 0, 1)
    return t.to("cuda")


def get_resolution_from_inputs(*tensors, default_h=768, default_w=768):
    """Get height/width from the first non-None tensor."""
    for t in tensors:
        if t is not None:
            return t.shape[1], t.shape[2]
    return default_h, default_w


AOV_PROMPTS = {
    "albedo": "Albedo (diffuse basecolor)",
    "normal": "Camera-space Normal",
    "roughness": "Roughness",
    "metallic": "Metallicness",
    "irradiance": "Irradiance (diffuse lighting)",
}


class RGB2X:
    """Decompose an RGB photo into intrinsic channels (albedo, normal, roughness, metallic, irradiance)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "aov": (["albedo", "normal", "roughness", "metallic", "irradiance"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
            },
            "optional": {
                "max_side": ("INT", {"default": 1000, "min": 256, "max": 4096, "step": 8}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "rgbx"

    def execute(self, image, aov, seed, steps, max_side=1000):
        pipe = load_pipeline("rgb2x")

        # BHWC -> CHW, first batch only
        photo = image[0].permute(2, 0, 1)  # (C, H, W)

        # sRGB to linear
        photo = photo ** 2.2

        # Resize to fit max_side while preserving aspect ratio
        old_height, old_width = photo.shape[1], photo.shape[2]
        aspect_ratio = old_height / old_width

        if old_height > old_width:
            new_height = max_side
            new_width = int(new_height / aspect_ratio)
        else:
            new_width = max_side
            new_height = int(new_width * aspect_ratio)

        # Round to multiple of 8 (required by VAE)
        new_width = new_width // 8 * 8
        new_height = new_height // 8 * 8

        photo = torchvision.transforms.Resize((new_height, new_width))(photo)

        generator = torch.Generator(device="cuda").manual_seed(seed)

        generated = pipe(
            prompt=AOV_PROMPTS[aov],
            photo=photo.to("cuda"),
            num_inference_steps=steps,
            height=new_height,
            width=new_width,
            generator=generator,
            required_aovs=[aov],
        ).images[0][0]

        # PIL -> tensor, resize back to original dimensions
        result = ToTensor()(generated)  # (C, H, W)
        result = torchvision.transforms.Resize((old_height, old_width))(result)

        # CHW -> BHWC
        result = result.permute(1, 2, 0).unsqueeze(0)

        return (result,)


class X2RGB:
    """Generate a realistic RGB image from intrinsic material maps."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "image_guidance_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "albedo": ("IMAGE",),
                "normal": ("IMAGE",),
                "roughness": ("IMAGE",),
                "metallic": ("IMAGE",),
                "irradiance": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "rgbx"

    def execute(self, seed, steps, guidance_scale, image_guidance_scale,
                albedo=None, normal=None, roughness=None, metallic=None,
                irradiance=None, prompt=""):
        pipe = load_pipeline("x2rgb")

        albedo_t = prep_material_map(albedo, from_srgb=True)
        normal_t = prep_material_map(normal, normalize=True)
        roughness_t = prep_material_map(roughness, clamp=True)
        metallic_t = prep_material_map(metallic, clamp=True)
        irradiance_t = prep_material_map(irradiance, from_srgb=True, clamp=True)

        height, width = get_resolution_from_inputs(
            albedo_t, normal_t, roughness_t, metallic_t, irradiance_t
        )

        generator = torch.Generator(device="cuda").manual_seed(seed)

        result = pipe(
            height=height,
            width=width,
            prompt=prompt,
            albedo=albedo_t,
            normal=normal_t,
            roughness=roughness_t,
            metallic=metallic_t,
            irradiance=irradiance_t,
            num_inference_steps=steps,
            generator=generator,
            required_aovs=["albedo", "normal", "roughness", "metallic", "irradiance"],
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            guidance_rescale=0.7,
            output_type="np",
        ).images[0]

        # numpy (H, W, C) [0,1] -> torch BHWC
        result = torch.from_numpy(result).unsqueeze(0).float()

        return (result,)


class X2RGBInpainting:
    """Generate a realistic RGB image from material maps with inpainting support."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "photo": ("IMAGE",),
                "mask": ("MASK",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "image_guidance_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "albedo": ("IMAGE",),
                "normal": ("IMAGE",),
                "roughness": ("IMAGE",),
                "metallic": ("IMAGE",),
                "irradiance": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("generated", "masked_photo", "photo_vae")
    FUNCTION = "execute"
    CATEGORY = "rgbx"

    def execute(self, photo, mask, seed, steps, guidance_scale, image_guidance_scale,
                albedo=None, normal=None, roughness=None, metallic=None,
                irradiance=None, prompt=""):
        pipe = load_pipeline("x2rgb_inpainting")

        albedo_t = prep_material_map(albedo, from_srgb=True)
        normal_t = prep_material_map(normal, normalize=True)
        roughness_t = prep_material_map(roughness, clamp=True)
        metallic_t = prep_material_map(metallic, clamp=True)
        irradiance_t = prep_material_map(irradiance, from_srgb=True, clamp=True)

        # Photo: BHWC sRGB -> CHW linear
        photo_t = photo[0].permute(2, 0, 1).to("cuda")
        photo_t = photo_t ** 2.2

        # Mask: ComfyUI MASK is (B, H, W) where 1.0 = area to inpaint
        # Pipeline expects inverted: 1.0 = keep, 0.0 = inpaint
        mask_t = mask[0].unsqueeze(0).to("cuda")  # (1, H, W)
        mask_t = 1.0 - mask_t

        masked_photo_t = photo_t * mask_t

        # Use photo dimensions by default, override with material map dims if provided
        height, width = photo_t.shape[1], photo_t.shape[2]
        for t in [albedo_t, normal_t, roughness_t, metallic_t, irradiance_t]:
            if t is not None:
                height, width = t.shape[1], t.shape[2]
                break

        generator = torch.Generator(device="cuda").manual_seed(seed)

        res = pipe(
            height=height,
            width=width,
            prompt=prompt,
            albedo=albedo_t,
            normal=normal_t,
            roughness=roughness_t,
            metallic=metallic_t,
            irradiance=irradiance_t,
            mask=mask_t,
            masked_image=masked_photo_t,
            photo=photo_t,
            num_inference_steps=steps,
            generator=generator,
            required_aovs=["albedo", "normal", "roughness", "metallic", "irradiance"],
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            guidance_rescale=0.7,
            output_type="np",
        ).images

        generated = torch.from_numpy(res[0][0]).unsqueeze(0).float()
        masked_photo_out = torch.from_numpy(res[1][0]).unsqueeze(0).float()
        photo_vae_out = torch.from_numpy(res[2][0]).unsqueeze(0).float()

        return (generated, masked_photo_out, photo_vae_out)


class CombineMetallicRoughness:
    """Combines separate metallic and roughness image batches into a single
    MR image batch compatible with the Hy3DBakeMultiViews node.

    The output packs metallic into the red channel and roughness into the
    green channel (blue is zeroed out), matching the convention used by
    Hunyuan 3D 2.1.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metallic": ("IMAGE",),
                "roughness": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mr",)
    FUNCTION = "combine"
    CATEGORY = "rgbx"

    def combine(self, metallic, roughness):
        # metallic / roughness: [B, H, W, C] float32 tensors in [0, 1]
        # If the inputs are RGB, take just the first channel (grayscale value).
        if metallic.shape[-1] >= 3:
            m = metallic[..., 0:1]
        else:
            m = metallic

        if roughness.shape[-1] >= 3:
            r = roughness[..., 0:1]
        else:
            r = roughness

        b = torch.zeros_like(m)

        # Pack: R=metallic, G=roughness, B=0
        mr = torch.cat([m, r, b], dim=-1)

        return (mr,)


NODE_CLASS_MAPPINGS = {
    "RGB2X": RGB2X,
    "X2RGB": X2RGB,
    "X2RGBInpainting": X2RGBInpainting,
    "CombineMetallicRoughness": CombineMetallicRoughness,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RGB2X": "RGB -> X (Intrinsic Decomposition)",
    "X2RGB": "X -> RGB (Material to Photo)",
    "X2RGBInpainting": "X -> RGB Inpainting",
    "CombineMetallicRoughness": "Combine Metallic + Roughness to MR",
}
