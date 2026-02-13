# ComfyUI RGB-X

ComfyUI nodes for [RGB-X](https://github.com/zheng95z/rgbx) — image decomposition and generation using intrinsic channels.

> **Difference with [ComfyUI_rgbx_Wrapper](https://github.com/zheng95z/ComfyUI_rgbx_Wrapper)?**
> The original wrapper only exposed 1 of the 3 RGB-X pipelines (older version of the init repo that has been updated since).

Provides three nodes:

| Node | Description |
|------|-------------|
| **RGB -> X** | Decompose an RGB photo into material channels (albedo, normal, roughness, metallic, irradiance) |
| **X -> RGB** | Generate a realistic photo from material channels |
| **X -> RGB Inpainting** | Generate a photo from material channels with mask-based inpainting |

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/leob03/ComfyUI-rgbx.git
```

### Dependencies

The pipeline files require `diffusers`, `transformers`, and `torchvision`. These are typically already installed in a ComfyUI environment. If not:

```bash
pip install diffusers>=0.20.0 transformers torchvision
```

## Model Setup

### Models

| Pipeline | HuggingFace Model ID | Local Directory |
|----------|---------------------|-----------------|
| RGB -> X | `zheng95z/rgb-to-x` | `ComfyUI/models/rgbx/rgb-to-x/` |
| X -> RGB | `zheng95z/x-to-rgb` | `ComfyUI/models/rgbx/x-to-rgb/` |
| X -> RGB Inpainting | `zheng95z/x-to-rgb-inpainting` | `ComfyUI/models/rgbx/x-to-rgb-inpainting/` |

Config files (model_index.json, scheduler configs, tokenizer vocab, etc.) are bundled in this repo under `configs/`. Only the large `.safetensors` weight files need to be placed in the model directories.

### Downloading weights for offline use

Only the `.safetensors` weight files are needed. The directory structure should look like:

```
ComfyUI/models/rgbx/rgb-to-x/
  text_encoder/model.safetensors
  unet/diffusion_pytorch_model.safetensors
  vae/diffusion_pytorch_model.safetensors
```

```bash
# Install huggingface-hub CLI if needed
pip install huggingface-hub

# Download only safetensors files from your ComfyUI root directory
huggingface-cli download zheng95z/rgb-to-x --include "*.safetensors" --local-dir models/rgbx/rgb-to-x
huggingface-cli download zheng95z/x-to-rgb --include "*.safetensors" --local-dir models/rgbx/x-to-rgb
huggingface-cli download zheng95z/x-to-rgb-inpainting --include "*.safetensors" --local-dir models/rgbx/x-to-rgb-inpainting
```

Alternatively, downloading the full repo (configs + weights) still works — the node detects the layout automatically.

### Model resolution behavior

At runtime, the node looks for weights in `<ComfyUI>/models/rgbx/<model-name>/`. If `model_index.json` is present in that directory (full HuggingFace layout), it is used directly. Otherwise, the node merges the bundled configs with the weight files via symlinks into a staging directory (`_staged/`). If the local directory does not exist at all, it falls back to downloading from HuggingFace.

## Node Details

### RGB -> X (Intrinsic Decomposition)

Decomposes a photo into one of five intrinsic channels.

**Inputs:**
- `image` — Input RGB photo
- `aov` — Which channel to extract: `albedo`, `normal`, `roughness`, `metallic`, or `irradiance`
- `seed` — Random seed for reproducibility
- `steps` — Number of inference steps (default: 50)
- `max_side` (optional) — Maximum dimension for processing (default: 1000). The image is downscaled to this size for inference, then the result is **upscaled back to the original resolution**.

**Output:** Single IMAGE at the same resolution as the input.

### X -> RGB (Material to Photo)

Generates a realistic photo from any combination of material maps.

**Inputs:**
- `seed`, `steps`, `guidance_scale`, `image_guidance_scale` — Generation parameters
- `albedo`, `normal`, `roughness`, `metallic`, `irradiance` (all optional) — Material channel images
- `prompt` (optional) — Text prompt to guide generation

**Output:** Single IMAGE.

### X -> RGB Inpainting

Generates a photo with inpainting — fill in masked regions guided by material maps.

**Inputs:**
- `photo` — Reference photo
- `mask` — Inpainting mask (white = area to inpaint)
- `seed`, `steps`, `guidance_scale`, `image_guidance_scale` — Generation parameters
- `albedo`, `normal`, `roughness`, `metallic`, `irradiance` (all optional) — Material channel images
- `prompt` (optional) — Text prompt to guide generation

**Outputs:**
- `generated` — The generated image
- `masked_photo` — The masked photo (VAE decoded)
- `photo_vae` — The reference photo (VAE decoded)

## Requirements

- GPU with at least 12GB VRAM
- CUDA-capable PyTorch installation
- Python >= 3.9

## Credits

Based on the [RGB-X](https://github.com/zheng95z/rgbx) research project.
