# ComfyUI RGB-X

ComfyUI nodes for [RGB-X](https://github.com/zheng95z/rgbx) — image decomposition and generation using intrinsic channels.

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

### Downloading models for offline use


```bash
# Install huggingface-hub CLI if needed
pip install huggingface-hub

# Run these from your ComfyUI root directory (where main.py is)
huggingface-cli download zheng95z/rgb-to-x --local-dir models/rgbx/rgb-to-x
huggingface-cli download zheng95z/x-to-rgb --local-dir models/rgbx/x-to-rgb
huggingface-cli download zheng95z/x-to-rgb-inpainting --local-dir models/rgbx/x-to-rgb-inpainting
```

Or using Python:

```python
from huggingface_hub import snapshot_download

COMFYUI_ROOT = "/path/to/ComfyUI"

for model in ["rgb-to-x", "x-to-rgb", "x-to-rgb-inpainting"]:
    snapshot_download(
        repo_id=f"zheng95z/{model}",
        local_dir=f"{COMFYUI_ROOT}/models/rgbx/{model}",
    )
```

### Model resolution behavior

At runtime, the node looks for models in `<ComfyUI>/models/rgbx/<model-name>/` (resolved via `folder_paths.models_dir`). If the local directory does not exist, it falls back to downloading from HuggingFace.

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
