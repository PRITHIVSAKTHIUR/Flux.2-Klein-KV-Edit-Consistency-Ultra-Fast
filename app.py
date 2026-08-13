import os
import gc
import random
import subprocess
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple
import base64
import json
from io import BytesIO

import spaces
import gradio as gr
from gradio import Server
from fastapi.responses import HTMLResponse

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

dtype = torch.bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

ADAPTER = {
    "title": "Klein-Consistency",
    "adapter_name": "klein-consistency",
    "repo": "dx8152/Flux2-Klein-9B-Consistency",
    "weights": "Klein-consistency.safetensors",
}

# --- Patch (required for the KV pipeline class) ---
def apply_patch():
    import diffusers

    site_packages = os.path.dirname(diffusers.__file__)
    patch_file = os.path.join(os.path.dirname(__file__), "flux2_klein_kv.patch")
    if os.path.exists(patch_file):
        result = subprocess.run(
            ["patch", "-p2", "--forward", "--batch"],
            cwd=os.path.dirname(site_packages),
            stdin=open(patch_file),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("Patch applied successfully")
        else:
            print(f"Patch output: {result.stdout}\n{result.stderr}")

apply_patch()

from diffusers.pipelines.flux2.pipeline_flux2_klein_kv import Flux2KleinKVPipeline

# --- Model Loading ---
print("Loading FLUX.2 Klein 9B KV model...")
pipe = Flux2KleinKVPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9b-kv",
    torch_dtype=dtype,
).to(device)
print("Base KV model loaded successfully.")

print(f"Loading adapter: {ADAPTER['title']}")
pipe.load_lora_weights(
    ADAPTER["repo"],
    weight_name=ADAPTER["weights"],
    adapter_name=ADAPTER["adapter_name"],
)
pipe.set_adapters([ADAPTER["adapter_name"]], adapter_weights=[1.0])
print(f"Adapter loaded successfully: {ADAPTER['adapter_name']}")

# ── Examples Config ───────────────────────────────────────────────────────────
EXAMPLES_CONFIG = [
    {"images": ["examples/1.jpg"], "prompt": "Change the weather to stormy."},
    {"images": ["examples/2.jpg"], "prompt": "Transform the scene into a snowy winter day while preserving the original subject identity, framing, and composition."},
    {"images": ["examples/3.jpg"], "prompt": "Relight the image with soft golden sunset lighting while keeping all structures and subject details consistent."},
    {"images": ["examples/4.jpg"], "prompt": "Make the texture high-resolution."},
    {"images": [], "prompt": "A futuristic cyberpunk cityscape at night, neon lights reflecting in puddles, flying cars in the background."},
]

def calc_dimensions(pil_img: Image.Image) -> Tuple[int, int]:
    """Calculates dimensions preserving aspect ratio, snapped to multiples of 8."""
    iw, ih = pil_img.size
    aspect = iw / ih

    if aspect >= 1:
        new_width = 1024
        new_height = int(round(1024 / aspect))
    else:
        new_height = 1024
        new_width = int(round(1024 * aspect))

    new_width = max(256, min(1024, round(new_width / 8) * 8))
    new_height = max(256, min(1024, round(new_height / 8) * 8))
    return new_width, new_height

def make_thumb_b64(path, max_dim=220):
    if not os.path.exists(path):
        return ""
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_dim, max_dim), LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=65)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception as e:
        print(f"Thumbnail error for {path}: {e}")
        return ""

def encode_full_image(path):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            data = f.read()
        ext = path.rsplit(".", 1)[-1].lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception as e:
        print(f"Encode error for {path}: {e}")
        return ""

def build_client_config():
    """Static config consumed by the frontend: example cards."""
    examples = []
    for i, ex in enumerate(EXAMPLES_CONFIG):
        examples.append({
            "idx": i,
            "thumbs": [make_thumb_b64(p) for p in ex["images"]],
            "n_images": len(ex["images"]),
            "prompt": ex["prompt"],
        })
    return {
        "examples": examples,
    }

print("Building client config (example thumbnails)…")
CLIENT_CONFIG = build_client_config()
print(f"Built config with {len(EXAMPLES_CONFIG)} examples.")

def b64_to_pil_list(b64_json_str):
    if not b64_json_str or b64_json_str.strip() in ("", "[]"):
        return []
    try:
        b64_list = json.loads(b64_json_str)
    except Exception:
        return []
    pil_images = []
    for b64_str in b64_list:
        if not b64_str or not isinstance(b64_str, str):
            continue
        try:
            if b64_str.startswith("data:image"):
                _, data = b64_str.split(",", 1)
            else:
                data = b64_str
            image_data = base64.b64decode(data)
            pil_images.append(Image.open(BytesIO(image_data)).convert("RGB"))
        except Exception as e:
            print(f"Error decoding image: {e}")
    return pil_images

def pil_to_b64_png(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

# ── Gradio Server (Server mode): FastAPI + Gradio queue/API engine ────────────
app = Server(title="Flux.2-Klein-KV-Edit-Consistency-Ultra-Fast")

@app.mcp.tool(name="edit_image")
@app.api(name="edit_image")
@spaces.GPU(size="xlarge")
def infer(
    images_b64_json: str,
    prompt: str,
    seed: int,
    randomize_seed: bool,
    width: int,
    height: int,
    steps: int,
) -> dict:
    """Edits an image or generates from text with FLUX.2 Klein 9B KV."""
    gc.collect()
    torch.cuda.empty_cache()

    if not prompt or prompt.strip() == "":
        raise gr.Error("Please enter a prompt.")

    pil_images = b64_to_pil_list(images_b64_json)
    
    if pil_images:
        # Calculate dims from first image and resize all
        calc_w, calc_h = calc_dimensions(pil_images[0])
        width, height = calc_w, calc_h
        processed_images = [
            img.resize((width, height), LANCZOS).convert("RGB") 
            for img in pil_images
        ]
        image_input = processed_images if len(processed_images) > 1 else processed_images[0]
    else:
        image_input = None

    # Ensure dimensions are multiples of 8
    final_width  = max(256, min(1024, round(int(width)  / 8) * 8))
    final_height = max(256, min(1024, round(int(height) / 8) * 8))

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    kwargs = dict(
        prompt=prompt,
        height=final_height,
        width=final_width,
        num_inference_steps=int(steps),
        generator=generator,
    )
    if image_input is not None:
        kwargs["image"] = image_input

    try:
        result_image = pipe(**kwargs).images[0]
        return {"image": pil_to_b64_png(result_image), "seed": seed}
    except Exception as e:
        raise e
    finally:
        gc.collect()
        torch.cuda.empty_cache()


@app.api(name="load_example", queue=False)
def load_example(idx: float) -> dict:
    """Return base64-encoded example images + prompt for a given example index."""
    try:
        i = int(idx)
    except (ValueError, TypeError):
        i = -1
    if i < 0 or i >= len(EXAMPLES_CONFIG):
        return {"images": [], "prompt": "", "names": [], "status": "error"}
    ex = EXAMPLES_CONFIG[i]
    b64_list, names = [], []
    for path in ex["images"]:
        b64 = encode_full_image(path)
        if b64:
            b64_list.append(b64)
            names.append(os.path.basename(path))
    return {"images": b64_list, "prompt": ex["prompt"], "names": names, "status": "ok"}


@app.get("/api/config")
def client_config():
    """Plain FastAPI route: example card data for the frontend."""
    return CLIENT_CONFIG


@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    app.launch(show_error=True, mcp_server=True)