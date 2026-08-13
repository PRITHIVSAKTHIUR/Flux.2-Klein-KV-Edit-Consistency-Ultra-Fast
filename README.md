# **Flux.2-Klein-KV-Edit-Consistency-Ultra-Fast**

Flux.2-Klein-KV-Edit-Consistency-Ultra-Fast is a high-performance image editing and generation workspace based on the `black-forest-labs/FLUX.2-klein-9b-kv` base model and the `dx8152/Flux2-Klein-9B-Consistency` LoRA adapter. The pipeline uses a custom diffusers patch (`flux2_klein_kv.patch`) to enable key-value attention consistency mechanisms, allowing fast, structure-preserving image-to-image edits and high-fidelity text-to-image generation in 4-step sampling passes.

The application operates via a FastAPI server (`gradio.Server`) hosting a dark crimson single-page web app (SPA) that features a dual-view canvas, an A/B comparison slider, input and history filmstrips, and quick prompt chips.

### **Key Features**

* **KV Attention Consistency Engine:** Integrates a core structural patch (`flux2_klein_kv.patch`) over local `diffusers` modules via subprocess initialization to enable Key-Value consistency mechanisms in FLUX.2.
* **Klein-Consistency LoRA Adapter:** Pre-loads the `dx8152/Flux2-Klein-9B-Consistency` adapter at a full weight scale ($1.0$) to guarantee structural and compositional identity during editing.
* **Dual-Mode Inference (I2I & T2I):** Performs structure-guided image editing when input images are provided, or falls back to text-to-image generation when the gallery is empty.
* **Studio SPA Interface:** An interactive web app built with modern vanilla web components, offering an A/B comparison slider, history filmstrip, quick prompt chips, and drag-and-drop file support.
* **Smart Aspect Ratio Snapping:** Calculates dimensions from the first input image, scaling parameters to fit within a 1024px boundary while snapping width and height to multiples of 8.

### **Repository Structure**

```text
├── examples/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   └── 4.jpg
├── app.py
├── flux2_klein_kv.patch
├── index.html
├── LICENSE.txt
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```

### **Installation and Requirements**

To configure Flux.2-Klein-KV-Edit-Consistency-Ultra-Fast locally, set up your system according to the specifications below. A modern CUDA-enabled GPU is required.

* **Python Version:** Minimum Python **3.10.13** or above is required; Python **3.12** or **3.14** is recommended.
* **PyTorch Version:** `torch==2.11.0` or above is required for optimal system compatibility.
* **CUDA Version:** **CUDA 13.0** is recommended (`--extra-index-url [https://download.pytorch.org/whl/cu130](https://download.pytorch.org/whl/cu130)`), matching the environment running on the live Hugging Face demo.

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust. It ensures rapid virtual environment setup and exact dependency synchronization based on the `uv.lock` file.

**Step 1 — Install `uv`**

* **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/Flux.2-Klein-KV-Edit-Consistency-Ultra-Fast.git
cd Flux.2-Klein-KV-Edit-Consistency-Ultra-Fast

```

**Step 3 — Initialize the project and install dependencies**

```bash
uv sync
```

**Step 4 — Run the script**

```bash
uv run app.py
```

#### **Standard PIP Implementation**

**1. Update Package Manager**
Upgrade your local package manager:

```bash
pip install pip>=26.1.2
```

**2. Install Core Dependencies**
Install the primary deep learning stack, transformer libraries, and core computing utilities listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### **Core Requirements List (`requirements.txt`)**

```text
--extra-index-url https://download.pytorch.org/whl/cu130
torch==2.11.0
torchvision==0.26.0
transformers==5.14.1
accelerate==1.14.0
diffusers==0.39.0
peft==0.19.1
gradio==6.22.0
av==17.1.0
spaces==0.51.1
huggingface-hub==1.24.0
```

### **Usage**

Once the web server initializes, open your browser to the local address output in your terminal (typically `http://127.0.0.1:7860/`).

1. **Upload Asset (Optional):** Drag and drop an image into the main canvas workspace, paste an image from your clipboard, or click the upload icon in the left rail. Leave empty for text-to-image generation.
2. **Refine Instructions:** Type your instructions inside the prompt field, or click one of the **Quick Prompts** chips to instantly fill it. Press ⌘/Ctrl + Enter or click **Edit Image**.
3. **Compare & Chain:** Use the **Compare** tool on the left rail to view an A/B slider of the before and after states. Click **Use as Input** to chain multiple edits sequentially.

### **License and Source**

* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/Flux.2-Klein-KV-Edit-Consistency-Ultra-Fast/blob/main/LICENSE.txt)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Flux.2-Klein-KV-Edit-Consistency-Ultra-Fast](https://github.com/PRITHIVSAKTHIUR/Flux.2-Klein-KV-Edit-Consistency-Ultra-Fast)
* **Hugging Face Live Space:** [https://huggingface.co/spaces/prithivMLmods/Flux.2-Klein-KV-Edit-Consistency-Ultra-Fast](https://huggingface.co/spaces/prithivMLmods/Flux.2-Klein-KV-Edit-Consistency-Ultra-Fast)
