GitHub link: https://github.com/hauha46/image-explorer

# For Graders:

For inference, we use SVC, found here: https://huggingface.co/stabilityai/stable-virtual-camera.

This is a gated model, meaning you will need to have an active huggingface token in order to access it. While other models don't require this, 
we chose to use SVC as it gives the best results.

# Requirements

To run the demo_pipeline.py file, which is one inference run, you will need at least 12 GB of VRAM. This will produce results but they will not be as nice
as what we have in our report and presentation because we used an A100 for those.


# Setup (Also for Graders)
First, run this command to install the necessary vendor repositories, model checkpoints, and Python dependencies:

```bash
bash setup.sh
```

This will download 9 models that were used in making our pipeline. During execution of our pipeline, only 3 are used; however, the other models were used in testing.

Second, ensure your Python packages are synced:

```bash
uv sync
```

Third, run the demo pipeline to see the end-to-end inference results (image → SEVA novel views → DUSt3R reconstruction → BPA meshing):

```bash
uv run python demo_pipeline.py
```

The default input image is `benchmark_image_1.jpg`. To use the default image:
```bash
uv run python demo_pipeline.py
```

To use a different image:

```bash
uv run python demo_pipeline.py --image path/to/your_image.jpg
```

Output will be saved to `final_demo_outputs/` at the repo root:
- `final_demo_outputs/views/_seva_work/samples-rgb/` — generated novel views
- `final_demo_outputs/reconstruction/scene.glb` — DUSt3R point cloud
- `final_demo_outputs/reconstruction/scene_mesh.glb` — BPA mesh

If you want to see the frontend (note: the frontend is not connected to the pipeline due to breaking code changes):

```bash
bash run.sh
```





# Image Explorer

This project uses `uv` for dependency management and integrates the Dust3r model for 3D reconstruction from images.

## Installation Guide

Follow these steps to set up the project environment and download the necessary submodules:

### 1. Initialize Python Environment

We use `uv` to manage the Python environment and dependencies from the root directory.

```bash
# Initialize uv (if not already done)
uv init

# Install dependencies and sync the environment
uv sync

# Make the run script executable
chmod +x run.sh
```

### 2. Set Up Vendor Directory and Dust3r

The `dust3r` repository needs to be cloned into the `backend/vendor` directory, and its submodules (like `croco`) must be initialized.

```bash
# Create the vendor directory
mkdir -p backend/vendor

# Clone the dust3r repository into the vendor directory
git clone https://github.com/naver/dust3r.git backend/vendor/dust3r

# Initialize the croco submodule required by dust3r
cd backend/vendor/dust3r
git submodule update --init --recursive
cd ../../../
```

### 3. Setup Depth Pro

The `ml-depth-pro` repository needs to be cloned into the `backend/vendor` directory and added as a dependency using `uv`.

```bash
# Clone the ml-depth-pro repository into the vendor directory
git clone https://github.com/apple/ml-depth-pro.git backend/vendor/ml-depth-pro

# Add ml-depth-pro as a local dependency
uv add ./backend/vendor/ml-depth-pro
```

### 4. Setup Input and Output Directories

Create the directories where your input images will be placed and where the resulting 3D models will be saved.

```bash
mkdir -p backend/images
mkdir -p backend/uploads/test_scene
```

## Usage

To run the Dust3r reconstructor, place at least two overlapping images of a scene/object in the `backend/images` folder.

Then, execute the reconstruction script from the main project directory:

```bash
uv run python backend/dust3r_reconstructor.py --images_dir backend/images --output_dir backend/uploads/test_scene
```

Once complete, the reconstructed 3D point cloud will be saved as `scene.glb` in your `backend/uploads/test_scene` folder.