# For Graders:


First, run this command to install the necessary and relevant models to your system.
This will download 9 models that were used in making our pipeline. During execution of 
our pipeline, only 3 are used, however the other models were used in testing.

<command

Second, run uv sync in order to get up to date packages:

< uv sync

Third, run chmod on the run.sh in order to be able to access the frontend.





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