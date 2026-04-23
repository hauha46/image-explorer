import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
import sys
from pathlib import Path

# Add backend/ to Python path so 'novel_view_synthesis' is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging
import traceback
from novel_view_synthesis.seva_synthesizer_4070ti import SevaSynthesizer

logging.basicConfig(level=logging.INFO)

def main():
    print("Initializing SVC (Stable Virtual Camera) Synthesizer...")
    synth = SevaSynthesizer()
    
    print("Loading model onto GPU (this may take a moment)...")
    try:
        synth.load_model(device='cuda')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"\nFailed to load model. Ensure SEVA is installed and huggingface-cli is authenticated: {e}", file=sys.stderr)
        traceback.print_exc()
        return

    print("Starting novel view generation")
    start_time = time.time()
    
    try:
        image_file = os.path.join(os.path.dirname(__file__), '../../../benchmark_image_1.jpg')
        
        # Utilizing the existing 'room.jpg' available in backend/novel_view_synthesis/
        views = synth.generate_views(
            image_path=image_file, 
            output_dir='test_svc_output',
            num_views=8
        )
        end_time = time.time()
        print(f"\nSuccess! Generated {len(views)} views in {end_time - start_time:.2f} seconds.")
        print("Generated files:")
        for view in views:
            print(f" - {view}")
    except Exception as e:
        print(f"\nError during generation: {e}", file=sys.stderr)
        traceback.print_exc()

if __name__ == '__main__':
    main()
