import time
import sys
import logging
import traceback
from backend.synthesizers.svc_synthesizer import SVCSynthesizer

import os

logging.basicConfig(level=logging.INFO)

def main():
    print("Initializing SVC (Stable Virtual Camera) Synthesizer...")
    synth = SVCSynthesizer()
    
    print("Loading model onto GPU (this may take a moment)...")
    try:
        synth.load_model(device='cuda')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"\nFailed to load model. Ensure SEVA is installed and huggingface-cli is authenticated: {e}", file=sys.stderr)
        traceback.print_exc()
        return

    print("Starting novel view generation from 'room.jpg'...")
    start_time = time.time()
    
    try:
        image_file = os.path.join(os.path.dirname(__file__), '../standard_benchmark.jpg')
        
        # Utilizing the existing 'room.jpg' available in backend/synthesizers/
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
