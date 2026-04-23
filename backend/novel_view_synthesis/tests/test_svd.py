import time
import sys
import logging
from novel_view_synthesis.svd_synthesizer import SVDSynthesizer

logging.basicConfig(level=logging.INFO)

def main():
    print("Initializing SVD Synthesizer...")
    synth = SVDSynthesizer()
    
    print("Loading model onto GPU (this may take a moment)...")
    synth.load_model(device='cuda')
    print("Model loaded successfully!")
    
    print("Starting video generation from 'test_rocket.png'...")
    start_time = time.time()
    
    try:
        views = synth.generate_views(image_path='test_rocket.png', output_dir='test_output')
        end_time = time.time()
        print(f"\nSuccess! Generated {len(views)} views in {end_time - start_time:.2f} seconds.")
        print("Generated files:")
        for view in views:
            print(f" - {view}")
    except Exception as e:
        print(f"\nError during generation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
