import time
import sys
import logging
from novel_view_synthesis.sv3d_synthesizer_4070ti import SV3DSynthesizer
import traceback


logging.basicConfig(level=logging.INFO)

def main():
    print("Initializing SV3D Synthesizer...")
    synth = SV3DSynthesizer()
    
    print("Loading model onto GPU (this may take a moment)...")
    synth.load_model(device='cuda')
    print("Model loaded successfully!")
    
    print("Starting 360-degree panning video generation from 'test_rocket.png'...")
    start_time = time.time()
    
    try:
        views = synth.generate_views(image_path='room.jpg', output_dir='test_sv3d_output')
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
