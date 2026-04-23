# Sweep run `living_room__sunset__l0p000`

- **input**: `backend/images/standard_benchmark.jpg`
- **prompt**: 'a living room at sunset with warm golden hour light'
- **neutral_prompt**: 'a photo of a living room interior'
- **lambda**: 0.0
- **seed**: 23
- **cfg**: 4.0
- **num_views**: 10
- **num_steps**: 25
- **dtype**: bf16
- **model_version**: seva-1.1
- **git_sha**: `a940dcfb922a9505a7cbdf8951d6964205b827c7`

## Reproduce

```bash
python -m backend.experiments.clip_recond_sweep --inputs 'backend/images/benchmark_image_1.jpg' --input-labels 'living_room' --extra-prompt 'sunset' 'a living room at sunset with warm golden hour light' --lambdas 0.0 --neutral-prompt 'a photo of a living room interior' --num-views 10 --num-steps 25 --dtype bf16 --seed 23
```

All artifacts in this folder were produced by
`backend/experiments/clip_recond_sweep.py`.