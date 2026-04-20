# Research notes: Depth-aware trajectories & CLIP text re-conditioning for SEVA

These are exploratory notes on two possible "delta" contributions on top of the
current single-image -> NVS -> DUSt3R -> scene pipeline, focusing on the two
synthesizers being used in the final report: ViewCrafter and SEVA
(`seva_synthesizer.py`, a.k.a. Stable Virtual Camera).

---

## 1) Depth-aware trajectory planning -- does it work for SEVA?

**Short answer: yes, it's arguably easier for SEVA than for ViewCrafter**,
because SEVA accepts *arbitrary* per-frame `c2w` and `K` matrices, whereas
ViewCrafter exposes only a small set of delta parameters.

### Where each synthesizer's trajectory enters

ViewCrafter -- offsets only:

```86:93:backend/synthesizers/viewcrafter_synthesizer.py
        d_theta=[d_theta],
        d_phi=[d_phi],
        d_r=[d_r],
        d_x=[0.0],
        d_y=[0.0],
        mask_image=False,
        mask_pc=True,
```

Those are lists (multi-segment) of: azimuth change, elevation change, radial
offset, and planar pan. ViewCrafter's renderer unprojects the input image's
DUSt3R point cloud and moves the camera along these increments, then the
diffusion UNet refines the render. It also accepts a `traj_txt` file for
piecewise trajectories.

SEVA -- full extrinsics:

```193:198:backend/synthesizers/seva_synthesizer.py
        image_cond = {
            "img": all_imgs_path,
            "input_indices": input_indices,
            "prior_indices": anchor_indices,
        }
        camera_cond = {
            "c2w": torch.tensor(c2ws[:, :3]).float().clone(),
            "K": torch.tensor(Ks).float().clone(),
```

`c2ws` is an `(N, 4, 4)` tensor and `Ks` is `(N, 3, 3)`. SEVA's UNet conditions
on Plucker ray maps derived from these per frame -- there is *no* trajectory
preset baked in. The current code just happens to fill it with
`get_preset_pose_fov("orbit", ...)` from `seva/geometry.py`, but it can be
filled with whatever is computed externally.

`get_preset_pose_fov` in
`backend/vendor/stable-virtual-camera/seva/geometry.py` supports `orbit`,
`spiral`, `lemniscate`, `zoom-in/out`, `dolly zoom-in/out`,
`move-{forward,backward,up,down,left,right}`, and `roll` as *built-in* presets
-- all of those are just producing `c2ws` arrays under the hood. A custom
depth-aware path is the same thing.

### What a "depth-aware trajectory planner" means concretely

The pieces are already in the pipeline: DepthPro produces a metric (or at least
well-calibrated relative) depth map and an FOV estimate; the viewer camera
starts at a known identity pose. The planner takes:

- `D(u,v)` -- per-pixel depth
- `fov` -- horizontal FOV (DepthPro gives this)
- `K` -- camera intrinsics built from `fov` and image size
- a target trajectory family (`orbit`, `dolly`, `forward-dolly`, custom)

and produces the `c2ws` sequence.

A reasonable scoring function for a candidate camera pose `P`:

1. **Reconstructable-frustum score** -- unproject image pixels to 3D via `D`,
   project them into `P`, and compute the fraction of the target view that is
   "covered" by warped points (i.e., not disoccluded). Penalize poses where
   coverage drops below some threshold (e.g., <70%).
2. **Disocclusion-severity score** -- measure the size of the largest
   contiguous "hole" in the warped image (the things the NVS model will have
   to hallucinate). Penalize large connected holes because that's where both
   ViewCrafter and SEVA start to make things up and lose 3D consistency.
3. **Depth-edge safety** -- penalize poses that point the camera straight at
   a strong depth discontinuity in the input image; those are where the point
   cloud has the worst artifacts.
4. **Scene scale awareness** -- use median scene depth to set orbit radius /
   `d_r` / `camera_scale`. Currently `seva_synthesizer.py` hard-codes
   `look_at = [0, 0, 10]` and `camera_scale = 2.0`, and
   `viewcrafter_synthesizer.py` hard-codes `d_r = -0.2`. These should scale
   with median depth.

Then a small grid search or gradient-free optimization over trajectory
parameters (e.g., max azimuth, radius, elevation) picks the trajectory that
keeps every frame's score above threshold.

### How this becomes a comparative study

ViewCrafter already contains an "iterative view synthesis + camera trajectory
planning" idea in its paper but the public code mostly exposes only delta
offsets. SEVA doesn't discuss depth-aware trajectory planning at all (it's
designed to take arbitrary user-specified trajectories). So a defensible study
would be:

> We build a depth-aware trajectory planner on top of DepthPro that jointly
> optimizes coverage and disocclusion for single-image NVS. We port the
> planner's output into both ViewCrafter (via `d_phi/d_theta/d_r` segments, or
> `traj_txt`) and SEVA (via raw `c2ws`), and measure downstream gains -- 3D
> self-consistency of the generated views (MVGBench-style), DUSt3R reprojection
> error, and LPIPS on synthetic holdouts.

Two bits of novelty fall out:

1. A *unified* planner that emits both ViewCrafter-offsets and SEVA-extrinsics
   -- nobody's published that.
2. An empirical answer to "does depth-aware trajectory planning help SEVA
   (which is generalist / trajectory-agnostic) as much as it helps ViewCrafter
   (which is point-cloud-guided and therefore more sensitive to disocclusion)?"
   The conventional wisdom would be *ViewCrafter benefits more*, because its
   point cloud explicitly encodes what's reconstructable. Confirming or
   refuting that on the pipeline is a clean result.

### ViewCrafter -> SEVA knowledge transfer

Two concrete avenues for reusing ViewCrafter ideas inside SEVA runs:

- **Reuse ViewCrafter's bundled DUSt3R point cloud** as the geometric prior
  the planner scores against, *even when generating with SEVA*. ViewCrafter
  already runs DUSt3R (`model_path=dust3r_path` in `_build_opts`) to construct
  a scene point cloud from the single image; SEVA doesn't. The planner can
  reuse the same point cloud regardless of which synthesizer consumes the
  trajectory.
- **Port ViewCrafter's iterative view synthesis** (paper section 3.3:
  "iteratively moving cameras, generating novel views, and updating the point
  cloud") to SEVA. SEVA supports up to 32 input views. So: round 1 = generate
  8 views along a safe short trajectory; unproject them; round 2 = feed all 9
  (original + 8) as inputs to SEVA for the next 8 farther poses; repeat.
  That's Voyager-lite / ViewCrafter-iterative, applied to SEVA's multi-input
  mode -- a direct "idea transfer."

---

## 2) CLIP-based text re-conditioning of SEVA -- how it would work

### Why SEVA currently ignores prompts

Here's SEVA's conditioner in full:

```7:39:backend/vendor/stable-virtual-camera/seva/modules/conditioner.py
class CLIPConditioner(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self):
        super().__init__()
        self.module = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k"
        )[0]
        self.module.eval().requires_grad_(False)  # type: ignore
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        x = self.module.encode_image(x)
        return x
```

Two important observations:

1. SEVA uses **OpenCLIP ViT-H-14** (laion2b_s32b_b79k). It's a *dual*
   encoder -- `encode_image` and `encode_text` both output tokens in the *same
   1024-d joint space*. That joint space is what contrastive pretraining
   produced.
2. SEVA's UNet was trained with `encode_image(reference_image)` only. The
   model has never seen a pure text embedding at its cross-attention input.

### The theoretical hook

Because CLIP is a joint vision-language encoder, image embeddings and text
embeddings are approximately interchangeable at cross-attention inputs that
were trained to consume CLIP image embeddings. That's the trick behind Stable
unCLIP, Versatile Diffusion, IP-Adapter, and CAT3D's text conditioning.
Concretely, if the UNet learned to read
`c_img = clip.encode_image(x)`, then feeding it
`c_txt = clip.encode_text(prompt)` (possibly rescaled / renormalized) has a
chance of working zero-shot because `c_txt` and `c_img` lie in approximately
the same distribution. It won't work as well as a model that was trained
jointly on both, but the *direction* of text-induced perturbations tends to be
meaningful.

### Three flavors, from simplest to most involved

**Flavor A -- Training-free CLIP-space arithmetic (CLIP "direction" trick)**

The safest version: never replace the image embedding, just perturb it along a
semantic direction.

1. Pick a **neutral** prompt: e.g. `"a photo of a scene"`.
2. Pick the **target** prompt: e.g. `"at sunset, warm golden light"` or
   `"in heavy rain"`.
3. Compute in CLIP text space:

   `Delta = clip.encode_text(target) - clip.encode_text(neutral)`

4. Replace the conditioner's output with:

   `c_tilde = c_img + lambda * (Delta / ||Delta||) * ||c_img||`

   for some small `lambda` (e.g. 0.05 - 0.3).

Why this is the smart one: `c_img` stays on-manifold for the UNet; we only
nudge it in a direction CLIP associates with the desired semantic. This is
essentially StyleCLIP / text2live / "directional CLIP guidance," applied as a
one-line hook in `CLIPConditioner.forward`. Implementable without training.
Expected effect: mild global style/lighting/weather changes in the
hallucinated (out-of-input-view) regions, without destroying 3D consistency.

**Flavor B -- Weighted blend with the text embedding**

`c_tilde = (1 - alpha) * c_img + alpha * c_txt`

with `alpha` ~ 0.1 - 0.3. Here `c_txt = clip.encode_text(prompt)`. Also
training-free. Riskier than Flavor A because at higher `alpha` the embedding
drifts off the distribution SEVA was trained on, and artifacts appear. But for
small `alpha` it's fine and has the advantage that the prompt can be arbitrary
(no need for a neutral anchor).

Both flavors drop in roughly as:

```python
def forward(self, x, prompt=None, alpha=0.2):
    z_img = self.module.encode_image(self.preprocess(x))
    if prompt is None:
        return z_img
    toks = open_clip.tokenize([prompt]).to(x.device)
    z_txt = self.module.encode_text(toks)
    return (1 - alpha) * z_img + alpha * z_txt
```

`prompt` would then be piped through `run_one_scene` via `image_cond` or a
small signature change.

**Flavor C -- Lightweight adapter (the CAT3D / IP-Adapter path)**

Train a small MLP (~a few M params) that maps `c_txt -> c_img_like`, using
paired captioned images. The MLP's job is to bring pure text embeddings onto
the image-embedding manifold SEVA was trained on. Then at inference:

`c_tilde = c_img + MLP(c_txt)`

This is the pattern IP-Adapter uses in reverse. Needs a training loop, some
paired data (any scene image with a descriptive caption would do; captions
could be generated with a VLM). Would likely give the best quality but is the
most work and has a real risk of not converging in final-project time.

### What's realistic as a delta

Flavor A as a **2-3 day experiment** is very doable and is a real
contribution:

1. Modify `CLIPConditioner` (or wrap it in a subclass) to accept a `prompt`
   and a `lambda`.
2. Thread `prompt` through `SevaSynthesizer.generate_views`. `app.py` already
   accepts it and logs it; lines 112-113 of `seva_synthesizer.py` just warn
   and ignore.
3. Compare generations with `prompt=None` vs `prompt="..."` qualitatively,
   and on self-consistency metrics (does the prompt change break 3D
   consistency?).
4. Ablate `lambda` in {0.05, 0.1, 0.2, 0.3, 0.5}.

This gives a clean, honest finding -- either "CLIP directional guidance
usefully controls SEVA's hallucinated regions without breaking consistency up
to lambda approx X" or "SEVA is too tightly tuned to image embeddings and
breaks down past lambda approx X." Both results are
publishable-in-a-final-report interesting, and the code change is tiny.

### One gotcha

SEVA uses `open_clip`'s ViT-H-14 with `laion2b_s32b_b79k`. Text tokenization
and the text head for that specific checkpoint need to come from the *same*
`open_clip.create_model_and_transforms` call -- don't mix with HuggingFace
`transformers.CLIPTextModel` of a different checkpoint, or the joint space
won't actually be joint. The conditioner already loads the full dual-encoder
model (`open_clip.create_model_and_transforms(...)` returns the full model),
so `self.module.encode_text` is already available, it just isn't called.
That's the one-line fix that unlocks the whole thing.

---

## Summary

- **(1)** is not just possible for SEVA -- SEVA's API is *more* amenable to
  depth-aware trajectory planning than ViewCrafter's (arbitrary `c2ws` vs
  fixed offset lists). ViewCrafter's bundled DUSt3R point cloud and
  iterative-refinement idea can be reused to plan safer, wider SEVA
  trajectories, and the comparison between "does this help ViewCrafter more
  or SEVA more" is itself the study.
- **(2)** works because SEVA's `CLIPConditioner` loads the full OpenCLIP
  ViT-H-14 dual encoder but only ever calls `encode_image`. Adding a
  training-free CLIP-direction or blended text-embedding path is a ~10-line
  modification to `conditioner.py` that turns `prompt` from "ignored" into
  "controls hallucinated regions." Flavor A (CLIP directional guidance) is
  the low-risk / high-information version; Flavor C (adapter) is more
  powerful but likely not a final-project-timescale bet.
