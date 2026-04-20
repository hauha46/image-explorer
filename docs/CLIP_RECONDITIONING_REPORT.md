# CLIP Text Re-Conditioning for SEVA — Project Write-up

A faithful write-up of the "CLIP text re-conditioning" delta I added on top of
the single-image → NVS → DUSt3R → 3D-scene pipeline. Grounded in the code that
is actually in the repo (`backend/vendor/stable-virtual-camera/seva/modules/conditioner.py`,
`backend/synthesizers/seva_synthesizer.py`, `backend/experiments/clip_recond_sweep.py`,
`backend/experiments/metrics.py`, `backend/app.py`), with references back to
the exploratory notes in `docs/RESEARCH_NOTES.md`.

---

## TL;DR (layman version, for friends)

SEVA is a diffusion model that takes one photo of a room and hallucinates what
it looks like from other camera angles, and internally it conditions on a CLIP
"summary vector" of the input image — but even though the codebase accepted a
text prompt, SEVA was just throwing it away. My delta hooks into that
conditioning step so the prompt actually does something: I encode both the
target prompt (`"a living room at sunset"`) and a neutral reference prompt
(`"a photo of a living room interior"`) with the same CLIP text encoder, take
the difference between them to get a direction that means "sunset-ness minus
generic-room-ness," and nudge the image's CLIP vector a small amount (λ) along
that direction before it goes into the model. Because I only *shift* the image
embedding instead of replacing it, SEVA's novel-view generation still looks
like your actual room from the right angles, but the parts it has to make up —
lighting, the stuff outside the window, overall mood — pick up the vibe of the
prompt. This is the same "directional CLIP edit" trick from the StyleGAN-NADA
paper, just applied to a 3D-view model instead of a GAN, and it's completely
training-free — about 150 lines of code. I then ran a sweep across a grid of
prompts (sunset, rain, cyberpunk, etc.) and strengths (λ from 0 to 0.3) and
measured three things: whether the generated views actually match the prompt
(CLIP score), how far they drift from the original photo (LPIPS), and whether
the 3D geometry stays consistent across views (a DUSt3R-based check) — so I
can quantify how hard you can steer SEVA with text before it stops looking
like a coherent 3D scene.

---

## 1. Background — what SEVA did before the delta

The pipeline is:

> single image → DepthPro (depth + FOV) → Novel View Synthesis (NVS) → DUSt3R
> (point cloud / GLB) → mesh.

For the NVS stage, **Stable Virtual Camera (SEVA)** is a 1.3-B-parameter
generalist multi-view diffusion model. Its UNet receives two streams of
conditioning per frame:

1. A per-view **Plücker ray map** derived from camera extrinsics `c2w` and
   intrinsics `K` — this is how SEVA is told "render the scene from *this*
   pose."
2. A **pooled CLIP image embedding** of the reference view, injected via
   cross-attention. This is produced by `CLIPConditioner` in
   `backend/vendor/stable-virtual-camera/seva/modules/conditioner.py`.

Critically, the upstream `CLIPConditioner.forward` only ever calls
`encode_image`:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.preprocess(x)
    x = self.module.encode_image(x)
    return x
```

So while `/process` accepts a `prompt` form field in `app.py`, SEVA
historically ignored it — the prompt was logged and dropped. That is the gap
the delta closes.

## 2. Why this is possible — CLIP's shared image/text space

SEVA loads OpenCLIP **ViT-H-14** with the `laion2b_s32b_b79k` checkpoint.
That's a *dual encoder*: `encode_image` and `encode_text` produce 1024-D
vectors in the **same joint space**, because CLIP was contrastively trained
to maximize cosine similarity between paired (image, caption) embeddings
(Radford et al., *Learning Transferable Visual Models From Natural Language
Supervision*, ICML 2021).

Consequence: if SEVA's UNet cross-attention learned to consume
`c_img = encode_image(x)`, then an embedding computed from text lies
approximately in the same distribution and can be used *at inference time*
without retraining. This is the trick that also underlies Stable unCLIP,
Versatile Diffusion, IP-Adapter, and CAT3D.

## 3. The paper this implementation is faithful to — directional CLIP guidance

Blending text and image embeddings directly is risky: at high blend weights
the embedding leaves the manifold the UNet was trained on and artifacts
explode. The safer recipe we used is the **CLIP directional-edit** idea that
originated in StyleCLIP (Patashnik et al., *StyleCLIP: Text-Driven
Manipulation of StyleGAN Imagery*, ICCV 2021) and was sharpened in
**StyleGAN-NADA** (Gal et al., *StyleGAN-NADA: CLIP-Guided Domain Adaptation
of Image Generators*, SIGGRAPH 2022), which introduced the explicit
*directional CLIP loss*.

The key observation from StyleGAN-NADA is that the vector difference between
two text embeddings,

$$\Delta_{\text{text}} = E_T(\text{target}) - E_T(\text{neutral})$$

points in a semantically meaningful direction in CLIP space — e.g., "photo of
a dog" minus "photo of an animal" points toward *dog-ness*. Rather than
replacing the image embedding with a text embedding, you **perturb** the
image embedding along the normalized text direction:

$$\tilde c = c_{\text{img}} + \lambda \cdot \lVert c_{\text{img}} \rVert \cdot \frac{\Delta_{\text{text}}}{\lVert \Delta_{\text{text}} \rVert}$$

Three properties matter here:

- The perturbation is **unit-direction** (only the orientation carries
  semantic meaning, not the text-embedding magnitude).
- The perturbation is **rescaled to the image embedding's own norm**, so it
  remains on the distribution the UNet is used to seeing — this is what the
  plan calls "keeping `c_img` on-manifold."
- The **neutral anchor** cancels out the "it's a photograph" direction that
  every CLIP text embedding contains, leaving only the *delta* of interest
  (sunset vs. daytime, rain vs. dry, etc.). Without it,
  `encode_text(target)` alone would drag every embedding toward a generic
  "photo" centroid.

This is the exact formula our delta implements. It corresponds to "Flavor A"
in `RESEARCH_NOTES.md` §2.

## 4. The code delta (three files, ~150 lines)

### 4.1 `CLIPConditioner` — adds a dormant text-direction hook

The upstream forward pass is preserved byte-for-byte in the `delta = None`
case, and a `set_direction(delta, lambda_)` setter lets the caller arm/disarm
the perturbation around each `run_one_scene` call. The full implementation:

```python
@torch.no_grad()
def encode_text_direction(
    self, target: str, neutral: str = "a photo"
) -> torch.Tensor:
    """
    Return a unit-norm CLIP-text direction ``target - neutral`` in the
    joint image/text space.  The returned tensor is shape ``(1024,)``.
    ...
    """
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    device = next(self.module.parameters()).device
    toks = tokenizer([target, neutral]).to(device)
    feats = self.module.encode_text(toks).float()  # (2, 1024)
    delta = feats[0] - feats[1]
    norm = delta.norm() + 1e-8
    return (delta / norm).detach()

def set_direction(
    self, delta: Optional[torch.Tensor], lambda_: float = 0.0
) -> None:
    ...

def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.preprocess(x)
    z = self.module.encode_image(x)  # (N, 1024)

    if self._direction is not None and self._lambda != 0.0:
        delta = self._direction.to(device=z.device, dtype=z.dtype)
        # Per-view scale so the direction magnitude tracks each embedding.
        scale = z.norm(dim=-1, keepdim=True) * self._lambda  # (N, 1)
        z = z + scale * delta.unsqueeze(0)

    return z
```

A few implementation details worth calling out:

- **Tokenizer consistency.** `open_clip.get_tokenizer("ViT-H-14")` is used
  rather than any HuggingFace `CLIPTokenizer`, so the text and image heads
  share the trained projection layer. Mixing tokenizers from different
  checkpoints silently destroys the "joint space" assumption — this is the
  gotcha flagged at the bottom of `RESEARCH_NOTES.md`.
- **Per-view rescaling.** `z.norm(dim=-1, keepdim=True)` yields an `(N, 1)`
  tensor, so even in SEVA's multi-input mode each view's embedding is
  perturbed *in proportion to its own norm* — different views get
  different-magnitude perturbations along the same semantic axis.
- **Stateful rather than kwarg-threaded.** The plan sketches a kwarg
  `forward(self, x, prompt=None, alpha=0.2)`. The real code uses
  `set_direction(...)` on the conditioner module because
  `CLIPConditioner.forward` is called from deep inside SEVA's
  `run_one_scene` — threading a new kwarg through would require patching
  vendor code in several places. The setter/teardown pattern keeps the
  vendor surface area untouched.

### 4.2 `SevaSynthesizer.generate_views` — arms the hook around each generation

```python
clip_direction_active = False
if prompt:
    try:
        delta = self.conditioner.encode_text_direction(prompt, eff_neutral)
        self.conditioner.set_direction(delta, eff_lambda)
        clip_direction_active = True
        logger.info(
            "SEVA CLIP text-reconditioning ON | prompt=%r neutral=%r lambda=%.3f",
            prompt, eff_neutral, eff_lambda,
        )
    except Exception as exc:
        logger.warning(
            "Failed to set CLIP text direction (%s); falling back to image-only.",
            exc,
        )
        clip_direction_active = False
...
try:
    return self._run_generation(...)
finally:
    if clip_direction_active and self.conditioner is not None:
        self.conditioner.set_direction(None, 0.0)
```

Properties of this layer:

- **Fail-open.** If text encoding fails for any reason (weights missing, OOM
  during `encode_text`, …), the run proceeds with `prompt=None` semantics
  rather than crashing. The synthesizer never kills a session over a prompt.
- **`try/finally` teardown.** The direction is cleared *after* generation, so
  subsequent calls in the same worker — including baseline runs in a sweep
  — see a pristine, image-only conditioner. This matters because the
  conditioner lives on `global_model_instances["synth-seva"]` in `app.py`
  and is reused across requests.
- **Instance-level defaults.** `SevaSynthesizer.__init__` now takes
  `neutral_prompt`, `clip_lambda`, `num_steps`, and `dtype`. Per-call kwargs
  override; this is what lets the sweep driver change λ per run while
  reusing one loaded model.

### 4.3 `app.py` — exposes `prompt`, `clip_lambda`, `neutral_prompt` end-to-end

The `/process` endpoint accepts three new form fields and plumbs them through
`run_scene_pipeline` → `generate_novel_views` →
`SevaSynthesizer.generate_views`. Both `run_report.txt` and `run_info.json`
record all three fields so every session is post-hoc reproducible:

```python
prompt: Optional[str] = Form(default=None),
clip_lambda: Optional[float] = Form(default=None),
neutral_prompt: Optional[str] = Form(default=None),
```

> `prompt` is an optional text prompt (used by ViewCrafter and PanoDreamer
> for scene guidance; now also used by SEVA via training-free CLIP-direction
> re-conditioning — see `docs/RESEARCH_NOTES.md`).

For non-SEVA backends these fields are silently ignored, so the same
`/process` schema covers both ViewCrafter (which uses `prompt` as a native
video-diffusion text guide) and SEVA (which uses it to build the CLIP
direction).

## 5. Experimental protocol

The sweep is driven by `backend/experiments/clip_recond_sweep.py`. It iterates
the Cartesian product of:

- **inputs**: interior-room reference photos, e.g. `standard_benchmark.jpg`.
- **prompts** (`backend/experiments/prompts.py`): six curated labels per axis
  of variation — lighting (`sunset`, `candles`, …), weather (`rain`, `snow`,
  …), style (`midcentury`, `cyberpunk`, …), condition, palette. The default
  `--prompt-set default` runs the six-prompt subset
  `sunset / candles / rain / snow / midcentury / cyberpunk`.
- **λ** (strength): `DEFAULT_LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.3]`, matching
  the plan's ablation grid.
- **neutral prompt**: `"a photo of a living room interior"` — an
  in-distribution anchor so the CLIP direction cancels the generic
  "photo-ness" component.

Each cell writes, under
`backend/experiments/clip_recond/<timestamp>_<input>__<prompt>__l<lam>/`:

- `config.json` — `run_id`, `prompt`, `neutral_prompt`, λ, seed, CFG,
  `num_views`, `num_steps`, `dtype`, `model_version`, git SHA.
- `input.png` (copy of the reference image).
- `views/view_###.png` + `views_grid.png` (10-view contact sheet).
- `metrics.json`, `per_view_metrics.csv`.
- `README.md` with the exact re-run command.

Plus a master `summary/all_runs.csv` that is appended atomically after each
cell, so a crash mid-sweep never loses finished runs (`--skip-existing`
resumes from where it left off).

The lightweight `scripts/run_experiment.sh` is the smaller "live demo"
version — 6 cells (SEVA × {promptA, promptB, baseline}, ViewCrafter ×
{promptA, promptB, baseline}) at λ = 0.25 — used to produce the side-by-side
figures.

## 6. Metrics (`backend/experiments/metrics.py`)

Three metrics per run, all implemented to degrade to `NaN` rather than crash:

1. **CLIP-score (prompt adherence).** Cosine similarity between the OpenCLIP
   image embedding of each generated view and the OpenCLIP text embedding of
   the target prompt, using the *same* ViT-H-14 / `laion2b_s32b_b79k`
   checkpoint SEVA itself uses. Higher = the generation looks more like the
   prompt. This is the standard "CLIPScore" metric (Hessel et al., EMNLP
   2021).
2. **LPIPS-vs-input.** Standard LPIPS (AlexNet backbone) between each
   generated view and the conditioning image. Intended to measure perceptual
   drift — how far the generation has walked away from the reference.
3. **3D self-consistency (MVGBench-style, DUSt3R split proxy).** Views are
   split into even-index and odd-index subsets, each reconstructed
   independently with DUSt3R, and the two point clouds are scale-normalized
   and compared with symmetric Chamfer distance. This is a faithful, cheap
   stand-in for the "3DGS split test" in MVGBench — the full render-based
   `cPSNR/cSSIM` would require training a 3D Gaussian Splat per split per
   run, which is impractical for a 90-run sweep. The fields are reported as
   `NaN` when consistency is skipped (`--skip-consistency`) or when DUSt3R
   returns <4 points.

The expected *reading* of the sweep is: as λ increases, `mean_clip_score`
should rise (prompt is landing) while `self_consistency_chamfer` should stay
flat until some λ\*, at which point the embedding has drifted far enough
off-manifold that 3D geometry breaks. That λ\* is the quantitative answer to
"how hard can we steer SEVA with text before it stops being geometrically
consistent."

## 7. What was *not* implemented (honesty section)

The notes describe three flavors; only **Flavor A (directional CLIP
guidance)** is in the code.

- **Flavor B** (weighted blend `c̃ = (1−α)·c_img + α·c_txt`) was rejected
  because at usable α it drifts off the image-embedding manifold — the kind
  of artifact the StyleGAN-NADA paper explicitly motivates the directional
  formulation against.
- **Flavor C** (IP-Adapter-style trained MLP mapping
  `c_txt → c_img_like`) would need paired (image, caption) supervision and a
  training loop. Out of scope for a final-project-timescale deliverable.

So the contribution is deliberately the "Flavor A, training-free, ~10-line
vendor patch" slice.

## 8. Abstract-style summary

> We extend a single-image-to-3D pipeline built around Stable Virtual Camera
> (SEVA) with a **training-free text re-conditioning** mechanism. SEVA's
> upstream `CLIPConditioner` loads the full OpenCLIP ViT-H-14 dual encoder
> but only ever calls `encode_image`, leaving the text head unused. We patch
> the conditioner to also accept a prompt, compute a unit-norm direction
> `ΔE_T = E_T(target) − E_T(neutral)` in CLIP's shared image/text space, and
> perturb the image embedding along that direction with per-view
> norm-matched magnitude:
> `c̃ = c_img + λ · ‖c_img‖ · ΔE_T / ‖ΔE_T‖`. This is the directional-edit
> recipe introduced by StyleCLIP (Patashnik et al., 2021) and formalized by
> StyleGAN-NADA (Gal et al., 2022), now applied inside a multi-view
> diffusion UNet rather than a GAN. The implementation is ~150 lines across
> three files, requires no retraining, and exposes three new parameters
> (`prompt`, `clip_lambda`, `neutral_prompt`) through the backend HTTP API.
> A sweep driver evaluates the effect across six prompt axes and five
> λ ∈ {0, 0.05, 0.1, 0.2, 0.3} values using CLIPScore, LPIPS-vs-input, and a
> DUSt3R-split Chamfer-distance proxy for 3D self-consistency, producing a
> quantitative answer to how far SEVA can be steered by text before its
> multi-view geometry degrades.

## 9. References

- Radford et al., *Learning Transferable Visual Models From Natural Language
  Supervision*, ICML 2021 (CLIP).
- Patashnik et al., *StyleCLIP: Text-Driven Manipulation of StyleGAN
  Imagery*, ICCV 2021.
- **Gal et al., *StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image
  Generators*, SIGGRAPH 2022** — the direct template for the
  directional-edit formula used here.
- Hessel et al., *CLIPScore: A Reference-free Evaluation Metric for Image
  Captioning*, EMNLP 2021 — the prompt-adherence metric.
- Zhou et al., *Stable Virtual Camera: Generative View Synthesis with
  Diffusion Models*, 2025 — the base model we're augmenting.
