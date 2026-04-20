"""
Interior-room prompt catalogue for the CLIP text re-conditioning sweep.

Exactly mirrors the "Interior-room prompt set" section of
``.cursor/plans/clip_text_reconditioning_*.plan.md`` so the sweep driver and
the plan stay in sync.  Each entry is a ``(label, prompt)`` tuple; the label
is used for short filenames and figure axes.
"""
from __future__ import annotations

NEUTRAL_PROMPT = "a photo of a living room interior"

# The six-prompt subset the plan recommends running first (3 inputs x 6 prompts
# x 5 lambdas = 90 runs, ~2-3h on an A100).  One prompt per axis of variation.
DEFAULT_SWEEP_PROMPTS: list[tuple[str, str]] = [
    ("sunset",        "a living room at sunset with warm golden hour light"),
    ("candles",       "a living room at night lit only by candles"),
    ("rain",          "a living room with heavy rain falling outside the window"),
    ("snow",          "a living room with a snowy winter landscape visible outside"),
    ("midcentury",    "a mid-century modern living room"),
    ("cyberpunk",     "a cyberpunk living room with neon lighting"),
]

# Full catalogue, grouped.  The sweep driver exposes ``--prompt-set full``
# to iterate over all of these for an extended ablation.
PROMPT_GROUPS: dict[str, list[tuple[str, str]]] = {
    "lighting": [
        ("sunset",     "a living room at sunset with warm golden hour light"),
        ("candles",    "a living room at night lit only by candles"),
        ("overcast",   "a living room on an overcast grey day"),
        ("noon",       "a living room with bright noon sunlight streaming through the windows"),
        ("moonlight",  "a living room lit by moonlight"),
    ],
    "weather": [
        ("rain",       "a living room with heavy rain falling outside the window"),
        ("snow",       "a living room with a snowy winter landscape visible outside"),
        ("fog",        "a living room with dense fog outside the window"),
        ("autumn",     "a living room with autumn leaves outside the window"),
    ],
    "style": [
        ("midcentury", "a mid-century modern living room"),
        ("scandi",     "a scandinavian minimalist living room"),
        ("victorian",  "an ornate victorian-style living room"),
        ("wabisabi",   "a japanese wabi-sabi living room"),
        ("cyberpunk",  "a cyberpunk living room with neon lighting"),
    ],
    "condition": [
        ("messy",      "a messy cluttered lived-in living room"),
        ("pristine",   "a pristine staged model-home living room"),
        ("abandoned",  "a dusty abandoned living room"),
    ],
    "palette": [
        ("warm",       "a living room with a warm earthy orange and brown palette"),
        ("cool",       "a living room with a cool blue and grey palette"),
        ("mono",       "a monochromatic black and white living room"),
    ],
}


def full_prompt_list() -> list[tuple[str, str]]:
    """All prompts flattened, in group order, deduplicated by label."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for group in PROMPT_GROUPS.values():
        for label, prompt in group:
            if label in seen:
                continue
            seen.add(label)
            out.append((label, prompt))
    return out


DEFAULT_LAMBDAS: list[float] = [0.0, 0.05, 0.1, 0.2, 0.3]
