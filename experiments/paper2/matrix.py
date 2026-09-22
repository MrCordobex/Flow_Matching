"""The experiment matrix for the sampling-budget study.

Blocks map one-to-one onto the claims they support, so a block can be dropped
or rerun without disturbing the others. Every run shares one pool of initial
noise, which makes the comparisons paired: a difference between two rows is a
difference between samplers, not between random draws.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

# Guidance strength of the published configuration: guidance_scale 250 x w_max 8.
PAPER_SCALE, PAPER_W_MAX = 250.0, 8.0
FULL_STEPS = 1000


@dataclass(frozen=True)
class Run:
    block: str
    provider: str = "noise_aware"
    steps: int = 50
    eta: float = 1.0
    clip_denoised: bool = True
    guidance_scale: float = PAPER_SCALE
    guide_w_max: float = PAPER_W_MAX
    grad_clip: float = 5.0
    bell_peak: float = 0.5
    bell_width: float = 0.22
    engineer: str = "pbunet"     # pbunet | hybrid
    guide_every: int = 1         # evaluate the surrogate every n-th step
    tag: str = ""

    @property
    def run_id(self) -> str:
        parts = [
            self.block, self.provider, f"K{self.steps}", f"eta{self.eta:g}",
            "clip" if self.clip_denoised else "noclip",
            f"g{self.guidance_scale:g}x{self.guide_w_max:g}",
            self.engineer,
        ]
        if self.guide_every != 1:
            parts.append(f"every{self.guide_every}")
        if self.tag:
            parts.append(self.tag)
        return "__".join(parts)

    @property
    def slug(self) -> str:
        """Short, stable name for paths.

        Windows caps a path at 260 characters and the Kratos evaluator nests
        `<out>/samples/sample_XXXX.json` under it, so the descriptive run_id is
        kept for tables and a hash is used on disk.
        """
        digest = hashlib.blake2s(self.run_id.encode("utf-8"), digest_size=4).hexdigest()
        return f"{self.block}_{digest}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["run_id"] = self.run_id
        data["slug"] = self.slug
        return data


STEP_GRID = (10, 20, 50, 100, 250, FULL_STEPS)


def block1_frontier() -> Iterator[Run]:
    """C1 + C2: does the noise-aware advantage widen as the budget shrinks?"""
    for provider in ("noise_aware", "tweedie_clean", "tweedie_self"):
        for steps in STEP_GRID:
            for eta in (0.0, 1.0):
                yield Run(block="b1", provider=provider, steps=steps, eta=eta)


def block2_stochasticity() -> Iterator[Run]:
    """C2: separate the two defences the published sampler already had on.

    Clipping x0 and injecting noise both suppress high-frequency artefacts, and
    the paper ran with both. Crossing them tells us which one is doing the work.
    """
    for eta in (0.0, 0.25, 0.5, 0.75, 1.0):
        for clip in (True, False):
            yield Run(block="b2", steps=50, eta=eta, clip_denoised=clip)


def block3_guidance_scale() -> Iterator[Run]:
    """Does the guidance scale need recalibrating when the budget shrinks?"""
    for steps in (20, FULL_STEPS):
        for scale in (0.0, 10.0, 50.0, 250.0, 1000.0):
            yield Run(block="b3", steps=steps, eta=1.0, guidance_scale=scale)


def block4_cheap_surrogate() -> Iterator[Run]:
    """C3: a 32x smaller spectral Engineer against the published PB-PUNet."""
    for steps in STEP_GRID:
        yield Run(block="b4", steps=steps, eta=1.0, engineer="hybrid")


def block5_sparse_guidance() -> Iterator[Run]:
    """C3b: the surrogate dominates cost, so spend it on fewer steps."""
    for every in (1, 2, 4, 10):
        yield Run(block="b5", steps=100, eta=1.0, guide_every=every)


BLOCKS = {
    "b1": block1_frontier,
    "b2": block2_stochasticity,
    "b3": block3_guidance_scale,
    "b4": block4_cheap_surrogate,
    "b5": block5_sparse_guidance,
}


def build(blocks: list[str] | None = None) -> list[Run]:
    chosen = blocks or list(BLOCKS)
    runs: list[Run] = []
    seen: set[str] = set()
    for name in chosen:
        if name not in BLOCKS:
            raise ValueError(f"unknown block {name}; choose from {sorted(BLOCKS)}")
        for run in BLOCKS[name]():
            if run.run_id not in seen:
                seen.add(run.run_id)
                runs.append(run)
    return runs


def summarise(runs: list[Run]) -> str:
    architect = sum(r.steps for r in runs)
    surrogate = sum(
        r.steps // r.guide_every + (1 if r.steps % r.guide_every else 0)
        for r in runs if r.guidance_scale > 0
    )
    tweedie = sum(r.steps for r in runs if r.provider.startswith("tweedie"))
    lines = [f"{len(runs)} runs"]
    for name in sorted({r.block for r in runs}):
        lines.append(f"  {name}: {sum(1 for r in runs if r.block == name)} runs")
    lines.append(f"architect evaluations : {architect + tweedie:,}")
    lines.append(f"surrogate evaluations : {surrogate:,} (forward + backward)")
    return "\n".join(lines)
