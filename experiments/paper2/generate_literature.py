"""GPU stage for the literature comparison: every method, same prior, same noise.

    python run_literature.py --models-dir models --conditioning shell_2600.npz \
        --output results/literature --stage probe
    python run_literature.py ... --stage generate

Two stages, so the strengths are fixed before any sample is judged by FEM:

probe     For every method and K, a log grid of strengths is run on the first
          `--probe-samples` pool positions. Each finished sample is scored by
          the clean surrogate at t = 0 and its gain over the unguided twin is
          recorded. Four strengths are then chosen by one rule applied to every
          method alike: the smallest strength reaching 25 / 50 / 75 / 100 % of
          that method's own largest gain. Written to literature_probe.json.
          Methods whose knob is a count (N, M, iterations) skip the probe and use
          their native grid.

generate  Runs the chosen strengths on the full pool and writes samples/<slug>.npz
          plus manifest.json in the format generate.py uses, so evaluate_all.py
          and analyze.py run unchanged on the output directory.

No strength is ever chosen by FEM, and the rule never looks at diversity, so a
method is not tuned to the comparison it is judged by. Both stages resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from paper2 import literature
from paper2.generate import OPTIONAL_SURROGATES, initial_noise, resolve_device
from paper2.literature import Context, Cost
from paper2.matrix import DEFAULT_SEED
from paper2.models import denormalize_heights, load_architect, load_surrogate, normalize_load

STEPS = (10, 20, 100)
PROBE_GRID = tuple(float(10.0 ** e) for e in range(-3, 5))  # 1e-3 .. 1e4
TARGETS = (0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Variant:
    """One entry of the comparison table: a method plus its fixed options."""

    name: str
    method: str
    options: tuple = ()          # extra keyword arguments, as (key, value) pairs
    probe: tuple | None = PROBE_GRID
    native: tuple = ()           # strengths when there is no probe
    needs: tuple = ("clean",)    # surrogates the variant cannot run without
    max_batch: int | None = None  # memory cap per chunk, for methods that multiply the batch with a graph


VARIANTS = (
    Variant("unguided", "unguided", probe=None, native=(0.0,), needs=()),
    Variant("ours-pbunet", "ours", (("engineer", "pbunet"),), needs=("pbunet",)),
    Variant("ours-hybrid_u2", "ours", (("engineer", "hybrid_u2"),), needs=("hybrid_u2",)),
    Variant("ours-hybrid_u3", "ours", (("engineer", "hybrid_u3"),), needs=("hybrid_u3",)),
    Variant("tc_detached", "tc_detached"),
    Variant("dps", "dps"),
    Variant("mpgd", "mpgd"),
    # 4 Monte Carlo copies per sample go through the 361 M clean PB-PUNet with a
    # graph kept for the gradient: 25 x 4 does not fit in 40 GB, 8 x 4 does.
    Variant("lgd", "lgd", max_batch=8),
    # The guidance rate lives in [0, 1] and already moves the predicted MF from
    # 0.49 to 0.71 at 0.02, so the grid reaches down to 1e-3 to resolve the
    # weak-guidance, high-diversity end of the curve.
    Variant("dsg", "dsg", probe=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)),
    Variant("freedom", "freedom"),
    Variant("ugd", "ugd"),
    Variant("tfg", "tfg"),
    Variant("bon-clean", "bon", (("scorer", "clean"),), probe=None, native=(2, 4, 8, 16)),
    Variant("bon-na", "bon", (("scorer", "na"),), probe=None, native=(2, 4, 8, 16), needs=("pbunet",)),
    Variant("fk-clean", "fk", (("scorer", "clean"),)),
    Variant("fk-na", "fk", (("scorer", "na"),), needs=("pbunet", "clean")),
    Variant("svdd-clean", "svdd", (("scorer", "clean"),), probe=None, native=(2, 4, 8, 16)),
    Variant("svdd-na", "svdd", (("scorer", "na"),), probe=None, native=(2, 4, 8, 16), needs=("pbunet",)),
    Variant("dflow", "dflow", probe=None, native=(1, 2, 4, 8)),
    Variant("terminal", "terminal"),
)
BY_NAME = {v.name: v for v in VARIANTS}


@dataclass(frozen=True)
class LitRun:
    variant: str
    steps: int
    strength: float
    seed: int = DEFAULT_SEED
    block: str = "lit"

    @property
    def run_id(self) -> str:
        return f"{self.block}__{self.variant}__K{self.steps}__s{self.strength:.4g}"

    @property
    def slug(self) -> str:
        digest = hashlib.blake2s(self.run_id.encode("utf-8"), digest_size=4).hexdigest()
        return f"{self.block}_{digest}"

    def to_dict(self) -> dict:
        variant = BY_NAME[self.variant]
        # The generate.py keys analyze.py and the figure scripts read.
        return asdict(self) | {
            "run_id": self.run_id, "slug": self.slug, "provider": self.variant,
            "method": variant.method, "options": dict(variant.options),
            "guidance_scale": float(self.strength), "eta": literature.ETA,
            "clip_denoised": literature.CLIP, "engineer": dict(variant.options).get("engineer", "-"),
            "guide_every": 1, "guide_w_max": 0.0, "tag": "",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--conditioning", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("probe", "generate", "all"), default="all")
    parser.add_argument("--variants", nargs="*", default=None, help=f"subset of {list(BY_NAME)}")
    parser.add_argument("--steps", nargs="*", type=int, default=list(STEPS))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--probe-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=25,
                        help="25 keeps the chunk seeding of b1-b7; particle methods multiply it internally")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_context(args: argparse.Namespace, device: torch.device) -> tuple[Context, set[str]]:
    architect, schedule, stats = load_architect(args.models_dir / "architect_solid.pt", device)
    surrogates = {"pbunet": load_surrogate("pbunet", args.models_dir / "engineer_solid.pt", device)}
    for key, filename in OPTIONAL_SURROGATES.items():
        path = args.models_dir / filename
        if path.exists():
            surrogates[key] = load_surrogate(key, path, device)
    load = normalize_load(args.conditioning, surrogates["pbunet"].stats, device)
    return Context(architect, schedule, stats, surrogates, load), set(surrogates)


def generate_batch(ctx: Context, run: LitRun, noise: torch.Tensor, batch_size: int,
                   device: torch.device) -> tuple[np.ndarray, Cost, float]:
    variant = BY_NAME[run.variant]
    method = literature.METHODS[variant.method]
    cost = Cost()
    chunks = []
    started = time.perf_counter()
    batch_size = min(batch_size, variant.max_batch or batch_size)
    for begin in range(0, noise.shape[0], batch_size):
        generator = torch.Generator(device=device).manual_seed(run.seed + begin)
        states = method(ctx, noise[begin:begin + batch_size], run.steps, generator, cost,
                        strength=float(run.strength), **dict(variant.options))
        chunks.append(states.detach())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if device.type == "cuda":
        torch.cuda.empty_cache()  # outside the timing: particle and MC methods leave large, odd-sized blocks
    return torch.cat(chunks), cost, elapsed


def choose(strengths: list[float], gains: list[float]) -> tuple[list[float], str]:
    """Smallest strengths reaching 25/50/75/100 % of the best gain, on the rising branch."""
    gains = np.asarray(gains, dtype=float)
    best = int(np.nanargmax(gains))
    if not np.isfinite(gains[best]) or gains[best] <= 0.01:
        picks = [strengths[i] for i in np.linspace(0, len(strengths) - 1, 4).round().astype(int)]
        return picks, "flat: no strength raised the predicted MF by 0.01; evenly spaced grid"
    rising = np.maximum.accumulate(np.nan_to_num(gains[:best + 1], nan=-np.inf))
    logs = np.log10(np.asarray(strengths[:best + 1]))
    picks = []
    for fraction in TARGETS:
        goal = fraction * gains[best]
        hit = int(np.argmax(rising >= goal))
        if hit == 0:
            picks.append(strengths[0])
            continue
        # log-linear interpolation between the last grid point below the goal and the first above
        low, high = rising[hit - 1], rising[hit]
        share = (goal - low) / (high - low) if high > low else 1.0
        picks.append(float(10 ** (logs[hit - 1] + share * (logs[hit] - logs[hit - 1]))))
    picks = sorted({float(f"{p:.4g}") for p in picks})
    status = "ok"
    if len(picks) < len(TARGETS):
        # The gain was already reached at the grid floor: the curve is not
        # resolved, so fill with the next grid points and flag it.
        status = "saturated at the grid floor; widen the grid downwards for this method"
        for value in strengths:
            if len(picks) >= len(TARGETS):
                break
            if value not in picks:
                picks.append(value)
        picks = sorted(picks)
    return picks, status


def probe(args, ctx: Context, variants: list[Variant], noise: torch.Tensor, device) -> dict:
    path = args.output / "literature_probe.json"
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    subset = noise[:args.probe_samples]
    scorer_cost = Cost()

    def score(states: torch.Tensor) -> float:
        return float(literature.mf(ctx, "clean", states, 0, scorer_cost).mean())

    for steps in args.steps:
        base_key = f"unguided__K{steps}"
        if base_key not in record:
            states, _, _ = generate_batch(ctx, LitRun("unguided", steps, 0.0), subset, args.batch_size, device)
            record[base_key] = {"score": score(states)}
        base = record[base_key]["score"]
        for variant in variants:
            if variant.probe is None:
                continue
            key = f"{variant.name}__K{steps}"
            if key in record and record[key].get("status"):
                continue
            entry = record.get(key, {"strengths": [], "scores": [], "seconds": []})
            for strength in variant.probe:
                if strength in entry["strengths"]:
                    continue
                states, _, elapsed = generate_batch(ctx, LitRun(variant.name, steps, strength), subset,
                                                    args.batch_size, device)
                entry["strengths"].append(strength)
                entry["scores"].append(score(states))
                entry["seconds"].append(elapsed)
                record[key] = entry
                path.write_text(json.dumps(record, indent=1), encoding="utf-8")
                print(f"  probe {key} s={strength:g} mf_pred={entry['scores'][-1]:.3f} "
                      f"({elapsed:.1f}s)", flush=True)
            order = np.argsort(entry["strengths"])
            strengths = [entry["strengths"][i] for i in order]
            gains = [entry["scores"][i] - base for i in order]
            entry["chosen"], entry["status"] = choose(strengths, gains)
            record[key] = entry
            path.write_text(json.dumps(record, indent=1), encoding="utf-8")
            print(f"probe {key}: chosen {entry['chosen']} ({entry['status']})", flush=True)
    return record


def build_runs(args, variants: list[Variant], probed: dict) -> list[LitRun]:
    runs = []
    for steps in args.steps:
        for variant in variants:
            if variant.probe is None:
                strengths = variant.native
            else:
                entry = probed.get(f"{variant.name}__K{steps}")
                if not entry or "chosen" not in entry:
                    print(f"WARNING: {variant.name} K={steps} has no probe yet; run --stage probe", flush=True)
                    continue
                strengths = entry["chosen"]
            runs.extend(LitRun(variant.name, steps, float(s)) for s in strengths)
    return runs


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    chosen = [BY_NAME[name] for name in args.variants] if args.variants else list(VARIANTS)
    print(f"device={device} torch={torch.__version__} host={platform.node()}", flush=True)
    ctx, available = load_context(args, device)
    variants = [v for v in chosen if set(v.needs) <= available]
    for v in chosen:
        if v not in variants:
            print(f"WARNING: skipping {v.name}: needs {sorted(set(v.needs) - available)}", flush=True)
    if "clean" not in available:
        raise SystemExit("clean_solid.pt is required: it scores the probe")
    noise = initial_noise(args.samples, int(ctx.architect.config.sample_size), device, DEFAULT_SEED)

    probe_path = args.output / "literature_probe.json"
    if args.dry_run:
        probed = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.exists() else {}
        print(f"surrogates: {sorted(available)}")
        print(f"probe: {sum(len(v.probe) for v in variants if v.probe) * len(args.steps)} short runs "
              f"of {args.probe_samples} samples")
        for run in build_runs(args, variants, probed):
            print(f"  {run.slug}  {run.run_id}")
        return

    probed = probe(args, ctx, variants, noise, device) if args.stage in ("probe", "all") else (
        json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.exists() else {})
    if args.stage == "probe":
        return

    samples_dir = args.output / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    done = {entry["run_id"] for entry in manifest}
    runs = build_runs(args, variants, probed)
    print(f"{len(runs)} runs, {len(done)} already in the manifest", flush=True)
    for position, run in enumerate(runs, start=1):
        if run.run_id in done:
            continue
        states, cost, elapsed = generate_batch(ctx, run, noise, args.batch_size, device)
        target = samples_dir / f"{run.slug}.npz"
        np.savez_compressed(target, z=denormalize_heights(states, ctx.stats))
        manifest.append(run.to_dict() | {
            "file": target.name, "samples": int(states.shape[0]), "batch_size": args.batch_size,
            "seconds": elapsed, "seconds_per_sample": elapsed / max(states.shape[0], 1),
            "architect_evaluations": cost.architect + cost.architect_grad,
            "architect_backward": cost.architect_grad,
            "surrogate_evaluations": cost.surrogate + cost.surrogate_grad,
            "surrogate_backward": cost.surrogate_grad,
        })
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[{position}/{len(runs)}] {run.run_id} | {elapsed:.1f}s | arch {cost.architect}"
              f"+{cost.architect_grad}bwd | surr {cost.surrogate}+{cost.surrogate_grad}bwd", flush=True)
    print(f"\nwrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
