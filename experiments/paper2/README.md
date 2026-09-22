# Sampling budget for physics-guided shell generation

Experiment suite for the follow-up to *Generative Design of Funicular Shells via
Physics-Guided Diffusion*. The question is how far the inference budget can be
cut, and what limits it.

All structural judgement is done with **Kratos only**. The published Abaqus
results are not reused: every geometry here is generated fresh and evaluated
with one tool, so the paper has one FEM pipeline to describe instead of two.

## Three stages, two machines

| Stage | Where | What it does |
|---|---|---|
| `generate.py` | GPU (A100) | produces every geometry, records cost |
| `evaluate_all.py` | CPU | drives Kratos over the geometries, in parallel |
| `analyze.py` | either | joins cost with FEM verdicts, writes `frontier.csv` |

Generation never judges mechanics and evaluation never touches torch, so the two
heavy stages run independently on the machine that suits them.

```bash
# GPU
python -m paper2.generate \
    --models-dir models --conditioning shell_2600.npz \
    --output results/budget --samples 100 --batch-size 50

# CPU, once results/budget is back
python -m paper2.evaluate_all --results results/budget --workers 10
python -m paper2.analyze --results results/budget --vae solid_vae/best.pt
```

Both heavy stages are resumable: `generate.py` skips runs already in
`manifest.json`, `evaluate_all.py` skips batches that already have a
`summary.json`. Add `--dry-run` to print the matrix without running it.

## Checkpoints expected in `--models-dir`

| File | Needed for | Status |
|---|---|---|
| `architect_solid.pt` | everything | published, v-prediction, T=1000 |
| `engineer_solid.pt` | everything | published PB-PUNet, 361 M |
| `clean_solid.pt` | `tweedie_clean`, `naive_clean` | the paper's `modelo_clean` |
| `engineer_hybrid.pt` | block b4 | the 11 M spectral surrogate |

Runs whose surrogate is absent are skipped with a warning rather than failing,
so a partial set of checkpoints still produces a partial sweep.

## What each block tests

- **b1 — frontier.** Provider × steps × stochasticity. The claim under test is
  that the time-conditioned Engineer's advantage over a Tweedie/DPS evaluation
  *widens* as the step budget shrinks, because a short trajectory spends
  proportionally more of itself at high noise.
- **b2 — stochasticity vs clipping.** The published sampler ran with
  `clip_sample=True` *and* ancestral noise, so two defences against
  high-frequency artefacts were active at once. Crossing them separates their
  contributions.
- **b3 — guidance scale.** Whether the published operating point still holds
  when there are 20 steps instead of 1000.
- **b4 — cheap surrogate.** The PB-PUNet is 89 % of the cost of a guided step.
  A 32× smaller spectral surrogate is tested at equal FEM-verified quality.
- **b5 — sparse guidance.** The bell schedule already makes the correction
  negligible outside a window, so the surrogate is evaluated every n-th step.

## Design notes

**Paired comparisons.** One pool of initial noise, drawn once from a fixed seed,
is reused by every run. A difference between two rows is a difference between
samplers, not between random draws. The ancestral noise stream is seeded from
the position in the pool, so results do not depend on the batch size.

**Discrete schedule.** The published Architect was trained on integer timesteps
with the Diffusers cosine betas, whose `alphas_cumprod` differs from the
analytic continuous cosine by up to 5e-4, and it never saw a fractional
timestep. `discrete_vp.py` therefore samples on the integer grid. It reproduces
`DDPMScheduler` at eta=1 to 1e-15 in the mean and `DDIMScheduler` at eta=0 to
1e-7, so one sampler spans the whole stochasticity axis while staying faithful
at both ends.

**Which Membrane Factor.** `analyze.py` reads `mf_resultants_area_mean`, the
resultant-based definition of the dataset pipeline. On the published guided
samples it reproduces the reported Abaqus verification to within 0.008, whereas
the element-energy `mf_mean` sits about 0.035 lower. The three definitions are
not interchangeable and only one of them matches the convention the dataset was
built with.

**Surrogate bias.** Quality is never read from the network that supplies the
guidance gradient. The published surrogate overestimates the Membrane Factor by
roughly +0.06 against FEM, so a frontier measured with it would be circular.
