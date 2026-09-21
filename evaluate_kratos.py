# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy>=2.1",
#   "plotly>=6.0",
#   "KratosMultiphysics==10.4.3",
#   "KratosStructuralMechanicsApplication==10.4.3",
#   "KratosLinearSolversApplication==10.4.3",
# ]
# ///
"""Evaluate generated solid shells with Kratos, independently of the Engineer.

Run: uv run evaluate_kratos.py artifacts/sample/RUN
PEP 723 dependencies keep the FEM environment separate from torch/CUDA.
Local sources: Code/Prueba Kratos/scripts/generate_funicular_dataset.py
(native energy MF and corotational nonlinear analysis), and
Code/Dataset_Kratos/generar_dataset_laminas.py (resultant-based MF).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np


@dataclass(frozen=True)
class Settings:
    span_x: float = 10.0
    span_y: float = 10.0
    thickness: float = 0.10
    young_modulus: float = 30e9
    poisson_ratio: float = 0.20
    density: float = 2500.0
    gravity: float = 9.81
    support_fraction: float = 0.10
    max_iterations: int = 50
    equilibrium_tolerance: float = 5e-4


def load_samples(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        z = np.array(data["z"], dtype=np.float64)
    if z.ndim == 4 and z.shape[1] == 1:
        z = z[:, 0]
    if z.ndim == 2:
        z = z[None]
    if z.ndim != 3 or min(z.shape) == 0 or min(z.shape[1:]) < 3:
        raise ValueError(f"Expected z (B,1,H,W), (B,H,W) or (H,W), got {z.shape}")
    if not np.isfinite(z).all():
        raise ValueError("Input z contains NaN/Inf")
    return z


def support_mask(z: np.ndarray, fraction: float) -> np.ndarray:
    # Heights are physical metres, already denormalized by sampling.
    if not np.isfinite(z).all() or float(z.max()) <= 0:
        raise ValueError("Support rule requires finite heights and z_max > 0")
    mask = z <= fraction * float(z.max())
    if mask.sum() < 3 or mask.all():
        raise ValueError(f"Invalid support mask: {mask.sum()} / {mask.size} nodes")
    points = np.column_stack(np.nonzero(mask))
    if np.linalg.matrix_rank(points - points[0]) < 2:
        raise ValueError("Support nodes are collinear")
    return mask


def nodal_average(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    h, w = values.shape
    valid = np.ones((h, w), bool) if valid is None else valid
    total, count = np.zeros((h + 1, w + 1)), np.zeros((h + 1, w + 1))
    for j in (0, 1):
        for i in (0, 1):
            total[j:j+h, i:i+w] += np.where(valid, values, 0.0)
            count[j:j+h, i:i+w] += valid
    return total / np.maximum(count, 1)


def resultant_mf(n: np.ndarray, m: np.ndarray, s: Settings) -> np.ndarray:
    """Exactly the compliance expression in Dataset_Kratos.membrane_factor."""
    def q(t):
        return (t[..., 0]**2 + t[..., 1]**2 - 2*s.poisson_ratio*t[..., 0]*t[..., 1]
                + 2*(1+s.poisson_ratio)*t[..., 2]**2)
    um = q(n) / (2*s.young_modulus*s.thickness)
    ub = 12*q(m) / (2*s.young_modulus*s.thickness**3)
    return um / np.maximum(um + ub, 1e-30)


def solve_shell(z: np.ndarray, s: Settings) -> tuple[dict, dict]:
    import KratosMultiphysics as KM
    import KratosMultiphysics.LinearSolversApplication as KLS
    import KratosMultiphysics.StructuralMechanicsApplication as SMA

    KM.Logger.GetDefaultOutput().SetSeverity(KM.Logger.Severity.WARNING)
    h, w = z.shape
    x, y = np.meshgrid(np.linspace(0, s.span_x, w), np.linspace(0, s.span_y, h))
    support = support_mask(z, s.support_fraction)
    model = KM.Model()
    mp = model.CreateModelPart("GeneratedShell")
    mp.SetBufferSize(2)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
    for variable in (KM.DISPLACEMENT, KM.ROTATION, KM.REACTION, KM.REACTION_MOMENT,
                     KM.VOLUME_ACCELERATION, SMA.POINT_LOAD):
        mp.AddNodalSolutionStepVariable(variable)
    prop = mp.CreateNewProperties(1)
    for variable, value in ((KM.YOUNG_MODULUS, s.young_modulus),
                            (KM.POISSON_RATIO, s.poisson_ratio),
                            (KM.THICKNESS, s.thickness), (KM.DENSITY, s.density)):
        prop.SetValue(variable, value)
    prop.SetValue(KM.VOLUME_ACCELERATION, [0., 0., 0.])
    prop.SetValue(KM.CONSTITUTIVE_LAW, SMA.LinearElasticPlaneStress2DLaw())
    for k, (xx, yy, zz) in enumerate(zip(x.flat, y.flat, z.flat), 1):
        mp.CreateNewNode(k, float(xx), float(yy), float(zz))
    dofs = ((KM.DISPLACEMENT_X, KM.REACTION_X), (KM.DISPLACEMENT_Y, KM.REACTION_Y),
            (KM.DISPLACEMENT_Z, KM.REACTION_Z), (KM.ROTATION_X, KM.REACTION_MOMENT_X),
            (KM.ROTATION_Y, KM.REACTION_MOMENT_Y), (KM.ROTATION_Z, KM.REACTION_MOMENT_Z))
    for dof, reaction in dofs:
        KM.VariableUtils().AddDof(dof, reaction, mp)
    element_name = "ShellThinElementCorotational3D4N"
    loads = np.zeros(h*w)
    areas = np.zeros((h-1, w-1))
    for j in range(h-1):
        for i in range(w-1):
            ids = [j*w+i+1, j*w+i+2, (j+1)*w+i+2, (j+1)*w+i+1]
            el = mp.CreateNewElement(element_name, j*(w-1)+i+1, ids, prop)
            areas[j, i] = el.GetGeometry().Area()
            # Dead self-weight on the actual undeformed surface, no projected pressure.
            loads[np.asarray(ids)-1] -= s.density*s.thickness*s.gravity*areas[j, i]/4
    for k, force in enumerate(loads, 1):
        node = mp.Nodes[k]
        node.SetSolutionStepValue(SMA.POINT_LOAD, 0, [0., 0., float(force)])
        mp.CreateNewCondition("PointLoadCondition3D1N", k, [k], prop)
        if support.flat[k-1]:
            for dof, _ in dofs:
                node.Fix(dof)
    mp.ProcessInfo[KM.STEP] = 1
    mp.CloneTimeStep(1.)
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    builder = KM.ResidualBasedBlockBuilderAndSolver(KLS.SparseLUSolver())
    criterion = KM.ResidualCriteria(1e-7, 1e-9)
    criterion.SetEchoLevel(0)
    strategy = KM.ResidualBasedNewtonRaphsonStrategy(
        mp, scheme, criterion, builder, s.max_iterations, True, False, True)
    strategy.SetEchoLevel(0)
    strategy.Initialize()
    strategy.Check()
    strategy.InitializeSolutionStep()
    strategy.Predict()
    if not strategy.SolveSolutionStep():
        raise RuntimeError("Kratos solver did not converge")
    strategy.FinalizeSolutionStep()

    u = np.array([list(mp.Nodes[k].GetSolutionStepValue(KM.DISPLACEMENT))
                  for k in range(1, h*w+1)]).reshape(h, w, 3)
    reactions = np.array([list(mp.Nodes[k].GetSolutionStepValue(KM.REACTION))
                          for k in range(1, h*w+1)]).reshape(h, w, 3)
    imbalance = reactions[support].sum(axis=0) + np.array([0., 0., loads.sum()])
    equilibrium = float(np.linalg.norm(imbalance) / abs(loads.sum()))
    if not np.isfinite(u).all() or not np.isfinite(equilibrium) or equilibrium > s.equilibrium_tolerance:
        raise RuntimeError(f"Invalid solution: equilibrium relative error = {equilibrium:.3e}")
    em, eb = np.zeros_like(areas), np.zeros_like(areas)
    n, m = np.zeros((*areas.shape, 3)), np.zeros((*areas.shape, 3))
    for el in mp.Elements:
        j, i = divmod(el.Id-1, w-1)
        for variable, target in ((SMA.SHELL_ELEMENT_MEMBRANE_ENERGY, em),
                                 (SMA.SHELL_ELEMENT_BENDING_ENERGY, eb)):
            values = np.asarray(el.CalculateOnIntegrationPoints(variable, mp.ProcessInfo))
            if not np.isfinite(values).all():
                raise RuntimeError(f"Non-finite energy in element {el.Id}")
            target[j, i] = np.maximum(values, 0).sum()
        for variable, target in ((SMA.SHELL_FORCE, n), (SMA.SHELL_MOMENT, m)):
            matrices = el.CalculateOnIntegrationPoints(variable, mp.ProcessInfo)
            target[j, i] = np.mean([[a[0, 0], a[1, 1], a[0, 1]] for a in matrices], axis=0)
    if not np.isfinite(n).all() or not np.isfinite(m).all() or (em+eb).sum() <= 0:
        raise RuntimeError("Non-finite resultants or zero total membrane/bending energy")
    mf_element = em / np.maximum(em + eb, 1e-30)
    mf = np.clip(nodal_average(mf_element), 0, 1)
    mf_nm = resultant_mf(n, m, s)
    valid = ~(support[:-1, :-1] & support[1:, :-1] & support[:-1, 1:] & support[1:, 1:])
    # Match Dataset_Kratos: exclude elements whose four nodes are fixed.
    mf_nm = np.where(valid, mf_nm, 0.)
    arrays = dict(z=z, x=x, y=y, support=support, u=u, reaction=reactions,
                  nodal_load_z=loads.reshape(h, w), mf=mf, mf_element=mf_element,
                  membrane_energy_measure=em, bending_energy_measure=eb, area=areas,
                  N=n, M=m, mf_resultants=nodal_average(mf_nm, valid),
                  mf_resultants_element=mf_nm)
    metrics = dict(mf_mean=float(mf.mean()),
                   mf_energy_ratio=float(em.sum() / (em+eb).sum()),
                   mf_resultants_area_mean=float((mf_nm*areas)[valid].sum()/areas[valid].sum()),
                   max_displacement_m=float(np.linalg.norm(u, axis=-1).max()),
                   equilibrium_relative_error=equilibrium, support_nodes=int(support.sum()),
                   support_cutoff_m=float(s.support_fraction*z.max()),
                   applied_force_z_n=float(loads.sum()), reaction_z_n=float(reactions[support, 2].sum()))
    strategy.Clear()
    return arrays, metrics


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def make_viewer(output: Path, rows: list[dict]) -> None:
    import plotly.graph_objects as go

    good = [row for row in rows if row["status"] == "ok"]
    fig = go.Figure()
    if not good:
        fig.add_annotation(text="No hay muestras válidas. Consulta metrics.csv.", showarrow=False)
    buttons = []
    for number, row in enumerate(good):
        with np.load(output / row["file"]) as d:
            x, y, z, mf, support = (d[k] for k in ("x", "y", "z", "mf", "support"))
            title = (f"Muestra {row['sample_index']} | MF Kratos (media nodal)={row['mf_mean']:.4f}"
                     f" | Ratio energético={row['mf_energy_ratio']:.4f}")
            if number == 0:
                fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=mf, cmin=0, cmax=1,
                                        colorscale="RdYlGn", colorbar=dict(title="MF Kratos"),
                                        hovertemplate="x=%{x:.3f} m<br>y=%{y:.3f} m<br>z=%{z:.3f} m<br>MF=%{surfacecolor:.4f}<extra></extra>"))
                fig.add_trace(go.Scatter3d(x=x[support], y=y[support], z=z[support],
                                          mode="markers", marker=dict(size=2, color="black"),
                                          name="Empotramientos", hovertemplate="Apoyo<br>z=%{z:.3f} m<extra></extra>"))
                fig.update_layout(title=title)
            buttons.append(dict(label=f"{row['sample_index']:04d} · MF={row['mf_mean']:.4f}",
                                method="update", args=[
                                    dict(x=[x.tolist(), x[support].tolist()],
                                         y=[y.tolist(), y[support].tolist()],
                                         z=[z.tolist(), z[support].tolist()],
                                         surfacecolor=[mf.tolist(), None]),
                                    dict(title=dict(text=title))]))
    fig.update_layout(height=720, margin=dict(l=0, r=0, b=0, t=100),
                      scene=dict(xaxis_title="X [m]", yaxis_title="Y [m]", zaxis_title="Z [m]",
                                 aspectmode="data", uirevision="keep-camera"),
                      updatemenus=[dict(buttons=buttons, x=0, y=1.08, xanchor="left", yanchor="top")])
    fig.write_html(output / "mf_gallery.html", include_plotlyjs=True, full_html=True)
    fig.write_json(output / "mf_gallery.plotly.json")
    # HTML in srcdoc works in both classic Jupyter and Colab, without a local HTTP server.
    code = ("from pathlib import Path\nimport html\nfrom IPython.display import HTML, display\n\n"
            "# Cambia OUTPUT si abres este notebook desde otra carpeta.\n"
            "OUTPUT = Path('.')\n"
            "document = (OUTPUT / 'mf_gallery.html').read_text(encoding='utf-8')\n"
            "display(HTML('<iframe style=\"width:100%;height:780px;border:0\" srcdoc=\"'\n"
            "             + html.escape(document, quote=True) + '\"></iframe>'))\n")
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3")),
                    cells=[dict(cell_type="markdown", id="intro", metadata={}, source=[
                        "# MF calculado con Kratos\n",
                        "Geometría original, coloreada por MF nodal; puntos negros: empotramientos.\n",
                        "Selecciona una muestra en el menú y arrastra para girar la vista.\n"]),
                           dict(cell_type="code", id="viewer", metadata={}, execution_count=None,
                                outputs=[], source=code.splitlines(keepends=True))])
    write_json(output / "view_mf.ipynb", notebook)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("sampling_dir", type=Path, help="Run folder, its name inside artifacts/sample, or NPZ")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--limit", type=int, default=0, help="0: all samples")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--overwrite", action="store_true", help="Recompute selected samples")
    for key in ("span_x", "span_y", "thickness", "young_modulus", "poisson_ratio", "density",
                "gravity", "support_fraction", "equilibrium_tolerance"):
        p.add_argument("--"+key.replace("_", "-"), type=float, default=getattr(Settings(), key))
    p.add_argument("--max-iterations", type=int, default=50)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    s = Settings(**{key: getattr(args, key) for key in Settings.__dataclass_fields__})
    for key in ("span_x", "span_y", "thickness", "young_modulus", "density", "gravity", "equilibrium_tolerance"):
        if not np.isfinite(getattr(s, key)) or getattr(s, key) <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not 0 < s.support_fraction < 1 or not -1 < s.poisson_ratio < .5:
        raise ValueError("Invalid support fraction or Poisson ratio")
    if args.limit < 0 or args.start < 0 or s.max_iterations < 1:
        raise ValueError("Invalid sample range or iteration count")
    source = args.sampling_dir.expanduser()
    if not source.exists():
        source = Path(__file__).resolve().parent / "artifacts" / "sample" / source
    source = source.resolve()
    if source.is_dir():
        source = source / "guided_samples.npz"
    if not source.is_file():
        raise FileNotFoundError(f"Sampling file not found: {source}")
    z = load_samples(source)
    if args.start >= len(z):
        raise ValueError("--start is beyond the input sample count")
    output = args.output_dir.resolve() if args.output_dir else source.parent / "kratos"
    output.mkdir(parents=True, exist_ok=True)
    samples_dir = output / "samples"
    samples_dir.mkdir(exist_ok=True)
    fingerprint = hashlib.sha256(source.read_bytes() + json.dumps(asdict(s), sort_keys=True).encode()
                                 + Path(__file__).read_bytes()).hexdigest()
    manifest_path = output / "evaluation.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Input, settings or evaluator changed. Use a different --output-dir to avoid mixing results.")
    write_json(manifest_path, dict(source=str(source), fingerprint=fingerprint, settings=asdict(s),
                                  z_units="m; unchanged from sampling", support_rule="z <= support_fraction * max(z)",
                                  analysis="nonlinear static; ShellThinElementCorotational3D4N; Newton-Raphson",
                                  primary_mf="Prueba Kratos: native element energy ratio, averaged to nodes, then nodal mean",
                                  secondary_mf="Dataset_Kratos: mean resultants compliance, element-area average excluding fully fixed cells",
                                  load="self-weight: rho * thickness * gravity * actual reference element area; quarter to each node",
                                  samples_total=len(z)))
    end = min(len(z), args.start + args.limit) if args.limit else len(z)
    print(f"Input: {source}\nOutput: {output}\nSettings: {json.dumps(asdict(s))}", flush=True)
    failures = 0
    for index in range(args.start, end):
        result_path = samples_dir / f"sample_{index:04d}.json"
        if result_path.exists() and not args.overwrite:
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            if cached["status"] == "ok" and (output / cached["file"]).exists():
                print(f"[{index}] cached MF={cached['mf_mean']:.6f}", flush=True)
                continue
        started = time.perf_counter()
        try:
            arrays, metrics = solve_shell(z[index], s)
            filename = f"samples/sample_{index:04d}.npz"
            np.savez_compressed(output / filename, **arrays)
            row = dict(sample_index=index, status="ok", file=filename, **metrics)
            print(f"[{index}] MF={row['mf_mean']:.6f}, energy ratio={row['mf_energy_ratio']:.6f}", flush=True)
        except Exception as exc:
            failures += 1
            row = dict(sample_index=index, status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"[{index}] FAILED: {row['error']}", flush=True)
        row["seconds"] = time.perf_counter()-started
        write_json(result_path, row)
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(samples_dir.glob("sample_*.json"))]
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    good = [row for row in rows if row["status"] == "ok"]
    summary = dict(samples_total=len(z), evaluated=len(rows), successful=len(good), failed=len(rows)-len(good))
    for metric in ("mf_mean", "mf_energy_ratio", "mf_resultants_area_mean"):
        values = np.array([row[metric] for row in good])
        summary[metric] = dict(mean=float(values.mean()), std=float(values.std()),
                               min=float(values.min()), max=float(values.max())) if len(values) else None
    write_json(output / "summary.json", summary)
    make_viewer(output, rows)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Jupyter viewer: {output / 'view_mf.ipynb'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
