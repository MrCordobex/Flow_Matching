from __future__ import annotations

from pathlib import Path
from typing import Any

from tfm_shells.utils.matplotlib_backend import configure_matplotlib_backend

configure_matplotlib_backend()

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tfm_shells.config import load_config, resolve_project_path
from tfm_shells.data.conditioning import split_records_balanced_validation
from tfm_shells.data.dataset import ShellDataset, compute_normalization_stats
from tfm_shells.data.index import build_dataset_index, dataset_summary, filter_records, split_records
from tfm_shells.flow import model_time, sample_flow_path
from tfm_shells.models.factory import build_unet, count_parameters
from tfm_shells.training.common import (
    expand_physics_stats,
    finalize_latest_symlink,
    format_metric,
    make_run_name,
    plot_training_curves,
    prepare_run_directories,
    resolve_device,
    save_history,
    save_run_metadata,
    seed_everything,
)
from tfm_shells.utils.io import save_json
from tfm_shells.utils.physics import (
    build_active_refinement_mask,
    branchwise_supervised_losses,
    compute_membrane_factor_from_prediction,
    compute_physical_residual,
    compute_energy_residual_map,
    compute_weak_form_residual,
)
from tfm_shells.utils.tracking import ExperimentTracker


def _epoch_weight(lambda_max: float, warmup: int, total_epochs: int, epoch_index: int) -> float:
    if epoch_index < warmup:
        return 0.0
    progress = (epoch_index - warmup) / max(total_epochs - warmup, 1)
    return lambda_max * min(max(progress, 0.0), 1.0)


def _epoch_lambda(config: dict[str, Any], epoch_index: int) -> float:
    return _epoch_weight(
        lambda_max=float(config["training"]["lambda_max"]),
        warmup=int(config["training"]["warmup_epochs"]),
        total_epochs=int(config["training"]["epochs"]),
        epoch_index=epoch_index,
    )


def _nested_epoch_lambda(
    training_cfg: dict[str, Any],
    section_name: str,
    epoch_index: int,
) -> float:
    section = training_cfg.get(section_name, {})
    if not bool(section.get("enabled", False)):
        return 0.0
    return _epoch_weight(
        lambda_max=float(section.get("lambda_max", 0.0)),
        warmup=int(section.get("warmup_epochs", training_cfg["warmup_epochs"])),
        total_epochs=int(training_cfg["epochs"]),
        epoch_index=epoch_index,
    )


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    stats: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    lambda_epoch: float,
    include_fz_channel: bool,
    include_type_channel: bool,
    epoch: int,
    total_epochs: int,
    phase: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    use_amp = bool(config["training"]["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    total_loss = 0.0
    total_mse = 0.0
    total_phys = 0.0
    total_weak = 0.0
    total_refine = 0.0
    total_uncertainty = 0.0
    total_mf_pred = 0.0
    total_mf_true = 0.0
    total_mf_mae = 0.0
    total_uz_mse = 0.0
    total_membrane_mse = 0.0
    total_flexion_mse = 0.0
    total_items = 0

    power = float(config["training"]["timestep_power"])
    weak_form_cfg = config["training"].get("weak_form", {})
    active_refinement_cfg = config["training"].get("active_refinement", {})
    use_uncertainty_weighting = bool(config["training"].get("uncertainty_weighting", False))
    weak_lambda_epoch = _nested_epoch_lambda(config["training"], "weak_form", epoch - 1)
    refine_lambda_epoch = _nested_epoch_lambda(config["training"], "active_refinement", epoch - 1)

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        progress = tqdm(loader, leave=False, desc=f"{phase} {epoch:03d}/{total_epochs:03d}")
        for batch in progress:
            z_clean = batch["z"].to(device, non_blocking=True)
            fz_cond = batch["fz_norm"].to(device, non_blocking=True)
            type_channel = batch["type_channel"].to(device, non_blocking=True)
            fz_real = batch["fz_real"].to(device, non_blocking=True)
            physics_clean = batch["physics"].to(device, non_blocking=True)
            ds = batch["ds"].to(device, non_blocking=True)
            dv = batch["dv"].to(device, non_blocking=True)
            mf_true = batch["mf_true"].to(device, non_blocking=True)
            batch_size = z_clean.shape[0]

            p_mean, p_std = expand_physics_stats(stats, batch_size, device)

            z_noisy, time, _ = sample_flow_path(z_clean)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=use_amp):
                input_channels = [z_noisy]
                if include_fz_channel:
                    input_channels.append(fz_cond)
                if include_type_channel:
                    input_channels.append(type_channel)
                model_input = torch.cat(input_channels, dim=1)
                model_output = model(model_input, model_time(time, batch_size, device))
                pred_norm = model_output.sample
                log_variance = getattr(model_output, "log_variance", None)
                if log_variance is not None and log_variance.shape[1] == 1:
                    log_variance = log_variance.expand(-1, pred_norm.shape[1], -1, -1)

                sq_error = (pred_norm - physics_clean).square()
                if use_uncertainty_weighting and log_variance is not None:
                    loss_mse = (0.5 * (torch.exp(-log_variance) * sq_error + log_variance)).mean()
                    mean_uncertainty = torch.exp(log_variance.detach()).mean()
                else:
                    loss_mse = sq_error.mean()
                    mean_uncertainty = torch.tensor(0.0, device=device)
                branch_losses = branchwise_supervised_losses(pred_norm, physics_clean)

                if lambda_epoch > 0.0:
                    phys_per_sample = compute_physical_residual(pred_norm, p_mean, p_std, ds, dv, fz_real)
                    weighted_phys = (phys_per_sample * time.pow(power)).mean()
                else:
                    weighted_phys = torch.tensor(0.0, device=device)

                if weak_lambda_epoch > 0.0:
                    weak_per_sample = compute_weak_form_residual(
                        pred_norm,
                        p_mean,
                        p_std,
                        ds,
                        dv,
                        fz_real,
                        num_test_modes=int(weak_form_cfg.get("num_test_modes", 4)),
                    )
                    weak_loss = weak_per_sample.mean()
                else:
                    weak_loss = torch.tensor(0.0, device=device)

                if refine_lambda_epoch > 0.0:
                    residual_map = compute_energy_residual_map(pred_norm, p_mean, p_std, ds, dv, fz_real).detach()
                    uncertainty_map = None
                    if log_variance is not None:
                        uncertainty_map = torch.exp(log_variance.detach().mean(dim=1, keepdim=True))
                    refine_mask = build_active_refinement_mask(
                        residual_map=residual_map,
                        uncertainty_map=uncertainty_map,
                        topk_ratio=float(active_refinement_cfg.get("topk_ratio", 0.20)),
                        residual_weight=float(active_refinement_cfg.get("residual_weight", 0.5)),
                        uncertainty_weight=float(active_refinement_cfg.get("uncertainty_weight", 0.5)),
                    )
                    local_sq_error = sq_error.mean(dim=1, keepdim=True)
                    refine_per_sample = (
                        (refine_mask * local_sq_error).sum(dim=(1, 2, 3))
                        / refine_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
                    )
                    refine_loss = refine_per_sample.mean()
                else:
                    refine_loss = torch.tensor(0.0, device=device)

                loss = (
                    loss_mse
                    + lambda_epoch * weighted_phys
                    + weak_lambda_epoch * weak_loss
                    + refine_lambda_epoch * refine_loss
                )

            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()

            with torch.no_grad():
                input_channels_t0 = [z_clean]
                if include_fz_channel:
                    input_channels_t0.append(fz_cond)
                if include_type_channel:
                    input_channels_t0.append(type_channel)
                model_input_t0 = torch.cat(input_channels_t0, dim=1)
                pred_t0 = model(model_input_t0, torch.zeros(batch_size, device=device, dtype=torch.long)).sample
                mf_pred_map = compute_membrane_factor_from_prediction(pred_t0, p_mean, p_std)
                mf_pred_mean = mf_pred_map.mean(dim=(1, 2, 3))
                mf_true_mean = mf_true.mean(dim=(1, 2, 3))
                mf_mae = torch.abs(mf_pred_map - mf_true).mean(dim=(1, 2, 3))

            total_loss += float(loss.item()) * batch_size
            total_mse += float(loss_mse.item()) * batch_size
            total_phys += float(weighted_phys.item()) * batch_size
            total_weak += float(weak_loss.item()) * batch_size
            total_refine += float(refine_loss.item()) * batch_size
            total_uncertainty += float(mean_uncertainty.item()) * batch_size
            total_mf_pred += float(mf_pred_mean.mean().item()) * batch_size
            total_mf_true += float(mf_true_mean.mean().item()) * batch_size
            total_mf_mae += float(mf_mae.mean().item()) * batch_size
            total_uz_mse += float(branch_losses["uz_mse"].item()) * batch_size
            total_membrane_mse += float(branch_losses["membrane_mse"].item()) * batch_size
            total_flexion_mse += float(branch_losses["flexion_mse"].item()) * batch_size
            total_items += batch_size
            progress.set_postfix(
                loss=format_metric(total_loss / max(total_items, 1)),
                mse=format_metric(total_mse / max(total_items, 1)),
                phys=format_metric(total_phys / max(total_items, 1)),
                weak=format_metric(total_weak / max(total_items, 1)),
                refine=format_metric(total_refine / max(total_items, 1)),
                mf_mae=format_metric(total_mf_mae / max(total_items, 1)),
                refresh=False,
            )

    return {
        "loss": total_loss / max(total_items, 1),
        "mse": total_mse / max(total_items, 1),
        "phys": total_phys / max(total_items, 1),
        "weak": total_weak / max(total_items, 1),
        "refine": total_refine / max(total_items, 1),
        "uncertainty": total_uncertainty / max(total_items, 1),
        "mf_pred": total_mf_pred / max(total_items, 1),
        "mf_true": total_mf_true / max(total_items, 1),
        "mf_mae": total_mf_mae / max(total_items, 1),
        "uz_mse": total_uz_mse / max(total_items, 1),
        "membrane_mse": total_membrane_mse / max(total_items, 1),
        "flexion_mse": total_flexion_mse / max(total_items, 1),
    }


def _save_validation_figure(
    model: torch.nn.Module,
    loader: DataLoader,
    stats: dict[str, Any],
    device: torch.device,
    output_path: Path,
    include_fz_channel: bool,
    include_type_channel: bool,
) -> None:
    batch = next(iter(loader))
    z_clean = batch["z"].to(device)
    fz_cond = batch["fz_norm"].to(device)
    type_channel = batch["type_channel"].to(device)
    mf_true = batch["mf_true"].to(device)

    batch_size = z_clean.shape[0]
    p_mean, p_std = expand_physics_stats(stats, batch_size, device)

    model.eval()
    with torch.no_grad():
        input_channels = [z_clean]
        if include_fz_channel:
            input_channels.append(fz_cond)
        if include_type_channel:
            input_channels.append(type_channel)
        model_input = torch.cat(input_channels, dim=1)
        pred_t1 = model(model_input, model_time(1.0, batch_size, device)).sample
        mf_pred = compute_membrane_factor_from_prediction(pred_t1, p_mean, p_std)

    n_show = min(4, batch_size)
    figure, axes = plt.subplots(n_show, 3, figsize=(12, 3.2 * n_show))
    if n_show == 1:
        axes = np.array([axes])

    for row in range(n_show):
        axes[row, 0].imshow(z_clean[row, 0].cpu().numpy(), cmap="viridis")
        axes[row, 0].set_title("z_norm")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mf_true[row, 0].cpu().numpy(), cmap="magma", vmin=0.0, vmax=1.0)
        axes[row, 1].set_title("mf_true")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(mf_pred[row, 0].cpu().numpy(), cmap="magma", vmin=0.0, vmax=1.0)
        axes[row, 2].set_title("mf_pred")
        axes[row, 2].axis("off")

    figure.suptitle("Engineer validation snapshots")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def train_engineer(
    config_path: str | Path,
    *,
    conditioned_on_type: bool = False,
    balanced_validation: bool = False,
    role: str = "engineer",
) -> dict[str, Any]:
    config = load_config(config_path)
    seed_everything(int(config["seed"]))
    directories = prepare_run_directories(config, role=role)
    device = resolve_device(str(config["runtime"]["device"]))

    dataset_dir = resolve_project_path(config, config["data"]["dataset_dir"])
    records = build_dataset_index(dataset_dir)
    filtered = filter_records(
        records,
        subset=str(config["data"]["subset"]),
        min_mf_mean=config["data"].get("min_mf_mean"),
    )
    if config["data"]["subset"] != "solid" or conditioned_on_type:
        raise ValueError("This flow-matching repository trains only solid shells without a type channel.")
    if balanced_validation:
        train_records, val_records = split_records_balanced_validation(
            filtered,
            val_ratio=float(config["data"]["val_ratio"]),
            seed=int(config["seed"]),
        )
    else:
        train_records, val_records = split_records(
            filtered,
            val_ratio=float(config["data"]["val_ratio"]),
            seed=int(config["seed"]),
        )

    stats = compute_normalization_stats(train_records, include_physics=True)
    splits = {
        "train": [record["name"] for record in train_records],
        "val": [record["name"] for record in val_records],
        "dataset_summary": {
            "all_filtered": dataset_summary(filtered),
            "train": dataset_summary(train_records),
            "val": dataset_summary(val_records),
        },
    }
    save_run_metadata(config, directories, stats, splits)

    train_dataset = ShellDataset(train_records, stats=stats, include_physics=True)
    val_dataset = ShellDataset(val_records, stats=stats, include_physics=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"]["pin_memory"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"]["pin_memory"]),
    )

    model = build_unet(config["model"]).to(device)
    model_parameters = count_parameters(model)
    if (config["model"].get("kind") != "parallel_pb_unet"
            or not bool(config["data"]["include_fz_channel"])
            or int(config["model"]["out_channels"]) != 13):
        raise ValueError("Flow Engineer requires the three-branch PBUNet with fz input.")
    expected_in_channels = 1 + int(bool(config["data"]["include_fz_channel"])) + int(conditioned_on_type)
    if int(config["model"]["in_channels"]) != expected_in_channels:
        raise ValueError(
            f"Engineer model.in_channels must be {expected_in_channels} "
            f"when include_fz_channel={bool(config['data']['include_fz_channel'])} "
            f"and conditioned_on_type={conditioned_on_type}."
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["training"]["lr_factor"]),
        patience=int(config["training"]["lr_patience"]),
    )

    history_rows: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_path = directories["model_root"] / "best.pt"
    last_path = directories["model_root"] / "last.pt"
    total_epochs = int(config["training"]["epochs"])

    run_name = make_run_name(config, role=role)
    with ExperimentTracker(config, directories["project_root"], run_name) as tracker:
        summary_filtered = splits["dataset_summary"]["all_filtered"]
        print("\n" + "=" * 96)
        print("ENGINEER TRAINING")
        print(f"config: {config['_meta']['config_path']}")
        print(f"device: {device}")
        print(f"dataset: {dataset_dir}")
        print(
            f"records: filtered={summary_filtered['count']} "
            f"train={len(train_records)} val={len(val_records)} "
            f"subsets={summary_filtered['subset_counts']}"
        )
        print(
            f"normalization: z=[{format_metric(float(stats['z_min']))}, {format_metric(float(stats['z_max']))}] "
            f"fz=[{format_metric(float(stats['fz_min']))}, {format_metric(float(stats['fz_max']))}]"
        )
        print(
            f"physics_channels={len(stats['physics_mean'])} "
            f"lambda_max={format_metric(float(config['training']['lambda_max']))} "
            f"warmup_epochs={int(config['training']['warmup_epochs'])}"
        )
        print(
            f"weak_form={bool(config['training'].get('weak_form', {}).get('enabled', False))} "
            f"active_refinement={bool(config['training'].get('active_refinement', {}).get('enabled', False))} "
            f"uncertainty_weighting={bool(config['training'].get('uncertainty_weighting', False))}"
        )
        print(f"model_parameters: {model_parameters:,}")
        print(f"artifacts: {directories['run_root']}")
        print(f"checkpoints: {directories['model_root']}")
        print(f"mlflow_run_id: {tracker.run_id}")
        print("=" * 96)

        tracker.log_config(config)
        tracker.log_artifact(directories["run_root"] / "config.yaml", artifact_path="run")
        tracker.log_artifact(directories["run_root"] / "stats.json", artifact_path="run")
        tracker.log_artifact(directories["run_root"] / "splits.json", artifact_path="run")
        tracker.log_metrics(
            {
                "dataset_count_filtered": float(len(filtered)),
                "dataset_count_train": float(len(train_records)),
                "dataset_count_val": float(len(val_records)),
                "mf_mean_filtered": float(splits["dataset_summary"]["all_filtered"]["mf_mean"]),
                "model_parameters": float(model_parameters),
            }
        )

        for epoch in range(1, total_epochs + 1):
            lambda_epoch = _epoch_lambda(config, epoch - 1)
            train_metrics = _run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                stats=stats,
                config=config,
                device=device,
                lambda_epoch=lambda_epoch,
                include_fz_channel=bool(config["data"]["include_fz_channel"]),
                include_type_channel=conditioned_on_type,
                epoch=epoch,
                total_epochs=total_epochs,
                phase="train",
            )
            val_metrics = _run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                stats=stats,
                config=config,
                device=device,
                lambda_epoch=lambda_epoch,
                include_fz_channel=bool(config["data"]["include_fz_channel"]),
                include_type_channel=conditioned_on_type,
                epoch=epoch,
                total_epochs=total_epochs,
                phase="val",
            )
            lr_scheduler.step(val_metrics["loss"])
            current_lr = float(optimizer.param_groups[0]["lr"])

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_mse": train_metrics["mse"],
                "val_mse": val_metrics["mse"],
                "train_phys": train_metrics["phys"],
                "val_phys": val_metrics["phys"],
                "train_weak": train_metrics["weak"],
                "val_weak": val_metrics["weak"],
                "train_refine": train_metrics["refine"],
                "val_refine": val_metrics["refine"],
                "train_uncertainty": train_metrics["uncertainty"],
                "val_uncertainty": val_metrics["uncertainty"],
                "train_mf_pred": train_metrics["mf_pred"],
                "val_mf_pred": val_metrics["mf_pred"],
                "train_mf_true": train_metrics["mf_true"],
                "val_mf_true": val_metrics["mf_true"],
                "train_mf_mae": train_metrics["mf_mae"],
                "val_mf_mae": val_metrics["mf_mae"],
                "train_uz_mse": train_metrics["uz_mse"],
                "val_uz_mse": val_metrics["uz_mse"],
                "train_membrane_mse": train_metrics["membrane_mse"],
                "val_membrane_mse": val_metrics["membrane_mse"],
                "train_flexion_mse": train_metrics["flexion_mse"],
                "val_flexion_mse": val_metrics["flexion_mse"],
                "lambda_epoch": lambda_epoch,
                "learning_rate": current_lr,
            }
            history_rows.append(row)
            tracker.log_metrics(row, step=epoch)

            checkpoint = {
                "epoch": epoch,
                "role": role,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": config["model"],
                "flow_config": {"path": "linear", "time_scale": 999.0, "base": "standard_normal"},
                "normalization_stats": stats,
                "config_path": str(config["_meta"]["config_path"]),
                "dataset_summary": splits["dataset_summary"],
                "conditioning": {
                    "conditioned_on_type": conditioned_on_type,
                    "type_channel_encoding": {"solid": -1.0, "hole": 1.0},
                    "balanced_validation": balanced_validation,
                },
            }
            torch.save(checkpoint, last_path)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
                torch.save(checkpoint, best_path)
                tracker.log_metrics({"best_val_loss": best_val_loss}, step=epoch)
                best_marker = " [best]"
            else:
                epochs_without_improvement += 1
                best_marker = ""

            tqdm.write(
                " | ".join(
                    [
                        f"epoch {epoch:03d}/{total_epochs:03d}",
                        f"train_loss={format_metric(train_metrics['loss'])}",
                        f"val_loss={format_metric(val_metrics['loss'])}",
                        f"train_mse={format_metric(train_metrics['mse'])}",
                        f"val_mse={format_metric(val_metrics['mse'])}",
                        f"train_phys={format_metric(train_metrics['phys'])}",
                        f"val_phys={format_metric(val_metrics['phys'])}",
                        f"train_weak={format_metric(train_metrics['weak'])}",
                        f"val_weak={format_metric(val_metrics['weak'])}",
                        f"val_refine={format_metric(val_metrics['refine'])}",
                        f"val_mf_mae={format_metric(val_metrics['mf_mae'])}",
                        f"lambda={format_metric(lambda_epoch)}",
                        f"lr={format_metric(current_lr)}",
                        f"best_val={format_metric(best_val_loss)}{best_marker}",
                    ]
                )
            )
            tqdm.write(
                " | ".join(
                    [
                        f"branches train(uz={format_metric(train_metrics['uz_mse'])}, m={format_metric(train_metrics['membrane_mse'])}, f={format_metric(train_metrics['flexion_mse'])})",
                        f"val(uz={format_metric(val_metrics['uz_mse'])}, m={format_metric(val_metrics['membrane_mse'])}, f={format_metric(val_metrics['flexion_mse'])})",
                        f"mf_pred={format_metric(val_metrics['mf_pred'])}",
                        f"mf_true={format_metric(val_metrics['mf_true'])}",
                        f"patience={epochs_without_improvement}/{int(config['training']['early_stopping_patience'])}",
                    ]
                )
            )

            if epochs_without_improvement >= int(config["training"]["early_stopping_patience"]):
                tqdm.write("early_stopping_triggered")
                break

        save_history(history_rows, directories)
        curves_path = directories["run_root"] / "engineer_curves.png"
        plot_training_curves(
            history_rows=history_rows,
            metrics=[
                ("train_loss", "val_loss"),
                ("train_mse", "val_mse"),
                ("train_mf_mae", "val_mf_mae"),
                ("train_weak", "val_weak"),
            ],
            title="Engineer training curves",
            output_path=curves_path,
        )
        val_fig_path = directories["run_root"] / "engineer_validation.png"
        _save_validation_figure(
            model,
            val_loader,
            stats,
            device,
            val_fig_path,
            include_fz_channel=bool(config["data"]["include_fz_channel"]),
            include_type_channel=conditioned_on_type,
        )

        tracker.log_artifact(curves_path, artifact_path="diagnostics")
        tracker.log_artifact(val_fig_path, artifact_path="diagnostics")
        tracker.log_artifact(directories["model_root"] / "history.csv", artifact_path="run")
        tracker.log_artifact(best_path, artifact_path="checkpoints")
        tracker.log_artifact(last_path, artifact_path="checkpoints")

        summary = {
            "best_val_loss": best_val_loss,
            "epochs_completed": len(history_rows),
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
        }
        save_json(summary, directories["run_root"] / "summary.json")
        save_json(summary, directories["model_root"] / "summary.json")
        tracker.log_artifact(directories["run_root"] / "summary.json", artifact_path="run")
        print(
            f"engineer_finished | epochs={len(history_rows)} | best_val={format_metric(best_val_loss)} "
            f"| best_ckpt={best_path}"
        )

    finalize_latest_symlink(directories["latest_root"], directories["model_root"])
    return {
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }
