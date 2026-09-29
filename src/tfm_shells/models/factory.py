from __future__ import annotations

from typing import Any

import torch
from diffusers import UNet2DModel

from tfm_shells.models.equino import EquiNOModel
from tfm_shells.models.hybrid_fourier_unet import HybridFourierUNet
from tfm_shells.models.ncf_hybrid import NoiseCalibratedHybrid
from tfm_shells.models.parallel_pb_unet import ParallelPBUNet
from tfm_shells.models.shell_weakrefine_operator import ShellWeakRefineOperator
from tfm_shells.models.wavelet import WaveletSplitSurrogate, extra_channels


def build_unet(model_config: dict[str, Any]) -> torch.nn.Module:
    kind = str(model_config.get("kind", "unet"))
    if kind == "wavelet_split":
        # The wrapper takes the usual [x_t, fz]; the backbone gets the wavelet
        # front end instead of x_t (see tfm_shells.models.wavelet).
        backbone_kind = str(model_config["backbone"])
        if backbone_kind == "wavelet_split":
            raise ValueError("wavelet_split needs a surrogate backbone, not another wavelet_split")
        wavelet = model_config.get("wavelet", {})
        levels = int(wavelet.get("levels", 4))
        mode = str(wavelet.get("mode", "split"))
        evidence = bool(wavelet.get("evidence", True))
        backbone = build_unet(dict(model_config, kind=backbone_kind,
                                   in_channels=int(model_config["in_channels"])
                                   + extra_channels(levels, mode, evidence)))
        return WaveletSplitSurrogate(
            backbone,
            sample_size=int(model_config["sample_size"]),
            levels=levels,
            threshold=float(wavelet.get("threshold", 3.0)),
            learnable_threshold=bool(wavelet.get("learnable_threshold", True)),
            mode=mode,
            evidence=evidence,
        )
    if kind == "ncf_hybrid":
        ncf = model_config.get("ncf", {})
        return NoiseCalibratedHybrid(
            sample_size=int(model_config["sample_size"]),
            in_channels=int(model_config["in_channels"]),
            out_channels=int(model_config["out_channels"]),
            base_channels=int(model_config.get("base_channels", 32)),
            spectral_modes=int(model_config.get("spectral_modes", 8)),
            spectral_layers=int(model_config.get("spectral_layers", 2)),
            fft_padding=int(model_config.get("fft_padding", 4)),
            time_embedding_dim=int(model_config.get("time_embedding_dim", 128)),
            dropout=float(model_config.get("dropout", 0.05)),
            branch_channels=model_config.get("branch_channels"),
            noise_condition=bool(ncf.get("noise_condition", True)),
            calibrated_stem=bool(ncf.get("calibrated_stem", True)),
            filter_scales=tuple(float(s) for s in ncf.get("filter_scales", (1.0, 2.0, 4.0))),
            conditional_filters=bool(ncf.get("conditional_filters", True)),
            attention=bool(ncf.get("attention", True)),
            attention_heads=int(ncf.get("attention_heads", 4)),
        )
    if kind == "hybrid_fourier_unet":
        return HybridFourierUNet(
            sample_size=int(model_config["sample_size"]),
            in_channels=int(model_config["in_channels"]),
            out_channels=int(model_config["out_channels"]),
            base_channels=int(model_config.get("base_channels", 32)),
            spectral_modes=int(model_config.get("spectral_modes", 8)),
            spectral_layers=int(model_config.get("spectral_layers", 2)),
            fft_padding=int(model_config.get("fft_padding", 4)),
            time_embedding_dim=int(model_config.get("time_embedding_dim", 128)),
            dropout=float(model_config.get("dropout", 0.05)),
            branch_channels=model_config.get("branch_channels"),
        )
    if kind == "parallel_pb_unet":
        return ParallelPBUNet(
            sample_size=int(model_config["sample_size"]),
            in_channels=int(model_config["in_channels"]),
            layers_per_block=int(model_config["layers_per_block"]),
            block_out_channels=tuple(int(value) for value in model_config["block_out_channels"]),
            down_block_types=tuple(model_config["down_block_types"]),
            up_block_types=tuple(model_config["up_block_types"]),
            branch_channels=model_config.get("branch_channels"),
        )
    if kind == "equino":
        return EquiNOModel(
            sample_size=int(model_config["sample_size"]),
            in_channels=int(model_config["in_channels"]),
            out_channels=int(model_config["out_channels"]),
            operator_width=int(model_config.get("operator_width", 128)),
            num_operator_layers=int(model_config.get("num_operator_layers", 6)),
            spectral_modes_height=int(model_config.get("spectral_modes_height", 16)),
            spectral_modes_width=int(model_config.get("spectral_modes_width", 16)),
            time_embedding_dim=int(model_config.get("time_embedding_dim", 256)),
            head_hidden_channels=int(model_config.get("head_hidden_channels", 128)),
            modal_rank=int(model_config.get("modal_rank", 12)),
            modal_residual_weight=float(model_config.get("modal_residual_weight", 0.25)),
            branch_channels=model_config.get("branch_channels"),
            use_coordinate_grid=bool(model_config.get("use_coordinate_grid", True)),
            dropout=float(model_config.get("dropout", 0.0)),
        )
    if kind == "shell_weakrefine_operator":
        return ShellWeakRefineOperator(
            sample_size=int(model_config["sample_size"]),
            in_channels=int(model_config["in_channels"]),
            out_channels=int(model_config["out_channels"]),
            operator_width=int(model_config.get("operator_width", 128)),
            num_operator_layers=int(model_config.get("num_operator_layers", 6)),
            spectral_modes_height=int(model_config.get("spectral_modes_height", 16)),
            spectral_modes_width=int(model_config.get("spectral_modes_width", 16)),
            time_embedding_dim=int(model_config.get("time_embedding_dim", 256)),
            branch_hidden_channels=int(model_config.get("branch_hidden_channels", 128)),
            branch_channels=model_config.get("branch_channels"),
            use_coordinate_grid=bool(model_config.get("use_coordinate_grid", True)),
            predict_log_variance=bool(model_config.get("predict_log_variance", True)),
            dropout=float(model_config.get("dropout", 0.0)),
        )
    return UNet2DModel(
        sample_size=int(model_config["sample_size"]),
        in_channels=int(model_config["in_channels"]),
        out_channels=int(model_config["out_channels"]),
        layers_per_block=int(model_config["layers_per_block"]),
        block_out_channels=tuple(int(value) for value in model_config["block_out_channels"]),
        down_block_types=tuple(model_config["down_block_types"]),
        up_block_types=tuple(model_config["up_block_types"]),
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
