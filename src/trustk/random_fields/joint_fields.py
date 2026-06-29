from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trustk.random_fields.gaussian_field import GaussianField2D, generate_gaussian_logk_field


@dataclass(frozen=True)
class JointFieldConfig:
    nx: int
    ny: int
    dx: float
    dy: float
    mean_logk: float
    sigma_logk: float
    corr_len_k_x: float
    corr_len_k_y: float
    orientation_rad: float
    mean_logss: float
    sigma_logss: float
    corr_len_ss_x: float
    corr_len_ss_y: float
    log_correlation: float
    min_logss: float
    max_logss: float
    seed: int | None = None


@dataclass(frozen=True)
class JointLogField2D:
    x: np.ndarray
    y: np.ndarray
    logk: np.ndarray
    logss: np.ndarray
    config: JointFieldConfig
    specific_storage_clipped_fraction: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.logk.shape

    @property
    def k_field(self) -> GaussianField2D:
        return GaussianField2D(
            x=self.x,
            y=self.y,
            logk=self.logk,
            mean_logk=self.config.mean_logk,
            sigma_logk=self.config.sigma_logk,
            corr_len_x=self.config.corr_len_k_x,
            corr_len_y=self.config.corr_len_k_y,
            orientation_rad=self.config.orientation_rad,
            seed=self.config.seed,
        )

    @property
    def ss_field(self) -> GaussianField2D:
        return GaussianField2D(
            x=self.x,
            y=self.y,
            logk=self.logss,
            mean_logk=self.config.mean_logss,
            sigma_logk=self.config.sigma_logss,
            corr_len_x=self.config.corr_len_ss_x,
            corr_len_y=self.config.corr_len_ss_y,
            orientation_rad=self.config.orientation_rad,
            seed=self.config.seed,
        )


def generate_joint_log_fields(config: JointFieldConfig) -> JointLogField2D:
    _validate_config(config)
    k_field = generate_gaussian_logk_field(
        nx=config.nx,
        ny=config.ny,
        dx=config.dx,
        dy=config.dy,
        mean_logk=config.mean_logk,
        sigma_logk=config.sigma_logk,
        corr_len_x=config.corr_len_k_x,
        corr_len_y=config.corr_len_k_y,
        orientation_rad=config.orientation_rad,
        seed=config.seed,
    )
    ss_seed = None if config.seed is None else int(config.seed) + 104_729
    ss_base = generate_gaussian_logk_field(
        nx=config.nx,
        ny=config.ny,
        dx=config.dx,
        dy=config.dy,
        mean_logk=0.0,
        sigma_logk=1.0,
        corr_len_x=config.corr_len_ss_x,
        corr_len_y=config.corr_len_ss_y,
        orientation_rad=config.orientation_rad,
        seed=ss_seed,
    )
    z_k = _standard_score(k_field.logk)
    z_i = _standard_score(ss_base.logk)
    z_ss = config.log_correlation * z_k + np.sqrt(1.0 - config.log_correlation**2) * z_i
    z_ss = _standard_score(z_ss)
    raw_logss = config.mean_logss + config.sigma_logss * z_ss
    logss = np.clip(raw_logss, config.min_logss, config.max_logss)
    clipped_fraction = float(np.mean(logss != raw_logss))
    return JointLogField2D(
        x=k_field.x,
        y=k_field.y,
        logk=k_field.logk,
        logss=logss,
        config=config,
        specific_storage_clipped_fraction=clipped_fraction,
    )


def _standard_score(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    centered = arr - float(np.mean(arr))
    scale = float(np.std(centered))
    if scale <= 0.0:
        raise ValueError("random field has zero standard deviation")
    return centered / scale


def _validate_config(config: JointFieldConfig) -> None:
    if not -0.999 <= config.log_correlation <= 0.999:
        raise ValueError("log_correlation must lie between -0.999 and 0.999")
    if config.min_logss >= config.max_logss:
        raise ValueError("min_logss must be smaller than max_logss")
    if config.sigma_logss <= 0.0:
        raise ValueError("sigma_logss must be positive")
