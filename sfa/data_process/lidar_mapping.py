"""
Utility functions for handling multi-dimensional LiDAR point clouds.

Implements Scheme B (零值处理映射) described in project documentation:
将 8 维点云数据中的 P (SNR/功率强度) 映射为强度通道，并对 98.5% 的零值进行处理。
The helpers maintain backward compatibility with legacy 5D/4D formats.
"""

from __future__ import annotations

import numpy as np

SCHEME_B_ZERO_MAPPING = 0.1
SCHEME_B_NONZERO_MIN = -1.36
SCHEME_B_NONZERO_MAX = 6.43
SCHEME_B_NONZERO_RANGE = (0.2, 1.0)


def map_P_to_intensity_scheme_B(
    p_values: np.ndarray,
    zero_mapping: float = SCHEME_B_ZERO_MAPPING,
    nonzero_min: float = SCHEME_B_NONZERO_MIN,
    nonzero_max: float = SCHEME_B_NONZERO_MAX,
    nonzero_range: tuple[float, float] = SCHEME_B_NONZERO_RANGE,
) -> np.ndarray:
    """
    Map SNR/功率 (P) values from the 8D point cloud to normalized intensity values.

    Zero values are mapped to ``zero_mapping``. Non-zero values are linearly scaled
    into ``nonzero_range`` using the calibrated bounds from the scheme B document.
    """
    intensity = np.full_like(p_values, zero_mapping, dtype=np.float32)
    nonzero_mask = p_values != 0
    if not np.any(nonzero_mask):
        return intensity

    min_out, max_out = nonzero_range
    denom = nonzero_max - nonzero_min
    if denom <= 0:
        return intensity

    normalized = (p_values[nonzero_mask] - nonzero_min) / denom
    normalized = np.clip(normalized, 0.0, 1.0)
    intensity[nonzero_mask] = min_out + (max_out - min_out) * normalized
    return intensity.astype(np.float32, copy=False)


def process_8d_lidar_data_scheme_B(lidar_points: np.ndarray) -> np.ndarray:
    """
    Convert 8D LiDAR points ``[x, y, z, D, P, R, A, E]`` to the 4D representation
    ``[x, y, z, intensity]`` expected by SFA3D, applying scheme B to the P channel.
    """
    if lidar_points.ndim != 2 or lidar_points.shape[1] != 8:
        raise ValueError(f"Expecting Nx8 array for 8D lidar points, got shape {lidar_points.shape}")

    xyz = lidar_points[:, [0, 1, 2]]
    p_values = lidar_points[:, 4]
    intensity = map_P_to_intensity_scheme_B(p_values)
    return np.concatenate([xyz, intensity[:, np.newaxis]], axis=1).astype(np.float32, copy=False)


def read_lidar_file_with_fallback(lidar_file: str) -> np.ndarray:
    """
    Read LiDAR binary data with scheme B support.

    Priority:
    1. 8D data -> scheme B mapping.
    2. 5D data -> select columns 0,1,2,4.
    3. 4D data -> use as-is.

    Raises:
        ValueError: if the binary file length is not divisible by 8, 5, or 4.
    """
    lidar_data = np.fromfile(lidar_file, dtype=np.float32)
    total_vals = lidar_data.size
    if total_vals == 0:
        return lidar_data.reshape(-1, 4)

    if total_vals % 8 == 0:
        try:
            reshaped = lidar_data.reshape(-1, 8)
            return process_8d_lidar_data_scheme_B(reshaped)
        except Exception as exc:
            print(f"Warning: Failed to process LiDAR data as 8D ({exc}); trying fallback formats.")

    if total_vals % 5 == 0:
        reshaped = lidar_data.reshape(-1, 5)
        return reshaped[:, [0, 1, 2, 4]].astype(np.float32, copy=False)

    if total_vals % 4 == 0:
        return lidar_data.reshape(-1, 4).astype(np.float32, copy=False)

    raise ValueError(f"Unexpected LiDAR data length in '{lidar_file}'; "
                     f"total float32 values: {total_vals}")


__all__ = [
    "map_P_to_intensity_scheme_B",
    "process_8d_lidar_data_scheme_B",
    "read_lidar_file_with_fallback",
    "SCHEME_B_ZERO_MAPPING",
    "SCHEME_B_NONZERO_MIN",
    "SCHEME_B_NONZERO_MAX",
    "SCHEME_B_NONZERO_RANGE",
]
