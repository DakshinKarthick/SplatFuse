"""SplatFuse 3D Gaussian Splatting trainer.

This module owns the production training workflow around the differentiable
``gsplat`` CUDA rasterizer:

* discover and validate a COLMAP sparse reconstruction;
* load registered cameras, images, and the sparse point cloud;
* undistort/downscale photographs while keeping intrinsics consistent;
* initialize Gaussian means, log-scales, quaternions, opacity logits, and SH;
* optimize an L1 + SSIM objective with Adam and adaptive density control;
* validate, preview, checkpoint/resume, and export a viewer-compatible PLY.

The repository's ``renderer-cuda`` directory remains a standalone forward-pass
prototype: its backward kernel and PyTorch binding are not implemented.  This
trainer therefore uses gsplat for the differentiable CUDA boundary.  All data,
training, lifecycle, and export behavior in this file remains SplatFuse-owned.

Realistic training intentionally requires an NVIDIA CUDA device.  ``--doctor``,
``--inspect-data``, ``--self-test``, and the unit tests are CPU-safe, which lets
the project be checked on the AMD development computer.
"""

from __future__ import annotations

import argparse
import atexit
import dataclasses
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from packaging.version import InvalidVersion, Version
from PIL import Image
import torch
import torch.nn.functional as F


LOGGER = logging.getLogger("splatfuse.trainer")
FORMAT_VERSION = 1
EXPECTED_GSPLAT_VERSION = "1.5.3"
SH_C0 = 0.28209479177387814
MODEL_STEMS = ("cameras", "images", "points3D")
PINHOLE_MODELS = {"SIMPLE_PINHOLE", "PINHOLE"}
STANDARD_DISTORTION_MODELS = {
    "SIMPLE_RADIAL",
    "RADIAL",
    "OPENCV",
    "FULL_OPENCV",
}
FISHEYE_MODELS = {
    "SIMPLE_RADIAL_FISHEYE",
    "RADIAL_FISHEYE",
    "OPENCV_FISHEYE",
}
ACTIVE_RUN_LOCKS: set[Path] = set()


class UserFacingError(RuntimeError):
    """An expected failure whose message is suitable for the command line."""


@dataclass(frozen=True)
class CameraRecord:
    """One COLMAP camera model and its calibration parameters."""

    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class FrameRecord:
    """One registered photograph and its world-to-camera transform."""

    image_id: int
    camera_id: int
    name: str
    world_to_camera: np.ndarray


@dataclass(frozen=True)
class PointCloudRecord:
    """Sparse COLMAP points used to initialize trainable Gaussians."""

    xyz: np.ndarray
    rgb: np.ndarray
    errors: np.ndarray
    track_lengths: np.ndarray


@dataclass(frozen=True)
class SceneRecord:
    """Validated reconstruction and image-folder contract."""

    data_root: Path
    model_path: Path
    images_path: Path
    cameras: Mapping[int, CameraRecord]
    frames: tuple[FrameRecord, ...]
    points: PointCloudRecord
    fingerprint: str
    scene_scale: float


@dataclass(frozen=True)
class GsplatRuntime:
    """Version-checked gsplat callables loaded only for CUDA training."""

    version: str
    rasterization: Callable[..., Any]
    strategy_type: type
    reset_opacity: Callable[..., Any]
    remove: Callable[..., Any]


@dataclass
class TrainConfig:
    """Serializable training configuration populated from command-line flags."""

    data: str
    output_dir: str | None = None
    images_dir: str | None = None
    colmap_model: str | None = None
    viewer_ply: str | None = None
    resume: str | None = None
    overwrite: bool = False
    allow_data_mismatch: bool = False
    allow_version_mismatch: bool = False
    device: str = "cuda"
    iterations: int = 30_000
    seed: int = 42
    downscale: int = 1
    max_resolution: int = 1_600
    cache_images: int = 8
    val_every: int = 8
    eval_max_images: int = 8
    max_initial_points: int = 200_000
    max_gaussians: int = 2_000_000
    point_error_max: float = math.inf
    init_opacity: float = 0.1
    init_scale: float = 1.0
    sh_degree: int = 3
    sh_degree_interval: int = 1_000
    means_lr: float = 1.6e-4
    means_lr_final_factor: float = 0.01
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opacities_lr: float = 5e-2
    sh0_lr: float = 2.5e-3
    shn_lr: float = 1.25e-4
    ssim_weight: float = 0.2
    opacity_regularization: float = 0.0
    scale_regularization: float = 0.0
    gradient_clip: float = 0.0
    background: str = "random"
    rasterize_mode: str = "classic"
    packed: bool = True
    near_plane: float = 0.01
    far_plane: float = 1e10
    refine_start: int = 500
    refine_stop: int = 15_000
    refine_every: int = 100
    reset_opacity_every: int = 3_000
    pause_refine_after_reset: int = 0
    prune_opacity: float = 0.005
    grow_gradient: float = 0.0002
    grow_scale_3d: float = 0.01
    prune_scale_3d: float = 0.1
    absgrad: bool = False
    checkpoint_every: int = 1_000
    eval_every: int = 1_000
    preview_every: int = 500
    export_every: int = 5_000


# These options may change at resume because they select hardware, artifact
# cadence, output placement, or a deliberate memory-quality tradeoff.  Every
# other field is restored from the checkpoint so omitted CLI flags cannot
# silently replace the original optimizer/loss/density configuration.
RESUME_OVERRIDE_FIELDS = {
    "data",
    "output_dir",
    "images_dir",
    "colmap_model",
    "viewer_ply",
    "resume",
    "overwrite",
    "allow_data_mismatch",
    "allow_version_mismatch",
    "device",
    "iterations",
    "downscale",
    "max_resolution",
    "cache_images",
    "max_gaussians",
    "eval_max_images",
    "checkpoint_every",
    "eval_every",
    "preview_every",
    "export_every",
}
RESUME_ALWAYS_REQUESTED_FIELDS = {
    "data",
    "output_dir",
    "images_dir",
    "colmap_model",
    "viewer_ply",
    "resume",
    "overwrite",
    "allow_data_mismatch",
    "allow_version_mismatch",
    "device",
}


def restore_checkpoint_config(
    config: TrainConfig,
    saved: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Restore trajectory-defining config and report allowed CLI overrides."""

    field_names = {field.name for field in dataclasses.fields(TrainConfig)}
    required = field_names - RESUME_OVERRIDE_FIELDS
    missing = sorted(required - saved.keys())
    if missing:
        raise UserFacingError(
            "Checkpoint configuration is incomplete; missing: " + ", ".join(missing)
        )
    restored: dict[str, dict[str, Any]] = {}
    overrides: dict[str, dict[str, Any]] = {}
    explicit_fields = set(getattr(config, "_explicit_fields", ()))
    for name in sorted(field_names):
        requested = getattr(config, name)
        checkpoint_value = saved.get(name)
        if name in RESUME_OVERRIDE_FIELDS:
            explicitly_requested = (
                name in RESUME_ALWAYS_REQUESTED_FIELDS or name in explicit_fields
            )
            if not explicitly_requested and name in saved:
                if requested != checkpoint_value:
                    restored[name] = {
                        "requested_default": requested,
                        "checkpoint": checkpoint_value,
                    }
                setattr(config, name, checkpoint_value)
            elif name in saved and requested != checkpoint_value:
                overrides[name] = {"checkpoint": checkpoint_value, "requested": requested}
            continue
        if requested != checkpoint_value:
            restored[name] = {"requested": requested, "checkpoint": checkpoint_value}
        setattr(config, name, checkpoint_value)
    return {"restored_fields": restored, "allowed_overrides": overrides}


def configure_logging(verbose: bool) -> None:
    """Configure concise, timestamped console logging once."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def sanitize_scene_name(value: str) -> str:
    """Return a portable output name without silently producing an empty name."""

    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return clean or "scene"


def json_ready(value: Any) -> Any:
    """Convert dataclasses, paths, NumPy values, and tensors into JSON values."""

    if dataclasses.is_dataclass(value):
        return json_ready(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf" if value < 0 else "nan"
    return value


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Durably replace a file using a temporary sibling and ``os.replace``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write stable, human-readable JSON without exposing a partial file."""

    text = json.dumps(json_ready(payload), indent=2, sort_keys=True, ensure_ascii=False)
    atomic_write_bytes(path, (text + "\n").encode("utf-8"))


def atomic_copy_file(source: Path, destination: Path) -> None:
    """Stream-copy a file to a temporary sibling, then atomically publish it."""

    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        return
    if destination.exists() and destination.is_dir():
        raise UserFacingError(f"Publish destination is a directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with source.open("rb") as reader, temp.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def acquire_run_lock(output: Path) -> Path:
    """Claim a stable sibling lock before creating/replacing a run directory."""

    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.splatfuse-run.lock"
    payload = json.dumps(
        {
            "pid": os.getpid(),
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "command": sys.argv,
        },
        sort_keys=True,
    ).encode("utf-8")
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        try:
            details = lock.read_text(encoding="utf-8", errors="replace")[:500]
        except OSError:
            details = "unreadable lock metadata"
        raise UserFacingError(
            f"Run directory is already locked: {lock}. Active/stale lock: {details}. "
            "Verify no trainer is running before removing a stale lock."
        ) from error
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    ACTIVE_RUN_LOCKS.add(lock)
    return lock


def release_run_lock(lock: Path) -> None:
    """Release a lock owned by this process."""

    if lock in ACTIVE_RUN_LOCKS:
        lock.unlink(missing_ok=True)
        ACTIVE_RUN_LOCKS.discard(lock)


def release_all_run_locks() -> None:
    """Best-effort process-exit cleanup for CLI and unexpected failures."""

    for lock in tuple(ACTIVE_RUN_LOCKS):
        release_run_lock(lock)


atexit.register(release_all_run_locks)


def package_version(name: str) -> str | None:
    """Return installed distribution version, or ``None`` when absent."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_version_command(command: Sequence[str]) -> str | None:
    """Read the first output line of a diagnostic executable without raising."""

    if shutil.which(command[0]) is None:
        return None
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (result.stdout or result.stderr).strip().splitlines()
    return output[0] if output else None


def package_base_version(version: str | None) -> str | None:
    """Strip only a PEP 440 local tag (for example ``+pt24cu124``)."""

    if version is None:
        return None
    try:
        return Version(version).public
    except InvalidVersion:
        return version.split("+", 1)[0]


def selected_model_files(path: Path) -> dict[str, Path]:
    """Select one complete COLMAP format, never a per-file TXT/BIN mixture.

    A complete binary model wins over a complete text model, matching COLMAP's
    normal preference.  An incomplete binary set cannot shadow a valid text
    set, and the exact selected files are reused for loading and fingerprinting.
    """

    for suffix in (".bin", ".txt"):
        files = {stem: path / f"{stem}{suffix}" for stem in MODEL_STEMS}
        if all(candidate.is_file() for candidate in files.values()):
            return files
    present = sorted(
        candidate.name
        for suffix in (".bin", ".txt")
        for stem in MODEL_STEMS
        if (candidate := path / f"{stem}{suffix}").is_file()
    )
    detail = ", ".join(present) if present else "no model component files"
    raise UserFacingError(
        f"Incomplete or mixed-format COLMAP model at {path}: found {detail}. "
        "Provide cameras/images/points3D all as .bin or all as .txt."
    )


def has_colmap_model(path: Path) -> bool:
    """Return whether a directory contains all three sparse-model components."""

    if not path.is_dir():
        return False
    try:
        selected_model_files(path)
    except UserFacingError:
        return False
    return True


def model_record_count(path: Path) -> int:
    """Read a COLMAP text line count or binary uint64 record count."""

    if path.suffix == ".txt":
        return sum(
            1
            for raw in path.read_text(encoding="utf-8").splitlines()
            if raw.strip() and not raw.lstrip().startswith("#")
        )
    try:
        with path.open("rb") as handle:
            header = handle.read(8)
        if len(header) == 8:
            count = int(struct.unpack("<Q", header)[0])
            # Every COLMAP binary record occupies at least eight bytes.  This
            # rejects garbage headers before they distort model ranking.
            if count <= max(0, (path.stat().st_size - 8) // 8):
                return count
    except OSError:
        pass
    # A corrupt candidate is rejected later by pycolmap.  File size remains a
    # deterministic discovery fallback for old/incomplete capture outputs.
    return path.stat().st_size


def discover_colmap_model(data_root: Path, explicit: Path | None = None) -> Path:
    """Discover the intended sparse model with SplatFuse Phase-0 precedence.

    The fast capture script exports the *largest* reconstruction directly into
    ``sparse/`` while retaining raw ``sparse/0``, ``sparse/1``, ... models.
    Direct files must therefore win over a naive ``sparse/0`` choice.
    """

    data_root = data_root.expanduser().resolve()
    if explicit is not None:
        selected = explicit.expanduser()
        if not selected.is_absolute():
            selected = data_root / selected
        selected = selected.resolve()
        selected_model_files(selected)
        return selected

    direct_candidates = (data_root / "sparse", data_root)
    for candidate in direct_candidates:
        if has_colmap_model(candidate):
            return candidate.resolve()

    sparse_root = data_root / "sparse"
    search_root = sparse_root if sparse_root.is_dir() else data_root
    numbered = [
        item
        for item in search_root.iterdir()
        if item.is_dir() and item.name.isdigit() and has_colmap_model(item)
    ] if search_root.is_dir() else []
    if numbered:
        def model_rank(path: Path) -> tuple[int, int, int]:
            files = selected_model_files(path)
            return (
                model_record_count(files["points3D"]),
                model_record_count(files["images"]),
                -int(path.name),
            )

        selected = max(numbered, key=model_rank)
        LOGGER.warning(
            "No model was exported directly under %s; selected largest numbered model %s.",
            search_root,
            selected.name,
        )
        return selected.resolve()

    malformed = []
    for candidate in direct_candidates:
        if candidate.is_dir() and any(
            (candidate / f"{stem}{suffix}").is_file()
            for stem in MODEL_STEMS
            for suffix in (".bin", ".txt")
        ):
            malformed.append(candidate)
    if malformed:
        selected_model_files(malformed[0])  # raises the detailed format error
    raise UserFacingError(
        "No COLMAP model found. Expected cameras, images, and points3D files "
        f"under {data_root / 'sparse'} (or pass --colmap-model)."
    )


def discover_images_path(
    data_root: Path,
    model_path: Path,
    explicit: Path | None = None,
) -> Path:
    """Find the image root associated with a sparse reconstruction."""

    if explicit is not None:
        candidate = explicit.expanduser()
        if not candidate.is_absolute():
            candidate = data_root / candidate
        candidate = candidate.resolve()
        if not candidate.is_dir():
            raise UserFacingError(f"--images-dir is not a directory: {candidate}")
        return candidate

    candidates = [
        data_root / "images",
        model_path.parent / "images",
        model_path.parent.parent / "images",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise UserFacingError(
        f"No images directory found for {model_path}; pass --images-dir explicitly."
    )


def qvec_to_rotation(qvec: Sequence[float]) -> np.ndarray:
    """Convert a COLMAP WXYZ quaternion into a 3x3 rotation matrix."""

    q = np.asarray(qvec, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise UserFacingError(f"Invalid COLMAP quaternion: {qvec}")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise UserFacingError("COLMAP quaternion has zero length.")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def validate_pose(matrix: np.ndarray, context: str) -> np.ndarray:
    """Validate a finite rigid world-to-camera transform."""

    pose = np.asarray(matrix, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise UserFacingError(f"Invalid or non-finite camera pose for {context}.")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise UserFacingError(f"Camera pose for {context} has an invalid homogeneous row.")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, rel_tol=1e-5, abs_tol=1e-5
    ):
        raise UserFacingError(f"Camera pose for {context} is not a rigid rotation.")
    return pose


def parse_cameras_text(path: Path) -> dict[int, CameraRecord]:
    """Parse COLMAP ``cameras.txt`` with strict dimensions and finite values."""

    cameras: dict[int, CameraRecord] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5:
            raise UserFacingError(f"Malformed {path.name}:{line_number}: {raw}")
        camera_id = int(fields[0])
        params = tuple(float(value) for value in fields[4:])
        camera = CameraRecord(
            camera_id=camera_id,
            model=fields[1].upper(),
            width=int(fields[2]),
            height=int(fields[3]),
            params=params,
        )
        if camera_id in cameras:
            raise UserFacingError(f"Duplicate camera id {camera_id} in {path}")
        if camera.width <= 0 or camera.height <= 0 or not np.isfinite(params).all():
            raise UserFacingError(f"Invalid camera {camera_id} in {path}")
        cameras[camera_id] = camera
    if not cameras:
        raise UserFacingError(f"No cameras found in {path}")
    return cameras


def parse_images_text(path: Path) -> tuple[FrameRecord, ...]:
    """Parse the two-line COLMAP image records while allowing empty point lines."""

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    frames: list[FrameRecord] = []
    seen_ids: set[int] = set()
    index = 0
    while index < len(raw_lines):
        raw = raw_lines[index]
        index += 1
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=9)
        if len(fields) != 10:
            raise UserFacingError(f"Malformed {path.name}:{index}: {raw}")
        image_id = int(fields[0])
        if image_id in seen_ids:
            raise UserFacingError(f"Duplicate image id {image_id} in {path}")
        qvec = [float(value) for value in fields[1:5]]
        tvec = np.asarray([float(value) for value in fields[5:8]], dtype=np.float64)
        camera_id = int(fields[8])
        rotation = qvec_to_rotation(qvec)
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = tvec
        world_to_camera = validate_pose(world_to_camera, f"{path.name}:{index}")
        frames.append(
            FrameRecord(
                image_id=image_id,
                camera_id=camera_id,
                name=fields[9],
                world_to_camera=world_to_camera,
            )
        )
        seen_ids.add(image_id)
        # The following physical line is the POINTS2D record, including when empty.
        if index >= len(raw_lines):
            raise UserFacingError(
                f"Missing POINTS2D line after image {image_id} in {path.name}:{index}."
            )
        points_line_number = index + 1
        points_line = raw_lines[index].strip()
        index += 1
        if points_line.startswith("#"):
            raise UserFacingError(
                f"Expected POINTS2D at {path.name}:{points_line_number}, found a comment."
            )
        point_fields = points_line.split()
        if len(point_fields) % 3 != 0:
            raise UserFacingError(
                f"Malformed POINTS2D at {path.name}:{points_line_number}: "
                "expected X Y POINT3D_ID triples."
            )
        for point_index in range(0, len(point_fields), 3):
            try:
                xy = [float(point_fields[point_index]), float(point_fields[point_index + 1])]
                int(point_fields[point_index + 2])
            except ValueError as error:
                raise UserFacingError(
                    f"Malformed POINTS2D value at {path.name}:{points_line_number}."
                ) from error
            if not np.isfinite(xy).all():
                raise UserFacingError(
                    f"Non-finite POINTS2D value at {path.name}:{points_line_number}."
                )
    if not frames:
        raise UserFacingError(f"No registered images found in {path}")
    return tuple(frames)


def parse_points_text(path: Path) -> PointCloudRecord:
    """Parse finite sparse points and preserve quality/track metadata."""

    xyz: list[list[float]] = []
    rgb: list[list[float]] = []
    errors: list[float] = []
    track_lengths: list[int] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8 or (len(fields) - 8) % 2 != 0:
            raise UserFacingError(f"Malformed {path.name}:{line_number}: {raw}")
        point_xyz = [float(value) for value in fields[1:4]]
        point_rgb = [float(value) / 255.0 for value in fields[4:7]]
        error = float(fields[7])
        if not np.isfinite(point_xyz).all() or not np.isfinite(error):
            continue
        xyz.append(point_xyz)
        rgb.append(point_rgb)
        errors.append(error)
        try:
            for track_index in range(8, len(fields), 2):
                int(fields[track_index])
                int(fields[track_index + 1])
        except ValueError as error:
            raise UserFacingError(
                f"Malformed track pair in {path.name}:{line_number}."
            ) from error
        track_lengths.append((len(fields) - 8) // 2)
    if len(xyz) < 4:
        raise UserFacingError(f"Need at least four finite sparse points in {path}")
    return PointCloudRecord(
        xyz=np.asarray(xyz, dtype=np.float64),
        rgb=np.asarray(rgb, dtype=np.float32),
        errors=np.asarray(errors, dtype=np.float32),
        track_lengths=np.asarray(track_lengths, dtype=np.int32),
    )


def _pycolmap_pose_matrix(image: Any) -> np.ndarray:
    """Extract a 4x4 world-to-camera matrix across supported pycolmap APIs."""

    pose = getattr(image, "cam_from_world", None)
    pose = pose() if callable(pose) else pose
    if pose is not None:
        matrix = getattr(pose, "matrix", None)
        matrix = matrix() if callable(matrix) else matrix
        if matrix is not None:
            matrix_np = np.asarray(matrix, dtype=np.float64)
            result = np.eye(4, dtype=np.float64)
            result[:3, :4] = matrix_np[:3, :4]
            return validate_pose(result, f"pycolmap image {getattr(image, 'image_id', '?')}")
    if hasattr(image, "qvec") and hasattr(image, "tvec"):
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = qvec_to_rotation(np.asarray(image.qvec))
        result[:3, 3] = np.asarray(image.tvec, dtype=np.float64)
        return validate_pose(result, f"pycolmap image {getattr(image, 'image_id', '?')}")
    raise UserFacingError("Unsupported pycolmap Image pose API.")


def load_model_pycolmap(
    model_path: Path,
) -> tuple[dict[int, CameraRecord], tuple[FrameRecord, ...], PointCloudRecord]:
    """Load text or binary sparse models through the official Python bindings."""

    try:
        import pycolmap  # type: ignore[import-not-found]
    except ImportError as error:
        raise UserFacingError(
            "Binary COLMAP input requires pycolmap==3.12.6. Install trainer/requirements.txt "
            "or export the model to TXT with `colmap model_converter`."
        ) from error
    reconstruction = pycolmap.Reconstruction(str(model_path))
    cameras: dict[int, CameraRecord] = {}
    for camera_id, camera in reconstruction.cameras.items():
        model_value = getattr(camera, "model", "")
        model_name = getattr(model_value, "name", str(model_value)).upper()
        cameras[int(camera_id)] = CameraRecord(
            camera_id=int(camera_id),
            model=model_name,
            width=int(camera.width),
            height=int(camera.height),
            params=tuple(float(value) for value in np.asarray(camera.params).reshape(-1)),
        )

    registered_ids = getattr(reconstruction, "reg_image_ids", None)
    image_ids = registered_ids() if callable(registered_ids) else reconstruction.images.keys()
    frames: list[FrameRecord] = []
    for image_id in image_ids:
        image = reconstruction.images[int(image_id)]
        frames.append(
            FrameRecord(
                image_id=int(image_id),
                camera_id=int(image.camera_id),
                name=str(image.name),
                world_to_camera=_pycolmap_pose_matrix(image),
            )
        )

    xyz: list[np.ndarray] = []
    rgb: list[np.ndarray] = []
    errors: list[float] = []
    track_lengths: list[int] = []
    for point in reconstruction.points3D.values():
        point_xyz = np.asarray(point.xyz, dtype=np.float64).reshape(3)
        error = float(getattr(point, "error", 0.0))
        if not np.isfinite(point_xyz).all() or not math.isfinite(error):
            continue
        xyz.append(point_xyz)
        rgb.append(np.asarray(point.color, dtype=np.float64).reshape(3) / 255.0)
        track = getattr(point, "track", None)
        length = getattr(track, "length", None)
        track_lengths.append(int(length() if callable(length) else len(getattr(track, "elements", []))))
        errors.append(error)
    if not cameras or not frames or len(xyz) < 4:
        raise UserFacingError(f"Incomplete COLMAP reconstruction: {model_path}")
    return (
        cameras,
        tuple(frames),
        PointCloudRecord(
            xyz=np.asarray(xyz, dtype=np.float64),
            rgb=np.asarray(rgb, dtype=np.float32),
            errors=np.asarray(errors, dtype=np.float32),
            track_lengths=np.asarray(track_lengths, dtype=np.int32),
        ),
    )


def load_colmap_model(
    model_path: Path,
) -> tuple[dict[int, CameraRecord], tuple[FrameRecord, ...], PointCloudRecord]:
    """Load a complete model, using the dependency-free parser for TXT input."""

    selected = selected_model_files(model_path)
    if selected["cameras"].suffix == ".txt":
        return (
            parse_cameras_text(selected["cameras"]),
            parse_images_text(selected["images"]),
            parse_points_text(selected["points3D"]),
        )
    return load_model_pycolmap(model_path)


def normalize_image_name(name: str) -> PurePosixPath:
    """Normalize Windows COLMAP separators and reject path traversal."""

    normalized = PurePosixPath(name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts or not normalized.parts:
        raise UserFacingError(f"Unsafe image path in COLMAP model: {name!r}")
    return normalized


def resolve_frame_path(images_path: Path, name: str) -> Path:
    """Resolve a registered image while ensuring it remains under the image root."""

    relative = normalize_image_name(name)
    candidate = images_path.joinpath(*relative.parts).resolve()
    try:
        candidate.relative_to(images_path.resolve())
    except ValueError as error:
        raise UserFacingError(f"Image escapes --images-dir: {name!r}") from error
    return candidate


def compute_scene_scale(frames: Sequence[FrameRecord], points: np.ndarray) -> float:
    """Estimate a stable world-space scale from camera centers, then points."""

    centers = []
    for frame in frames:
        rotation = frame.world_to_camera[:3, :3]
        translation = frame.world_to_camera[:3, 3]
        centers.append(-rotation.T @ translation)
    centers_np = np.asarray(centers, dtype=np.float64)
    if len(centers_np) >= 2:
        origin = np.median(centers_np, axis=0)
        distances = np.linalg.norm(centers_np - origin, axis=1)
        scale = float(np.percentile(distances, 90))
        if math.isfinite(scale) and scale > 1e-6:
            return scale
    point_origin = np.median(points, axis=0)
    point_distances = np.linalg.norm(points - point_origin, axis=1)
    scale = float(np.percentile(point_distances, 90))
    return scale if math.isfinite(scale) and scale > 1e-6 else 1.0


def hash_file(path: Path, digest: Any) -> None:
    """Stream a file into an existing SHA-256 digest."""

    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)


def scene_fingerprint(model_path: Path, images_path: Path, frames: Sequence[FrameRecord]) -> str:
    """Fingerprint geometry plus registered image identities/sizes for safe resume."""

    digest = hashlib.sha256()
    selected = selected_model_files(model_path)
    suffix = selected["cameras"].suffix
    model_inputs = list(selected.values())
    model_inputs.extend(
        candidate
        for stem in ("rigs", "frames")
        if (candidate := model_path / f"{stem}{suffix}").is_file()
    )
    for path in model_inputs:
        digest.update(path.name.encode("utf-8"))
        hash_file(path, digest)
    for frame in sorted(frames, key=lambda item: item.image_id):
        path = resolve_frame_path(images_path, frame.name)
        digest.update(f"{frame.image_id}:{frame.name}:{path.stat().st_size}".encode("utf-8"))
        hash_file(path, digest)
    return digest.hexdigest()


def load_scene(config: TrainConfig) -> SceneRecord:
    """Discover, parse, and fully validate the requested training scene."""

    data_root = Path(config.data).expanduser().resolve()
    if not data_root.is_dir():
        raise UserFacingError(f"--data is not a directory: {data_root}")
    explicit_model = Path(config.colmap_model) if config.colmap_model else None
    model_path = discover_colmap_model(data_root, explicit_model)
    explicit_images = Path(config.images_dir) if config.images_dir else None
    images_path = discover_images_path(data_root, model_path, explicit_images)
    cameras, frames, points = load_colmap_model(model_path)

    valid_frames: list[FrameRecord] = []
    missing: list[str] = []
    for frame in frames:
        if frame.camera_id not in cameras:
            raise UserFacingError(
                f"Image {frame.image_id} references missing camera {frame.camera_id}."
            )
        validate_pose(frame.world_to_camera, f"COLMAP image {frame.image_id}")
        camera_intrinsics(cameras[frame.camera_id])
        path = resolve_frame_path(images_path, frame.name)
        if path.is_file():
            valid_frames.append(frame)
        else:
            missing.append(frame.name)
    if missing:
        example = ", ".join(missing[:3])
        raise UserFacingError(
            f"{len(missing)} registered COLMAP images are missing under {images_path}; "
            f"examples: {example}"
        )
    if not valid_frames:
        raise UserFacingError("The COLMAP model has no registered images available for training.")

    finite = (
        np.isfinite(points.xyz).all(axis=1)
        & np.isfinite(points.rgb).all(axis=1)
        & np.isfinite(points.errors)
        & (points.errors <= config.point_error_max)
    )
    points = PointCloudRecord(
        xyz=points.xyz[finite],
        rgb=np.clip(points.rgb[finite], 0.0, 1.0),
        errors=points.errors[finite],
        track_lengths=points.track_lengths[finite],
    )
    if len(points.xyz) < 4:
        raise UserFacingError(
            "Fewer than four sparse points remain after --point-error-max filtering."
        )
    valid_frames.sort(key=lambda item: (item.name.casefold(), item.image_id))
    scale = compute_scene_scale(valid_frames, points.xyz)
    coordinate_magnitudes = [np.abs(points.xyz).reshape(-1)]
    for frame in valid_frames:
        rotation = frame.world_to_camera[:3, :3]
        translation = frame.world_to_camera[:3, 3]
        coordinate_magnitudes.append(np.abs(-rotation.T @ translation).reshape(-1))
    max_coordinate = float(np.max(np.concatenate(coordinate_magnitudes)))
    float32_spacing = float(np.spacing(np.float32(max_coordinate)))
    if float32_spacing > scale * 1e-4:
        raise UserFacingError(
            "COLMAP world coordinates are too large relative to the scene extent for "
            "float32 CUDA training. Recenter/normalize the reconstruction before training "
            f"(float32 spacing {float32_spacing:.6g}, scene scale {scale:.6g})."
        )
    fingerprint = scene_fingerprint(model_path, images_path, valid_frames)
    return SceneRecord(
        data_root=data_root,
        model_path=model_path,
        images_path=images_path,
        cameras=cameras,
        frames=tuple(valid_frames),
        points=points,
        fingerprint=fingerprint,
        scene_scale=scale,
    )


def camera_intrinsics(camera: CameraRecord) -> tuple[np.ndarray, np.ndarray, str]:
    """Convert supported COLMAP camera parameters to OpenCV calibration form."""

    model = camera.model.upper()
    params = np.asarray(camera.params, dtype=np.float64)

    def require(count: int) -> None:
        if len(params) != count:
            raise UserFacingError(
                f"Camera {camera.camera_id} model {model} expects {count} parameters, "
                f"received {len(params)}."
            )

    distortion_kind = "none"
    distortion = np.zeros(8, dtype=np.float64)
    if model == "SIMPLE_PINHOLE":
        require(3)
        fx = fy = params[0]
        cx, cy = params[1:3]
    elif model == "PINHOLE":
        require(4)
        fx, fy, cx, cy = params
    elif model == "SIMPLE_RADIAL":
        require(4)
        fx = fy = params[0]
        cx, cy, distortion[0] = params[1:4]
        distortion_kind = "standard"
    elif model == "RADIAL":
        require(5)
        fx = fy = params[0]
        cx, cy, distortion[0], distortion[1] = params[1:5]
        distortion_kind = "standard"
    elif model == "OPENCV":
        require(8)
        fx, fy, cx, cy = params[:4]
        distortion[:4] = params[4:8]  # k1, k2, p1, p2
        distortion_kind = "standard"
    elif model == "FULL_OPENCV":
        require(12)
        fx, fy, cx, cy = params[:4]
        distortion[:] = params[4:12]  # k1,k2,p1,p2,k3,k4,k5,k6
        distortion_kind = "standard"
    elif model == "SIMPLE_RADIAL_FISHEYE":
        require(4)
        fx = fy = params[0]
        cx, cy = params[1:3]
        distortion[0] = params[3]
        distortion_kind = "fisheye"
    elif model == "RADIAL_FISHEYE":
        require(5)
        fx = fy = params[0]
        cx, cy = params[1:3]
        distortion[:2] = params[3:5]
        distortion_kind = "fisheye"
    elif model == "OPENCV_FISHEYE":
        require(8)
        fx, fy, cx, cy = params[:4]
        distortion[:4] = params[4:8]
        distortion_kind = "fisheye"
    else:
        raise UserFacingError(
            f"Camera model {model} is not supported by the trainer. Run COLMAP's "
            "image_undistorter to create a PINHOLE workspace, or use one of: "
            f"{', '.join(sorted(PINHOLE_MODELS | STANDARD_DISTORTION_MODELS | FISHEYE_MODELS))}."
        )
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        raise UserFacingError(f"Invalid intrinsics for camera {camera.camera_id}.")
    intrinsic = np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return intrinsic, distortion, distortion_kind


def scale_intrinsics(intrinsic: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Scale COLMAP intrinsics, whose image coordinates use pixel corners."""

    result = intrinsic.copy()
    result[0, :] *= sx
    result[1, :] *= sy
    return result


def colmap_to_opencv_intrinsics(intrinsic: np.ndarray) -> np.ndarray:
    """Convert COLMAP corner coordinates to OpenCV integer-center coordinates."""

    result = intrinsic.copy()
    result[0, 2] -= 0.5
    result[1, 2] -= 0.5
    return result


def opencv_to_colmap_intrinsics(intrinsic: np.ndarray) -> np.ndarray:
    """Convert OpenCV integer-center coordinates back to COLMAP coordinates."""

    result = intrinsic.copy()
    result[0, 2] += 0.5
    result[1, 2] += 0.5
    return result


def import_cv2() -> Any:
    """Import OpenCV with an actionable dependency error."""

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise UserFacingError(
            "Image resizing/undistortion requires opencv-python-headless. "
            "Install trainer/requirements.txt."
        ) from error
    return cv2


class ImageStore:
    """Small LRU cache that returns prepared HWC float images and matching K."""

    def __init__(self, scene: SceneRecord, config: TrainConfig) -> None:
        self.scene = scene
        self.config = config
        self.cache: OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    def _read_rgb(self, path: Path) -> np.ndarray:
        with Image.open(path) as source:
            # COLMAP calibration/poses describe the decoded raster dimensions.
            # Applying EXIF orientation here would rotate pixels without also
            # transforming K and the camera pose.
            image = source
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, rgba).convert("RGB")
            else:
                image = image.convert("RGB")
            return np.array(image, dtype=np.uint8, copy=True)

    def _prepare(self, frame: FrameRecord) -> tuple[torch.Tensor, torch.Tensor]:
        path = resolve_frame_path(self.scene.images_path, frame.name)
        rgb = self._read_rgb(path)
        source_h, source_w = rgb.shape[:2]
        camera = self.scene.cameras[frame.camera_id]
        intrinsic, distortion, distortion_kind = camera_intrinsics(camera)
        if (source_w, source_h) != (camera.width, camera.height):
            sx = source_w / camera.width
            sy = source_h / camera.height
            if not math.isclose(sx, sy, rel_tol=1e-3, abs_tol=1e-6):
                raise UserFacingError(
                    f"Image {frame.name} is {source_w}x{source_h}, but camera "
                    f"{camera.camera_id} declares {camera.width}x{camera.height}. "
                    "The aspect ratio changed (crop/rotation); regenerate COLMAP "
                    "calibration instead of applying anisotropic intrinsics."
                )
            LOGGER.warning(
                "Image %s is %dx%d but camera %d declares %dx%d; scaling K.",
                frame.name,
                source_w,
                source_h,
                camera.camera_id,
                camera.width,
                camera.height,
            )
            intrinsic = scale_intrinsics(
                intrinsic,
                sx,
                sy,
            )

        scale = 1.0 / self.config.downscale
        if self.config.max_resolution > 0:
            scale = min(scale, self.config.max_resolution / max(source_w, source_h))
        scale = min(1.0, scale)
        output_w = max(1, int(round(source_w * scale)))
        output_h = max(1, int(round(source_h * scale)))
        output_k = scale_intrinsics(intrinsic, output_w / source_w, output_h / source_h)
        cv2 = import_cv2()
        if distortion_kind == "standard" and np.any(np.abs(distortion) > 1e-14):
            cv_intrinsic = colmap_to_opencv_intrinsics(intrinsic)
            cv_output_k, _ = cv2.getOptimalNewCameraMatrix(
                cv_intrinsic,
                distortion,
                (source_w, source_h),
                0.0,
                (output_w, output_h),
            )
            map_x, map_y = cv2.initUndistortRectifyMap(
                cv_intrinsic,
                distortion,
                None,
                cv_output_k,
                (output_w, output_h),
                cv2.CV_32FC1,
            )
            prepared = cv2.remap(
                rgb,
                map_x,
                map_y,
                interpolation=cv2.INTER_AREA,
                borderMode=cv2.BORDER_CONSTANT,
            )
            output_k = opencv_to_colmap_intrinsics(cv_output_k)
        elif distortion_kind == "fisheye":
            cv_intrinsic = colmap_to_opencv_intrinsics(intrinsic)
            cv_output_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                cv_intrinsic,
                distortion[:4],
                (source_w, source_h),
                np.eye(3, dtype=np.float64),
                balance=0.0,
                new_size=(output_w, output_h),
            )
            map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
                cv_intrinsic,
                distortion[:4],
                np.eye(3, dtype=np.float64),
                cv_output_k,
                (output_w, output_h),
                cv2.CV_32FC1,
            )
            prepared = cv2.remap(
                rgb,
                map_x,
                map_y,
                interpolation=cv2.INTER_AREA,
                borderMode=cv2.BORDER_CONSTANT,
            )
            output_k = opencv_to_colmap_intrinsics(cv_output_k)
        elif (output_w, output_h) != (source_w, source_h):
            prepared = cv2.resize(rgb, (output_w, output_h), interpolation=cv2.INTER_AREA)
        else:
            prepared = rgb
        image = torch.from_numpy(np.ascontiguousarray(prepared)).to(torch.float32).div_(255.0)
        k_tensor = torch.from_numpy(output_k.astype(np.float32))
        return image, k_tensor

    def get(self, frame: FrameRecord) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a cached prepared image/K pair, evicting the oldest if needed."""

        if frame.image_id in self.cache:
            value = self.cache.pop(frame.image_id)
            self.cache[frame.image_id] = value
            return value
        value = self._prepare(frame)
        if self.config.cache_images > 0:
            self.cache[frame.image_id] = value
            while len(self.cache) > self.config.cache_images:
                self.cache.popitem(last=False)
        return value


def preflight_images(
    store: ImageStore,
    frames: Sequence[FrameRecord],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode and calibrate every registered image before GPU optimization."""

    first: tuple[torch.Tensor, torch.Tensor] | None = None
    for frame in frames:
        try:
            prepared = store.get(frame)
        except UserFacingError:
            raise
        except Exception as error:
            raise UserFacingError(
                f"Unable to decode/prepare registered image {frame.name!r}: {error}"
            ) from error
        if first is None:
            first = prepared
    if first is None:
        raise UserFacingError("No registered images were available for preflight.")
    return first


def split_frames(
    frames: Sequence[FrameRecord], val_every: int
) -> tuple[tuple[FrameRecord, ...], tuple[FrameRecord, ...]]:
    """Create a deterministic holdout without leaving tiny scenes untrainable."""

    ordered = tuple(sorted(frames, key=lambda item: (item.name.casefold(), item.image_id)))
    if val_every <= 0 or len(ordered) < 4:
        return ordered, ()
    validation = tuple(frame for index, frame in enumerate(ordered) if index % val_every == 0)
    validation_ids = {frame.image_id for frame in validation}
    training = tuple(frame for frame in ordered if frame.image_id not in validation_ids)
    if not training:
        return ordered, ()
    return training, validation


def estimate_log_scales(points: np.ndarray, init_scale: float, scene_scale: float) -> np.ndarray:
    """Estimate isotropic log-scales from three nearest neighbors without O(N²)."""

    count = len(points)
    if count < 2:
        raise UserFacingError("At least two points are required for scale initialization.")
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore[import-not-found]
    except ImportError as error:
        raise UserFacingError(
            "Scale initialization requires scikit-learn. Install trainer/requirements.txt."
        ) from error
    neighbor_count = min(4, count)
    neighbors = NearestNeighbors(n_neighbors=neighbor_count, algorithm="auto", n_jobs=-1)
    distances, _ = neighbors.fit(points).kneighbors(points, return_distance=True)
    neighbor_distances = distances[:, 1:]
    squared_mean = np.mean(np.square(neighbor_distances), axis=1)
    distance = np.sqrt(np.maximum(squared_mean, 0.0))
    positive = distance[np.isfinite(distance) & (distance > 1e-12)]
    fallback = float(np.median(positive)) if len(positive) else scene_scale * 1e-3
    floor = max(scene_scale * 1e-6, 1e-9)
    distance = np.where(np.isfinite(distance) & (distance > floor), distance, fallback)
    scales = np.maximum(distance * init_scale, floor)
    return np.log(scales).astype(np.float32)[:, None].repeat(3, axis=1)


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert linear/base RGB into the degree-zero SH coefficient convention."""

    return (rgb - 0.5) / SH_C0


def initialize_splats(
    scene: SceneRecord,
    config: TrainConfig,
    device: torch.device,
) -> torch.nn.ParameterDict:
    """Initialize all raw/unbounded Gaussian parameters from the sparse cloud."""

    xyz = scene.points.xyz
    rgb = scene.points.rgb
    initial_limit = min(config.max_initial_points, config.max_gaussians)
    if len(xyz) > initial_limit:
        generator = np.random.default_rng(config.seed)
        indices = np.sort(
            generator.choice(len(xyz), size=initial_limit, replace=False)
        )
        xyz = xyz[indices]
        rgb = rgb[indices]
        LOGGER.info("Subsampled sparse cloud from %d to %d points.", len(scene.points.xyz), len(xyz))
    log_scales = estimate_log_scales(xyz, config.init_scale, scene.scene_scale)
    count = len(xyz)
    means = torch.from_numpy(np.ascontiguousarray(xyz)).to(device=device, dtype=torch.float32)
    colors = torch.from_numpy(np.ascontiguousarray(rgb)).to(device=device, dtype=torch.float32)
    quats = torch.zeros((count, 4), device=device, dtype=torch.float32)
    quats[:, 0] = 1.0
    opacity_value = min(max(config.init_opacity, 1e-6), 1.0 - 1e-6)
    opacities = torch.full((count,), torch.logit(torch.tensor(opacity_value)).item(), device=device)
    coefficient_count = (config.sh_degree + 1) ** 2
    sh = torch.zeros((count, coefficient_count, 3), device=device, dtype=torch.float32)
    sh[:, 0, :] = rgb_to_sh(colors)
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(torch.from_numpy(log_scales).to(device)),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opacities),
            "sh0": torch.nn.Parameter(sh[:, :1, :]),
            "shN": torch.nn.Parameter(sh[:, 1:, :]),
        }
    )


def make_optimizers(
    splats: torch.nn.ParameterDict,
    config: TrainConfig,
    scene_scale: float,
) -> dict[str, torch.optim.Optimizer]:
    """Create one Adam instance per parameter, as gsplat strategies require."""

    learning_rates = {
        "means": config.means_lr * scene_scale,
        "scales": config.scales_lr,
        "quats": config.quats_lr,
        "opacities": config.opacities_lr,
        "sh0": config.sh0_lr,
        "shN": config.shn_lr,
    }
    return {
        name: torch.optim.Adam(
            [{"params": [splats[name]], "lr": learning_rates[name], "name": name}],
            eps=1e-15,
        )
        for name in splats.keys()
    }


def load_gsplat_runtime(allow_version_mismatch: bool) -> GsplatRuntime:
    """Load the pinned differentiable backend without triggering it for diagnostics."""

    version = package_version("gsplat")
    if version is None:
        raise UserFacingError(
            "gsplat is not installed. Install the CUDA PyTorch build first, then run "
            "`pip install -r trainer/requirements.txt`."
        )
    if package_base_version(version) != EXPECTED_GSPLAT_VERSION and not allow_version_mismatch:
        raise UserFacingError(
            f"Expected gsplat=={EXPECTED_GSPLAT_VERSION}, found {version}. API and density "
            "strategy behavior are version-sensitive; use the pinned requirements or pass "
            "--allow-version-mismatch after validating the target stack."
        )
    try:
        from gsplat.rendering import rasterization  # type: ignore[import-not-found]
        from gsplat.strategy import DefaultStrategy  # type: ignore[import-not-found]
        from gsplat.strategy.ops import remove, reset_opa  # type: ignore[import-not-found]
    except (ImportError, OSError) as error:
        raise UserFacingError(f"Unable to import gsplat CUDA backend: {error}") from error
    return GsplatRuntime(
        version=version,
        rasterization=rasterization,
        strategy_type=DefaultStrategy,
        reset_opacity=reset_opa,
        remove=remove,
    )


def make_strategy(runtime: GsplatRuntime, config: TrainConfig) -> Any:
    """Configure original-3DGS-style clone/split/prune/reset density control."""

    return runtime.strategy_type(
        prune_opa=config.prune_opacity,
        grow_grad2d=config.grow_gradient,
        grow_scale3d=config.grow_scale_3d,
        prune_scale3d=config.prune_scale_3d,
        refine_start_iter=config.refine_start,
        refine_stop_iter=config.refine_stop,
        reset_every=config.reset_opacity_every,
        refine_every=config.refine_every,
        pause_refine_after_reset=config.pause_refine_after_reset,
        absgrad=config.absgrad,
        verbose=True,
    )


def ssim(image_a: torch.Tensor, image_b: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Compute differentiable channel-averaged structural similarity on NCHW images."""

    if image_a.shape != image_b.shape or image_a.ndim != 4:
        raise ValueError("SSIM inputs must have the same NCHW shape.")
    height, width = image_a.shape[-2:]
    size = min(window_size, height, width)
    if size % 2 == 0:
        size -= 1
    size = max(size, 1)
    padding = size // 2
    mu_a = F.avg_pool2d(image_a, size, stride=1, padding=padding)
    mu_b = F.avg_pool2d(image_b, size, stride=1, padding=padding)
    mu_a_sq = mu_a.square()
    mu_b_sq = mu_b.square()
    mu_ab = mu_a * mu_b
    sigma_a = F.avg_pool2d(image_a.square(), size, 1, padding) - mu_a_sq
    sigma_b = F.avg_pool2d(image_b.square(), size, 1, padding) - mu_b_sq
    sigma_ab = F.avg_pool2d(image_a * image_b, size, 1, padding) - mu_ab
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a + sigma_b + c2)
    return (numerator / denominator.clamp_min(1e-12)).mean()


def image_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    ssim_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted loss, L1 component, and SSIM score for NHWC images."""

    l1 = F.l1_loss(rendered, target)
    score = ssim(rendered.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2))
    total = (1.0 - ssim_weight) * l1 + ssim_weight * (1.0 - score)
    return total, l1, score


def psnr(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute peak signal-to-noise ratio for [0,1] images."""

    mse = F.mse_loss(rendered, target).clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


def choose_background(config: TrainConfig, device: torch.device) -> torch.Tensor:
    """Return a per-view RGB background tensor."""

    if config.background == "white":
        return torch.ones((1, 3), device=device)
    if config.background == "black":
        return torch.zeros((1, 3), device=device)
    return torch.rand((1, 3), device=device)


def render_frame(
    runtime: GsplatRuntime,
    splats: torch.nn.ParameterDict,
    frame: FrameRecord,
    image: torch.Tensor,
    intrinsic: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
    sh_degree: int,
    background: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Rasterize one registered camera using raw trainable Gaussian parameters."""

    height, width = image.shape[:2]
    view = torch.from_numpy(frame.world_to_camera.astype(np.float32)).to(device).unsqueeze(0)
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    rendered, alpha, info = runtime.rasterization(
        means=splats["means"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=colors,
        viewmats=view,
        Ks=intrinsic.to(device).unsqueeze(0),
        width=width,
        height=height,
        packed=config.packed,
        absgrad=config.absgrad,
        sparse_grad=False,
        rasterize_mode=config.rasterize_mode,
        near_plane=config.near_plane,
        far_plane=config.far_plane,
        render_mode="RGB",
        sh_degree=sh_degree,
    )
    rendered = rendered + background[:, None, None, :] * (1.0 - alpha)
    return rendered, alpha, info


def all_gradients_finite(splats: torch.nn.ParameterDict) -> bool:
    """Return false if any populated gradient contains NaN or infinity."""

    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in splats.values()
    )


def enforce_gaussian_limit(
    runtime: GsplatRuntime,
    splats: torch.nn.ParameterDict,
    optimizers: Mapping[str, torch.optim.Optimizer],
    strategy_state: dict[str, Any],
    maximum: int,
) -> int:
    """Prune the lowest-opacity excess splats after a refinement operation."""

    count = len(splats["means"])
    if maximum <= 0 or count <= maximum:
        return 0
    remove_count = count - maximum
    opacities = torch.sigmoid(splats["opacities"].detach())
    indices = torch.topk(opacities, k=remove_count, largest=False).indices
    mask = torch.zeros(count, dtype=torch.bool, device=opacities.device)
    mask[indices] = True
    runtime.remove(params=splats, optimizers=dict(optimizers), state=strategy_state, mask=mask)
    LOGGER.warning("Pruned %d splats to enforce --max-gaussians=%d.", remove_count, maximum)
    return remove_count


def recursive_cpu(value: Any) -> Any:
    """Move every tensor in nested checkpoint state to CPU for portability."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: recursive_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(recursive_cpu(item) for item in value)
    if isinstance(value, list):
        return [recursive_cpu(item) for item in value]
    return value


def atomic_torch_save(path: Path, payload: Any) -> None:
    """Atomically serialize a checkpoint through a temporary sibling."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temp.open("wb") as handle:
            torch.save(recursive_cpu(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def capture_rng_state(
    sampler: random.Random,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Capture host streams and only the active training GPU stream."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "sampler": sampler.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        active = device if device is not None and device.type == "cuda" else torch.cuda.current_device()
        state["cuda"] = torch.cuda.get_rng_state(active)
    return state


def restore_rng_state(
    state: Mapping[str, Any],
    sampler: random.Random,
    device: torch.device | None = None,
    legacy_device_index: int | None = None,
) -> None:
    """Restore checkpoint RNG streams on the current device stack."""

    random.setstate(state["python"])
    sampler.setstate(state["sampler"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        active = device if device is not None and device.type == "cuda" else torch.cuda.current_device()
        cuda_state = state["cuda"]
        # Accept early SplatFuse checkpoints that stored all visible devices.
        if isinstance(cuda_state, (list, tuple)):
            saved_index = legacy_device_index if legacy_device_index is not None else 0
            if not 0 <= saved_index < len(cuda_state):
                LOGGER.warning(
                    "Legacy checkpoint has no CUDA RNG stream %d; using stream 0.",
                    saved_index,
                )
                saved_index = 0
            cuda_state = cuda_state[saved_index]
        torch.cuda.set_rng_state(cuda_state.cpu(), active)


def checkpoint_payload(
    step: int,
    splats: torch.nn.ParameterDict,
    optimizers: Mapping[str, torch.optim.Optimizer],
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    strategy_state: Mapping[str, Any],
    config: TrainConfig,
    scene: SceneRecord,
    train_frames: Sequence[FrameRecord],
    val_frames: Sequence[FrameRecord],
    sampler: random.Random,
) -> dict[str, Any]:
    """Build a complete post-step checkpoint suitable for exact continuation."""

    return {
        "format_version": FORMAT_VERSION,
        "completed_step": step,
        "splats": splats.state_dict(),
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "scheduler": scheduler.state_dict(),
        "strategy_state": dict(strategy_state),
        "config": dataclasses.asdict(config),
        "scene_fingerprint": scene.fingerprint,
        "train_image_ids": [frame.image_id for frame in train_frames],
        "val_image_ids": [frame.image_id for frame in val_frames],
        "rng": capture_rng_state(sampler, splats["means"].device),
    }


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load and validate the SplatFuse checkpoint envelope."""

    if not path.is_file():
        raise UserFacingError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise UserFacingError(f"Unsupported or invalid checkpoint: {path}")
    required = {
        "completed_step",
        "splats",
        "optimizers",
        "scheduler",
        "strategy_state",
        "config",
        "scene_fingerprint",
        "train_image_ids",
        "val_image_ids",
        "rng",
    }
    missing = required - payload.keys()
    if missing:
        raise UserFacingError(f"Checkpoint is missing: {', '.join(sorted(missing))}")
    completed_step = payload["completed_step"]
    if isinstance(completed_step, bool) or not isinstance(completed_step, int) or completed_step < 0:
        raise UserFacingError("Checkpoint completed_step must be a non-negative integer.")
    mapping_fields = ("splats", "optimizers", "scheduler", "strategy_state", "config", "rng")
    invalid_mappings = [name for name in mapping_fields if not isinstance(payload[name], Mapping)]
    if invalid_mappings:
        raise UserFacingError(
            f"Checkpoint fields must be mappings: {', '.join(invalid_mappings)}."
        )
    if not isinstance(payload["scene_fingerprint"], str) or not payload["scene_fingerprint"]:
        raise UserFacingError("Checkpoint scene_fingerprint must be a non-empty string.")
    for name in ("train_image_ids", "val_image_ids"):
        values = payload[name]
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise UserFacingError(f"Checkpoint {name} must be a list of integer IDs.")
    rng_required = {"python", "sampler", "numpy", "torch"}
    missing_rng = rng_required - payload["rng"].keys()
    if missing_rng:
        raise UserFacingError(
            f"Checkpoint RNG state is missing: {', '.join(sorted(missing_rng))}."
        )
    return payload


def validate_checkpoint_dynamic_payload(payload: Mapping[str, Any]) -> int:
    """Validate CPU-side optimizer/refinement arrays before CUDA allocation."""

    tensors = payload["splats"]
    expected = {"means", "scales", "quats", "opacities", "sh0", "shN"}
    if set(tensors) != expected:
        raise UserFacingError("Checkpoint Gaussian parameter keys do not match this trainer.")
    count = len(tensors["means"])
    optimizers = payload["optimizers"]
    if set(optimizers) != expected:
        raise UserFacingError("Checkpoint optimizer keys do not match Gaussian parameters.")
    for name, optimizer_payload in optimizers.items():
        if not isinstance(optimizer_payload, Mapping):
            raise UserFacingError(f"Checkpoint optimizer {name} is invalid.")
        states = optimizer_payload.get("state")
        groups = optimizer_payload.get("param_groups")
        if not isinstance(states, Mapping) or not isinstance(groups, list):
            raise UserFacingError(f"Checkpoint optimizer {name} has an invalid envelope.")
        for state in states.values():
            if not isinstance(state, Mapping):
                raise UserFacingError(f"Checkpoint optimizer {name} state is invalid.")
            for key, value in state.items():
                if not isinstance(value, torch.Tensor):
                    continue
                if not bool(torch.isfinite(value).all()):
                    raise UserFacingError(f"Checkpoint optimizer {name}.{key} is non-finite.")
                if value.ndim > 0 and tuple(value.shape) != tuple(tensors[name].shape):
                    raise UserFacingError(
                        f"Checkpoint optimizer {name}.{key} has shape {tuple(value.shape)}; "
                        f"expected {tuple(tensors[name].shape)}."
                    )
    strategy_state = payload["strategy_state"]
    for key in ("grad2d", "count"):
        if key not in strategy_state:
            raise UserFacingError(f"Checkpoint strategy state is missing {key}.")
    scene_scale = strategy_state.get("scene_scale")
    if isinstance(scene_scale, torch.Tensor):
        if scene_scale.numel() != 1:
            raise UserFacingError("Checkpoint strategy scene_scale must be scalar.")
        scene_scale = float(scene_scale)
    if not isinstance(scene_scale, (int, float)) or not math.isfinite(scene_scale) or scene_scale <= 0:
        raise UserFacingError("Checkpoint strategy scene_scale must be finite and positive.")
    for key in ("grad2d", "count", "radii"):
        value = strategy_state.get(key)
        if value is None:
            continue
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (count,)
            or not bool(torch.isfinite(value).all())
        ):
            raise UserFacingError(
                f"Checkpoint strategy state {key} must be a finite ({count},) tensor."
            )
    return count


def cap_checkpoint_payload(payload: dict[str, Any], maximum: int) -> int:
    """Prune a restored checkpoint on CPU before allocating CUDA storage."""

    count = validate_checkpoint_dynamic_payload(payload)
    if count <= maximum:
        return 0
    logits = payload["splats"]["opacities"]
    keep = torch.topk(torch.sigmoid(logits), k=maximum, largest=True).indices.sort().values
    for name, value in payload["splats"].items():
        payload["splats"][name] = value[keep].contiguous()
    for optimizer_payload in payload["optimizers"].values():
        for state in optimizer_payload["state"].values():
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor) and value.ndim > 0:
                    state[key] = value[keep].contiguous()
    for key in ("grad2d", "count", "radii"):
        value = payload["strategy_state"].get(key)
        if isinstance(value, torch.Tensor):
            payload["strategy_state"][key] = value[keep].contiguous()
    return count - maximum


def parameters_from_checkpoint(payload: Mapping[str, Any], device: torch.device) -> torch.nn.ParameterDict:
    """Construct dynamic parameter shapes before optimizer-state restoration."""

    tensors = payload["splats"]
    expected = {"means", "scales", "quats", "opacities", "sh0", "shN"}
    if set(tensors) != expected or any(not isinstance(value, torch.Tensor) for value in tensors.values()):
        raise UserFacingError("Checkpoint Gaussian parameter keys do not match this trainer.")
    count = len(tensors["means"])
    expected_shapes = {
        "means": (count, 3),
        "scales": (count, 3),
        "quats": (count, 4),
        "opacities": (count,),
        "sh0": (count, 1, 3),
    }
    if count < 1 or any(tuple(tensors[name].shape) != shape for name, shape in expected_shapes.items()):
        raise UserFacingError("Checkpoint Gaussian parameter shapes are inconsistent.")
    shn = tensors["shN"]
    if shn.ndim != 3 or tuple(shn.shape[::2]) != (count, 3) or shn.shape[1] not in {0, 3, 8, 15}:
        raise UserFacingError("Checkpoint higher-order SH tensor has an invalid shape.")
    if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise UserFacingError("Checkpoint contains non-finite Gaussian parameters.")
    quaternion_norms = torch.linalg.vector_norm(tensors["quats"].to(torch.float64), dim=-1)
    if not bool(torch.isfinite(quaternion_norms).all()) or bool(
        (quaternion_norms <= 1e-12).any()
    ):
        raise UserFacingError("Checkpoint contains an invalid Gaussian quaternion.")
    activated_scales = torch.exp(tensors["scales"].to(torch.float32))
    if not bool(torch.isfinite(activated_scales).all()) or bool(
        (activated_scales <= 0).any()
    ):
        raise UserFacingError("Checkpoint contains invalid activated Gaussian scales.")
    splats = torch.nn.ParameterDict(
        {
            name: torch.nn.Parameter(tensor.to(device=device, dtype=torch.float32))
            for name, tensor in tensors.items()
        }
    )
    return splats


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move restored Adam moments to the parameter device across PyTorch versions."""

    for state in optimizer.state.values():
        for key, value in state.items():
            # Adam's scalar step counter normally remains on CPU when the
            # optimizer is not capturable; moving it adds a per-step sync.
            if key != "step" and isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def validate_dynamic_state(
    splats: torch.nn.ParameterDict,
    optimizers: Mapping[str, torch.optim.Optimizer],
    strategy_state: Mapping[str, Any],
) -> None:
    """Reject checkpoint moments/refinement arrays misaligned with splat rows."""

    count = len(splats["means"])
    for name, optimizer in optimizers.items():
        parameter = splats[name]
        for key, value in optimizer.state.get(parameter, {}).items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape != parameter.shape:
                raise UserFacingError(
                    f"Checkpoint optimizer state {name}.{key} has shape {tuple(value.shape)}; "
                    f"expected {tuple(parameter.shape)}."
                )
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                raise UserFacingError(f"Checkpoint optimizer state {name}.{key} is non-finite.")
    scene_scale = strategy_state.get("scene_scale")
    if isinstance(scene_scale, torch.Tensor):
        scene_scale = float(scene_scale)
    if not isinstance(scene_scale, (int, float)) or not math.isfinite(scene_scale) or scene_scale <= 0:
        raise UserFacingError("Checkpoint strategy scene_scale must be finite and positive.")
    for key in ("grad2d", "count", "radii"):
        value = strategy_state.get(key)
        if value is None:
            continue
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (count,)
            or value.device != splats["means"].device
            or not bool(torch.isfinite(value).all())
        ):
            raise UserFacingError(
                f"Checkpoint strategy state {key} must be a finite ({count},) tensor "
                "on the training device."
            )


def validate_ply_schema(splats: torch.nn.ParameterDict) -> tuple[int, int]:
    """Validate fixed 3DGS tensor ranks and return row/SH-rest counts."""

    expected = {"means", "scales", "quats", "opacities", "sh0", "shN"}
    if set(splats.keys()) != expected:
        raise UserFacingError("Gaussian parameter keys do not match the PLY schema.")
    count = len(splats["means"])
    shapes = {
        "means": (count, 3),
        "scales": (count, 3),
        "quats": (count, 4),
        "opacities": (count,),
        "sh0": (count, 1, 3),
    }
    if count < 1 or any(tuple(splats[name].shape) != shape for name, shape in shapes.items()):
        raise UserFacingError("Gaussian parameter shapes do not match the PLY schema.")
    shn = splats["shN"]
    if shn.ndim != 3 or shn.shape[0] != count or shn.shape[2] != 3:
        raise UserFacingError("Higher-order SH parameters do not match the PLY schema.")
    if shn.shape[1] not in {0, 3, 8, 15}:
        raise UserFacingError("PLY export supports SH degrees 0 through 3 only.")
    return count, int(shn.shape[1] * 3)


def ply_header(count: int, sh_rest_fields: int) -> bytes:
    """Build a standard raw-3DGS binary little-endian PLY header."""

    header = [
        "ply",
        "format binary_little_endian 1.0",
        "comment Generated by SplatFuse trainer/train.py",
        f"element vertex {count}",
        "property float x",
        "property float y",
        "property float z",
    ]
    header.extend(f"property float f_dc_{index}" for index in range(3))
    header.extend(f"property float f_rest_{index}" for index in range(sh_rest_fields))
    header.append("property float opacity")
    header.extend(f"property float scale_{index}" for index in range(3))
    header.extend(f"property float rot_{index}" for index in range(4))
    header.append("end_header")
    return ("\n".join(header) + "\n").encode("ascii")


def iter_ply_payload(
    splats: torch.nn.ParameterDict,
    chunk_size: int = 65_536,
) -> Iterable[bytes]:
    """Yield validated CPU chunks without constructing a full GPU/host matrix."""

    count, _ = validate_ply_schema(splats)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    with torch.no_grad():
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            means = splats["means"][start:end].detach().to(device="cpu", dtype=torch.float32)
            scales = splats["scales"][start:end].detach().to(device="cpu", dtype=torch.float32)
            quats = splats["quats"][start:end].detach().to(device="cpu", dtype=torch.float32)
            opacities = (
                splats["opacities"][start:end]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .reshape(-1, 1)
            )
            sh0 = (
                splats["sh0"][start:end]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .squeeze(1)
            )
            shn = (
                splats["shN"][start:end]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .permute(0, 2, 1)
                .reshape(end - start, -1)
            )
            raw_fields = (means, sh0, shn, opacities, scales, quats)
            if not all(bool(torch.isfinite(field).all()) for field in raw_fields):
                raise UserFacingError("Refusing to export NaN/Inf Gaussian parameters.")
            quaternion_norms = torch.linalg.vector_norm(quats.to(torch.float64), dim=-1)
            if not bool(torch.isfinite(quaternion_norms).all()) or bool(
                (quaternion_norms <= 1e-12).any()
            ):
                raise UserFacingError("Refusing to export a zero-length Gaussian quaternion.")
            activated_scales = torch.exp(scales)
            if not bool(torch.isfinite(activated_scales).all()) or bool(
                (activated_scales <= 0).any()
            ):
                raise UserFacingError("Refusing to export invalid activated Gaussian scales.")
            quats = (quats.to(torch.float64) / quaternion_norms[:, None]).to(torch.float32)
            data = torch.cat((means, sh0, shn, opacities, scales, quats), dim=1)
            array = data.numpy().astype("<f4", copy=False)
            yield array.tobytes(order="C")


def encode_ply(splats: torch.nn.ParameterDict) -> bytes:
    """Encode a PLY in memory; production exports use the streaming writer."""

    count, sh_rest_fields = validate_ply_schema(splats)
    return ply_header(count, sh_rest_fields) + b"".join(iter_ply_payload(splats))


def export_ply(path: Path, splats: torch.nn.ParameterDict) -> None:
    """Stream an atomic PLY without a full-model GPU or host concatenation."""

    count, sh_rest_fields = validate_ply_schema(splats)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temp.open("wb") as handle:
            handle.write(ply_header(count, sh_rest_fields))
            for chunk in iter_ply_payload(splats):
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    LOGGER.info("Exported %d Gaussians to %s", count, path)


def save_preview(path: Path, target: torch.Tensor, rendered: torch.Tensor) -> None:
    """Save ground truth and render side-by-side as an atomic PNG."""

    canvas = torch.cat([target, rendered.clamp(0.0, 1.0)], dim=2).squeeze(0)
    array = (canvas.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.tmp-{os.getpid()}.png")
    try:
        Image.fromarray(array, mode="RGB").save(temp, format="PNG")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@torch.no_grad()
def evaluate(
    runtime: GsplatRuntime,
    splats: torch.nn.ParameterDict,
    frames: Sequence[FrameRecord],
    store: ImageStore,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a bounded deterministic validation subset."""

    if not frames:
        return {}
    selected = tuple(frames[: config.eval_max_images]) if config.eval_max_images > 0 else tuple(frames)
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    for frame in selected:
        image_cpu, intrinsic = store.get(frame)
        target = image_cpu.to(device, non_blocking=True).unsqueeze(0)
        background = torch.ones((1, 3), device=device) if config.background == "white" else torch.zeros((1, 3), device=device)
        rendered, _, _ = render_frame(
            runtime,
            splats,
            frame,
            image_cpu,
            intrinsic,
            config,
            device,
            config.sh_degree,
            background,
        )
        rendered = rendered.clamp(0.0, 1.0)
        psnr_values.append(float(psnr(rendered, target)))
        ssim_values.append(
            float(ssim(rendered.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2)))
        )
    return {
        "val_psnr": float(np.mean(psnr_values)),
        "val_ssim": float(np.mean(ssim_values)),
        "val_images": float(len(selected)),
    }


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Append one flushed metrics record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(row), sort_keys=True) + "\n")
        handle.flush()


def truncate_metrics_for_resume(path: Path, completed_step: int) -> int:
    """Keep the valid monotonic metrics prefix through a checkpoint boundary."""

    if not path.is_file():
        return 0
    maximum_step = completed_step + 1  # metrics use one-based display steps
    kept: list[str] = []
    previous = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(raw)
            step = row["step"]
            if isinstance(step, bool) or not isinstance(step, int):
                break
            if step <= previous or step > maximum_step:
                break
        except (json.JSONDecodeError, KeyError, TypeError):
            break
        kept.append(json.dumps(row, sort_keys=True))
        previous = step
    atomic_write_bytes(path, (("\n".join(kept) + "\n") if kept else "").encode("utf-8"))
    return len(kept)


def archive_stale_summary(output: Path) -> Path | None:
    """Move an earlier completion summary aside before extending its run."""

    source = output / "summary.json"
    if not source.is_file():
        return None
    history = output / "history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = history / f"summary-{stamp}.json"
    suffix = 1
    while destination.exists():
        destination = history / f"summary-{stamp}-{suffix}.json"
        suffix += 1
    os.replace(source, destination)
    return destination


def paths_overlap(path_a: Path, path_b: Path) -> bool:
    """Return whether either resolved path contains the other."""

    left = path_a.resolve()
    right = path_b.resolve()
    return left == right or left in right.parents or right in left.parents


def prepare_output_directory(
    config: TrainConfig,
    scene_name: str,
    protected_paths: Sequence[Path] = (),
) -> Path:
    """Create output folders without risking source/data replacement."""

    associated_resume_root: Path | None = None
    if config.resume and not config.output_dir:
        checkpoint = Path(config.resume).expanduser().resolve()
        if checkpoint.parent.name != "checkpoints":
            raise UserFacingError(
                "A checkpoint outside a run's checkpoints/ folder requires --output-dir."
            )
        associated_resume_root = checkpoint.parent.parent
        output = associated_resume_root
    elif config.output_dir:
        output = Path(config.output_dir).expanduser().resolve()
    else:
        output = (Path(__file__).resolve().parent / "runs" / scene_name).resolve()
    if config.resume:
        checkpoint = Path(config.resume).expanduser().resolve()
        if checkpoint.parent.name == "checkpoints":
            associated_resume_root = checkpoint.parent.parent

    repo_root = Path(__file__).resolve().parents[1]
    trainer_root = Path(__file__).resolve().parent
    filesystem_root = Path(output.anchor).resolve()
    home = Path.home().resolve()
    exact_project_roots = [
        repo_root,
        trainer_root,
        repo_root / "viewer",
        repo_root / "renderer-cuda",
        repo_root / "phase0-capture",
        repo_root / "cloud",
        repo_root / "demo",
    ]
    if output in {filesystem_root, home} or output in repo_root.parents:
        raise UserFacingError(f"Refusing unsafe --output-dir: {output}")
    if any(output == path.resolve() for path in exact_project_roots):
        raise UserFacingError(f"--output-dir cannot replace a project source directory: {output}")
    if output == repo_root or repo_root in output.parents:
        allowed_runs_root = (trainer_root / "runs").resolve()
        if output == allowed_runs_root or allowed_runs_root not in output.parents:
            raise UserFacingError(
                "Repository-local runs must be placed below trainer/runs; refusing "
                f"source-tree output path: {output}"
            )
    for protected in protected_paths:
        if paths_overlap(output, protected):
            raise UserFacingError(
                f"--output-dir {output} overlaps protected input path {protected.resolve()}."
            )
    if output.exists() and not output.is_dir():
        raise UserFacingError(f"--output-dir is not a directory: {output}")
    lock = acquire_run_lock(output)
    try:
        if output.exists() and any(output.iterdir()):
            if config.resume and associated_resume_root == output:
                pass
            elif config.resume:
                raise UserFacingError(
                    "A branched resume --output-dir must be empty; refusing to mix run artifacts."
                )
            elif not config.overwrite:
                raise UserFacingError(
                    f"Output directory is not empty: {output}. Pass --resume or --overwrite."
                )
            else:
                stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                backup = output.with_name(f"{output.name}.backup-{stamp}")
                suffix = 1
                while backup.exists():
                    backup = output.with_name(f"{output.name}.backup-{stamp}-{suffix}")
                    suffix += 1
                os.replace(output, backup)
                LOGGER.warning("Preserved the previous output as %s", backup)
        for child in ("checkpoints", "previews", "exports"):
            (output / child).mkdir(parents=True, exist_ok=True)
        return output
    except Exception:
        release_run_lock(lock)
        raise


def environment_report() -> dict[str, Any]:
    """Return reproducibility metadata without importing/compiling gsplat kernels."""

    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_gib": round(properties.total_memory / 1024**3, 3),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_devices,
        "gsplat": package_version("gsplat"),
        "pycolmap": package_version("pycolmap"),
        "numpy": np.__version__,
        "pillow": package_version("Pillow"),
        "opencv_python_headless": package_version("opencv-python-headless") or package_version("opencv-python"),
        "scikit_learn": package_version("scikit-learn"),
        "nvcc": run_version_command(["nvcc", "--version"]),
        "nvidia_smi": run_version_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
        "command": sys.argv,
    }


def set_determinism(seed: int) -> None:
    """Seed Python, NumPy, CPU, and CUDA streams."""

    if not 0 <= seed <= 2**32 - 1:
        raise UserFacingError("--seed must be in NumPy's uint32 range [0, 2^32-1].")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config(config: TrainConfig) -> None:
    """Reject contradictory or unsafe values before allocating GPU memory."""

    positive_ints = {
        "iterations": config.iterations,
        "downscale": config.downscale,
        "max_initial_points": config.max_initial_points,
        "max_gaussians": config.max_gaussians,
        "sh_degree_interval": config.sh_degree_interval,
        "refine_every": config.refine_every,
        "reset_opacity_every": config.reset_opacity_every,
    }
    invalid = [name for name, value in positive_ints.items() if value <= 0]
    if invalid:
        raise UserFacingError(f"These options must be positive: {', '.join(invalid)}")
    if min(config.max_initial_points, config.max_gaussians) < 4:
        raise UserFacingError("--max-initial-points and --max-gaussians must each be at least 4.")
    if config.resume and config.overwrite:
        raise UserFacingError("--resume and --overwrite are mutually exclusive.")
    if not 0 <= config.seed <= 2**32 - 1:
        raise UserFacingError("--seed must be in NumPy's uint32 range [0, 2^32-1].")
    if config.sh_degree not in range(4):
        raise UserFacingError("--sh-degree must be between 0 and 3.")
    if not math.isfinite(config.ssim_weight) or not 0.0 <= config.ssim_weight <= 1.0:
        raise UserFacingError("--ssim-weight must be in [0,1].")
    if not math.isfinite(config.init_opacity) or not 0.0 < config.init_opacity < 1.0:
        raise UserFacingError("--init-opacity must be strictly between 0 and 1.")
    if not math.isfinite(config.prune_opacity) or not 0.0 < config.prune_opacity < 0.5:
        raise UserFacingError("--prune-opacity must be strictly between 0 and 0.5.")
    if (
        not math.isfinite(config.near_plane)
        or not math.isfinite(config.far_plane)
        or config.near_plane <= 0
        or config.far_plane <= config.near_plane
    ):
        raise UserFacingError("Require 0 < --near-plane < --far-plane.")
    positive_floats = {
        "init_scale": config.init_scale,
        "means_lr": config.means_lr,
        "means_lr_final_factor": config.means_lr_final_factor,
        "scales_lr": config.scales_lr,
        "quats_lr": config.quats_lr,
        "opacities_lr": config.opacities_lr,
        "sh0_lr": config.sh0_lr,
        "shn_lr": config.shn_lr,
        "grow_gradient": config.grow_gradient,
        "grow_scale_3d": config.grow_scale_3d,
        "prune_scale_3d": config.prune_scale_3d,
    }
    invalid_floats = [
        name for name, value in positive_floats.items()
        if not math.isfinite(value) or value <= 0
    ]
    if invalid_floats:
        raise UserFacingError(
            f"These options must be positive: {', '.join(invalid_floats)}"
        )
    if config.means_lr_final_factor > 1.0:
        raise UserFacingError("--means-lr-final-factor must be in (0,1].")
    nonnegative = {
        "max_resolution": config.max_resolution,
        "cache_images": config.cache_images,
        "val_every": config.val_every,
        "eval_max_images": config.eval_max_images,
        "gradient_clip": config.gradient_clip,
        "opacity_regularization": config.opacity_regularization,
        "scale_regularization": config.scale_regularization,
        "pause_refine_after_reset": config.pause_refine_after_reset,
        "checkpoint_every": config.checkpoint_every,
        "eval_every": config.eval_every,
        "preview_every": config.preview_every,
        "export_every": config.export_every,
    }
    invalid_nonnegative = [
        name for name, value in nonnegative.items()
        if isinstance(value, float) and not math.isfinite(value) or value < 0
    ]
    if invalid_nonnegative:
        raise UserFacingError(
            f"These options cannot be negative: {', '.join(invalid_nonnegative)}"
        )
    if config.refine_start < 0 or (
        config.refine_stop != 0 and config.refine_stop < config.refine_start
    ):
        raise UserFacingError(
            "Require non-negative --refine-start and either --refine-stop=0 "
            "(disabled) or --refine-stop >= --refine-start."
        )
    if math.isnan(config.point_error_max) or config.point_error_max < 0:
        raise UserFacingError("--point-error-max must be non-negative or +inf.")


def inspect_scene(config: TrainConfig) -> dict[str, Any]:
    """Validate every image and return a CPU-safe dataset readiness report."""

    scene = load_scene(config)
    training, validation = split_frames(scene.frames, config.val_every)
    store = ImageStore(scene, config)
    sample, intrinsic = preflight_images(store, scene.frames)
    models = sorted({camera.model for camera in scene.cameras.values()})
    return {
        "data_root": scene.data_root,
        "model_path": scene.model_path,
        "images_path": scene.images_path,
        "fingerprint": scene.fingerprint,
        "cameras": len(scene.cameras),
        "registered_images": len(scene.frames),
        "train_images": len(training),
        "validation_images": len(validation),
        "sparse_points": len(scene.points.xyz),
        "scene_scale": scene.scene_scale,
        "camera_models": models,
        "sample_image": training[0].name,
        "sample_shape_hwc": list(sample.shape),
        "sample_intrinsic": intrinsic,
    }


def doctor(data: str | None, config: TrainConfig | None = None) -> dict[str, Any]:
    """Diagnose the complete CUDA/backend stack and optionally inspect data."""

    report = environment_report()
    report["expected_gsplat"] = EXPECTED_GSPLAT_VERSION
    checks: dict[str, bool] = {
        "python_supported": sys.version_info[:2] in {(3, 10), (3, 11)},
        "torch_has_cuda_build": torch.version.cuda is not None,
        "cuda_device_visible": torch.cuda.is_available(),
        "nvcc_available": report["nvcc"] is not None,
        "nvidia_smi_available": report["nvidia_smi"] is not None,
        "gsplat_release_compatible": (
            package_base_version(report["gsplat"]) == EXPECTED_GSPLAT_VERSION
        ),
    }
    import_errors: dict[str, str] = {}
    for label, module in {
        "opencv": "cv2",
        "scikit_learn": "sklearn",
        "pycolmap": "pycolmap",
        "tqdm": "tqdm",
    }.items():
        try:
            importlib.import_module(module)
            checks[f"{label}_importable"] = True
        except Exception as error:
            checks[f"{label}_importable"] = False
            import_errors[label] = str(error)
    checks["gsplat_importable"] = False
    if checks["cuda_device_visible"] and checks["gsplat_release_compatible"]:
        try:
            load_gsplat_runtime(False)
            backend = importlib.import_module("gsplat.cuda._backend")
            getattr(backend, "_C")
            checks["gsplat_importable"] = True
        except Exception as error:
            import_errors["gsplat"] = str(error)
    report["checks"] = checks
    if import_errors:
        report["import_errors"] = import_errors
    runtime_required = {
        key: value
        for key, value in checks.items()
        if key not in {"nvcc_available", "nvidia_smi_available"}
    }
    report["cuda_development_tools_ready"] = bool(
        checks["nvcc_available"] and checks["nvidia_smi_available"]
    )
    report["ready_for_cuda_training"] = all(runtime_required.values())
    if data and config is not None:
        try:
            validate_config(config)
            report["dataset"] = inspect_scene(config)
        except Exception as error:  # doctor should report every independent check
            report["dataset_error"] = str(error)
            report["ready_for_cuda_training"] = False
    return report


def run_self_test() -> dict[str, Any]:
    """Run fast CPU-only checks over parsers, math, loss, and PLY export."""

    rotation = qvec_to_rotation([1.0, 0.0, 0.0, 0.0])
    if not np.allclose(rotation, np.eye(3), atol=1e-7):
        raise AssertionError("Quaternion conversion failed.")
    image = torch.linspace(0.0, 1.0, 3 * 8 * 8).reshape(1, 3, 8, 8)
    if not torch.allclose(ssim(image, image), torch.tensor(1.0), atol=1e-5):
        raise AssertionError("Identity SSIM failed.")
    with tempfile.TemporaryDirectory(prefix="splatfuse-selftest-") as directory:
        root = Path(directory)
        sparse = root / "sparse"
        images = root / "images"
        sparse.mkdir()
        images.mkdir()
        (sparse / "cameras.txt").write_text(
            "# test\n1 PINHOLE 8 8 8 8 3.5 3.5\n", encoding="utf-8"
        )
        (sparse / "images.txt").write_text(
            "# test\n1 1 0 0 0 0 0 0 1 frame.png\n\n", encoding="utf-8"
        )
        point_lines = [
            f"{index + 1} {x} {y} 2 255 128 0 0.1 1 0"
            for index, (x, y) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1)))
        ]
        (sparse / "points3D.txt").write_text("\n".join(point_lines) + "\n", encoding="utf-8")
        Image.new("RGB", (8, 8), (128, 64, 32)).save(images / "frame.png")
        config = TrainConfig(data=str(root), iterations=1, sh_degree=0, max_resolution=8)
        scene = load_scene(config)
        store = ImageStore(scene, config)
        prepared, intrinsic = store.get(scene.frames[0])
        splats = initialize_splats(scene, config, torch.device("cpu"))
        encoded = encode_ply(splats)
        if b"element vertex 4\n" not in encoded or not encoded.startswith(b"ply\n"):
            raise AssertionError("PLY export failed.")
        if prepared.shape != (8, 8, 3) or intrinsic.shape != (3, 3):
            raise AssertionError("Image preparation failed.")
    return {
        "status": "passed",
        "checks": [
            "COLMAP quaternion",
            "SSIM identity",
            "model discovery/text parsing",
            "image preparation/intrinsics",
            "nearest-neighbor initialization",
            "binary PLY encoding",
        ],
    }


def run_training(config: TrainConfig) -> Path:
    """Execute the complete CUDA training lifecycle and return the final PLY path."""

    validate_config(config)
    if config.device != "cuda" and not config.device.startswith("cuda:"):
        raise UserFacingError(
            "Production training supports CUDA only. Use --self-test/--inspect-data on CPU."
        )
    if not torch.cuda.is_available():
        raise UserFacingError(
            "No CUDA-capable NVIDIA GPU is visible to PyTorch. This AMD computer can run "
            "--doctor, --inspect-data, and --self-test; run training on the NVIDIA PC."
        )
    try:
        device = torch.device(config.device)
    except (RuntimeError, ValueError) as error:
        raise UserFacingError(f"Invalid --device {config.device!r}: {error}") from error
    selected_index = device.index if device.index is not None else torch.cuda.current_device()
    if selected_index < 0 or selected_index >= torch.cuda.device_count():
        raise UserFacingError(
            f"CUDA device index {selected_index} is unavailable; found "
            f"{torch.cuda.device_count()} device(s)."
        )
    device = torch.device("cuda", selected_index)
    torch.cuda.set_device(device)
    resume_path = Path(config.resume).expanduser().resolve() if config.resume else None
    checkpoint: dict[str, Any] | None = None
    resume_configuration: dict[str, Any] | None = None
    saved_config: Mapping[str, Any] = {}
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, device)
        saved_config = dict(checkpoint["config"])
        resume_configuration = restore_checkpoint_config(config, saved_config)
        validate_config(config)

    runtime = load_gsplat_runtime(config.allow_version_mismatch)
    set_determinism(config.seed)
    scene = load_scene(config)
    training_frames, validation_frames = split_frames(scene.frames, config.val_every)
    store = ImageStore(scene, config)
    # Full preflight keeps corrupt later frames/secondary cameras from failing
    # after expensive CUDA allocations or many completed optimization steps.
    preflight_images(store, scene.frames)

    checkpoint_step = -1
    if checkpoint is not None:
        checkpoint_step = int(checkpoint["completed_step"])
        if checkpoint["scene_fingerprint"] != scene.fingerprint and not config.allow_data_mismatch:
            raise UserFacingError(
                "Checkpoint scene fingerprint does not match --data. Pass --allow-data-mismatch "
                "only if this substitution is intentional and compatible."
            )
        expected_train = [frame.image_id for frame in training_frames]
        expected_validation = [frame.image_id for frame in validation_frames]
        split_matches = (
            checkpoint["train_image_ids"] == expected_train
            and checkpoint["val_image_ids"] == expected_validation
        )
        if not split_matches and not config.allow_data_mismatch:
            raise UserFacingError("Train/validation split differs from the checkpoint.")
        if not split_matches:
            LOGGER.warning("Resuming with an explicitly allowed train/validation split mismatch.")
        if checkpoint_step + 1 >= config.iterations:
            raise UserFacingError(
                f"Checkpoint already completed {checkpoint_step + 1} steps; "
                "--iterations must be larger."
            )

    scene_name = sanitize_scene_name(scene.data_root.name)
    output = prepare_output_directory(
        config,
        scene_name,
        protected_paths=(scene.data_root, scene.model_path, scene.images_path),
    )
    run_lock = output.parent / f".{output.name}.splatfuse-run.lock"
    config.output_dir = str(output)
    metrics_path = output / "metrics.jsonl"
    sampler = random.Random(config.seed)

    if checkpoint is not None:
        # Validate and lower the cap while all checkpoint storage is still on
        # CPU. This makes --max-gaussians useful for OOM recovery instead of
        # allocating the oversized model before pruning it.
        cpu_probe = parameters_from_checkpoint(checkpoint, torch.device("cpu"))
        del cpu_probe
        removed = cap_checkpoint_payload(checkpoint, config.max_gaussians)
        if removed:
            LOGGER.warning(
                "Pruned %d checkpoint Gaussians on CPU before CUDA allocation.", removed
            )
        splats = parameters_from_checkpoint(checkpoint, device)
    else:
        splats = initialize_splats(scene, config, device)
    expected_sh_rest = (config.sh_degree + 1) ** 2 - 1
    if splats["shN"].shape[1] != expected_sh_rest:
        raise UserFacingError(
            f"Gaussian SH tensor has {splats['shN'].shape[1]} higher-order coefficients; "
            f"--sh-degree={config.sh_degree} requires {expected_sh_rest}."
        )

    optimizers = make_optimizers(splats, config, scene.scene_scale)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"],
        gamma=config.means_lr_final_factor ** (1.0 / config.iterations),
    )
    strategy = make_strategy(runtime, config)
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene.scene_scale)
    start_step = 0
    if checkpoint is not None:
        try:
            for name, optimizer in optimizers.items():
                optimizer.load_state_dict(checkpoint["optimizers"][name])
                move_optimizer_state_to_device(optimizer, device)
            scheduler.load_state_dict(checkpoint["scheduler"])
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise UserFacingError(f"Checkpoint optimizer/scheduler state is invalid: {error}") from error
        strategy_state = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in checkpoint["strategy_state"].items()
        }
        validate_dynamic_state(splats, optimizers, strategy_state)
        legacy_device_index = 0
        try:
            saved_device = torch.device(str(saved_config.get("device", "cuda")))
            legacy_device_index = saved_device.index or 0
        except (RuntimeError, ValueError):
            pass
        restore_rng_state(
            checkpoint["rng"],
            sampler,
            device,
            legacy_device_index=legacy_device_index,
        )
        start_step = checkpoint_step + 1
        # The CPU payload otherwise retains a second complete copy of model and
        # Adam state throughout the resumed run.
        del checkpoint
        checkpoint = None
        torch.cuda.empty_cache()
        strategy.check_sanity(splats, optimizers)
        LOGGER.info("Resumed %s after %d completed steps.", resume_path, start_step)

    # Mutate prior metrics/summary only after the checkpoint has passed every
    # CPU and CUDA restoration check. A failed resume attempt therefore cannot
    # discard evidence from a newer completed boundary.
    if resume_path is not None:
        kept_rows = truncate_metrics_for_resume(metrics_path, checkpoint_step)
        archived_summary = archive_stale_summary(output)
        history_row = {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "checkpoint": resume_path,
            "completed_step": checkpoint_step + 1,
            "metrics_rows_kept": kept_rows,
            "archived_summary": archived_summary,
            "configuration": resume_configuration,
        }
        append_jsonl(output / "resume-history.jsonl", history_row)
    config_document = {
        "project": "SplatFuse",
        "format_version": FORMAT_VERSION,
        "effective": dataclasses.asdict(config),
        "resume": (
            {
                "source": resume_path,
                "checkpoint_config": saved_config,
                **(resume_configuration or {}),
            }
            if resume_path is not None
            else None
        ),
    }
    atomic_write_json(output / "config.json", config_document)
    atomic_write_json(output / "environment.json", environment_report())

    LOGGER.info(
        "Training %s: %d images (%d validation), %d initial Gaussians, scale %.6g, %s.",
        scene_name,
        len(training_frames),
        len(validation_frames),
        len(splats["means"]),
        scene.scene_scale,
        torch.cuda.get_device_name(device),
    )
    last_checkpoint: Path | None = resume_path

    try:
        from tqdm.auto import tqdm  # type: ignore[import-not-found]
    except ImportError as error:
        raise UserFacingError("tqdm is missing; install trainer/requirements.txt.") from error

    def save_state(completed_step: int, label: str | None = None) -> Path:
        nonlocal last_checkpoint
        filename = label or f"step-{completed_step + 1:06d}.pt"
        destination = output / "checkpoints" / filename
        payload = checkpoint_payload(
            completed_step,
            splats,
            optimizers,
            scheduler,
            strategy_state,
            config,
            scene,
            training_frames,
            validation_frames,
            sampler,
        )
        atomic_torch_save(destination, payload)
        last_checkpoint = destination
        return destination

    last_completed_step = start_step - 1
    stop_requested = False
    stop_signal: str | None = None
    safe_interrupt_saved = False
    previous_handlers: dict[signal.Signals, Any] = {}

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested, stop_signal
        name = signal.Signals(signum).name
        if stop_requested:
            LOGGER.error("Received a second termination signal; aborting without an in-flight save.")
            raise KeyboardInterrupt
        stop_requested = True
        stop_signal = name
        LOGGER.warning("Received %s; finishing the current step before checkpointing.", name)

    def honor_finalization_stop(stage: str) -> None:
        """Exit after finalization work once a completed-state checkpoint exists."""

        nonlocal safe_interrupt_saved
        if stop_requested:
            safe_interrupt_saved = True
            LOGGER.warning(
                "%s requested shutdown during %s; completed-state checkpoint is %s",
                stop_signal,
                stage,
                last_checkpoint,
            )
            raise KeyboardInterrupt

    for candidate in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[candidate] = signal.getsignal(candidate)
            signal.signal(candidate, request_stop)
        except (OSError, ValueError):
            previous_handlers.pop(candidate, None)

    progress = tqdm(range(start_step, config.iterations), initial=start_step, total=config.iterations)
    try:
        for step in progress:
            frame = training_frames[sampler.randrange(len(training_frames))]
            image_cpu, intrinsic = store.get(frame)
            target = image_cpu.to(device, non_blocking=True).unsqueeze(0)
            active_sh = min(config.sh_degree, step // config.sh_degree_interval)
            background = choose_background(config, device)
            rendered, _, info = render_frame(
                runtime,
                splats,
                frame,
                image_cpu,
                intrinsic,
                config,
                device,
                active_sh,
                background,
            )
            strategy.step_pre_backward(
                params=splats,
                optimizers=optimizers,
                state=strategy_state,
                step=step,
                info=info,
            )
            loss, l1, ssim_score = image_loss(rendered, target, config.ssim_weight)
            if config.opacity_regularization > 0:
                loss = loss + config.opacity_regularization * torch.sigmoid(splats["opacities"]).mean()
            if config.scale_regularization > 0:
                loss = loss + config.scale_regularization * torch.exp(splats["scales"]).mean()
            if not bool(torch.isfinite(loss)):
                raise UserFacingError(f"Non-finite loss at step {step + 1}.")
            loss.backward()
            if not all_gradients_finite(splats):
                raise UserFacingError(f"Non-finite gradient at step {step + 1}.")
            if config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(splats.parameters(), config.gradient_clip)
            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            # A DefaultStrategy refinement can add at most one net splat per
            # existing splat. Suppress only growth beyond half the configured
            # ceiling, while still allowing its opacity/size pruning pass.
            growth_due = (
                step < config.refine_stop
                and step > config.refine_start
                and step % config.refine_every == 0
                and step % config.reset_opacity_every >= config.pause_refine_after_reset
            )
            suppress_growth = growth_due and len(splats["means"]) > config.max_gaussians // 2
            original_grow_threshold = strategy.grow_grad2d
            if suppress_growth:
                strategy.grow_grad2d = math.inf
                LOGGER.warning(
                    "Suppressed density growth at %d Gaussians to honor --max-gaussians=%d.",
                    len(splats["means"]),
                    config.max_gaussians,
                )
            try:
                strategy.step_post_backward(
                    params=splats,
                    optimizers=optimizers,
                    state=strategy_state,
                    step=step,
                    info=info,
                    packed=config.packed,
                )
            finally:
                strategy.grow_grad2d = original_grow_threshold
            # gsplat 1.5.3 uses ``== 0 & step > 0`` and never performs this reset.
            if (
                package_base_version(runtime.version) == EXPECTED_GSPLAT_VERSION
                and step > 0
                and step < config.refine_stop
                and step % config.reset_opacity_every == 0
            ):
                runtime.reset_opacity(
                    params=splats,
                    optimizers=optimizers,
                    state=strategy_state,
                    value=config.prune_opacity * 2.0,
                )
                LOGGER.info("Applied gsplat 1.5.3 opacity-reset workaround at step %d.", step + 1)
            enforce_gaussian_limit(
                runtime,
                splats,
                optimizers,
                strategy_state,
                config.max_gaussians,
            )
            if not all(bool(torch.isfinite(value).all()) for value in splats.values()):
                raise UserFacingError(f"Non-finite Gaussian parameter after step {step + 1}.")

            row: dict[str, Any] = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "l1": float(l1.detach()),
                "ssim": float(ssim_score.detach()),
                "gaussians": len(splats["means"]),
                "active_sh_degree": active_sh,
                "means_lr": optimizers["means"].param_groups[0]["lr"],
                "cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                "image": frame.name,
            }
            if config.eval_every > 0 and (step + 1) % config.eval_every == 0:
                row.update(evaluate(runtime, splats, validation_frames, store, config, device))
            append_jsonl(metrics_path, row)
            last_completed_step = step
            progress.set_postfix(
                loss=f"{row['loss']:.4f}",
                ssim=f"{row['ssim']:.3f}",
                gs=row["gaussians"],
            )
            if stop_requested:
                emergency = save_state(step, "interrupted.pt")
                safe_interrupt_saved = True
                LOGGER.warning(
                    "%s requested shutdown; safe-boundary checkpoint saved to %s",
                    stop_signal,
                    emergency,
                )
                raise KeyboardInterrupt
            if config.preview_every > 0 and (step + 1) % config.preview_every == 0:
                save_preview(
                    output / "previews" / f"step-{step + 1:06d}.png",
                    target,
                    rendered,
                )
            if config.export_every > 0 and (step + 1) % config.export_every == 0:
                export_ply(output / "exports" / f"step-{step + 1:06d}.ply", splats)
            if config.checkpoint_every > 0 and (step + 1) % config.checkpoint_every == 0:
                save_state(step)
            if stop_requested:
                emergency = save_state(step, "interrupted.pt")
                safe_interrupt_saved = True
                LOGGER.warning(
                    "%s requested shutdown; safe-boundary checkpoint saved to %s",
                    stop_signal,
                    emergency,
                )
                raise KeyboardInterrupt
        if stop_requested and last_completed_step >= start_step:
            emergency = save_state(last_completed_step, "interrupted.pt")
            safe_interrupt_saved = True
            LOGGER.warning(
                "%s requested shutdown; safe-boundary checkpoint saved to %s",
                stop_signal,
                emergency,
            )
            raise KeyboardInterrupt
        if last_completed_step < start_step:
            raise UserFacingError(
                f"Checkpoint already completed {start_step} steps, which is not below "
                "--iterations."
            )
        save_state(last_completed_step, "final.pt")
        honor_finalization_stop("final checkpoint serialization")
        final_ply = output / "final.ply"
        export_ply(final_ply, splats)
        honor_finalization_stop("final PLY export")
        if config.viewer_ply:
            viewer_path = Path(config.viewer_ply).expanduser().resolve()
            protected_publish_paths = [
                scene.data_root,
                scene.model_path,
                scene.images_path,
                output,
            ]
            if resume_path is not None:
                protected_publish_paths.append(resume_path)
            if viewer_path.suffix.lower() != ".ply" or any(
                paths_overlap(viewer_path, protected)
                for protected in protected_publish_paths
            ):
                raise UserFacingError(
                    "--viewer-ply must be a .ply outside COLMAP inputs, checkpoints, "
                    "and the active run directory."
                )
            atomic_copy_file(final_ply, viewer_path)
            LOGGER.info("Published viewer copy to %s", viewer_path)
            honor_finalization_stop("viewer publication")
        summary = {
            "status": "completed",
            "completed_steps": last_completed_step + 1,
            "gaussians": len(splats["means"]),
            "final_ply": final_ply,
            "final_checkpoint": last_checkpoint,
            "scene_fingerprint": scene.fingerprint,
        }
        atomic_write_json(output / "summary.json", summary)
        honor_finalization_stop("summary publication")
        release_run_lock(run_lock)
        honor_finalization_stop("run-lock release")
    except KeyboardInterrupt:
        if not safe_interrupt_saved:
            LOGGER.error(
                "Stopped during an in-flight operation; no partial checkpoint was written. "
                "Last resumable checkpoint: %s",
                last_checkpoint or "none",
            )
        raise
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        raise UserFacingError(
            "CUDA out of memory; no potentially partial OOM state was saved. Resume "
            f"{last_checkpoint or 'the latest earlier scheduled checkpoint'}, then use a "
            "larger --downscale, smaller --max-resolution/--max-gaussians, or more VRAM."
        ) from error
    finally:
        progress.close()
        for candidate, previous in previous_handlers.items():
            try:
                signal.signal(candidate, previous)
            except (OSError, ValueError):
                pass

    return final_ply


def build_parser() -> argparse.ArgumentParser:
    """Build the documented command-line interface."""

    parser = argparse.ArgumentParser(
        description="Train a 3D Gaussian Splatting scene from COLMAP using CUDA/gsplat.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", help="Scene root containing images/ and sparse/.")
    parser.add_argument("--output-dir", help="Run artifact directory.")
    parser.add_argument("--images-dir", help="Override the registered image directory.")
    parser.add_argument("--colmap-model", help="Override the sparse model directory.")
    parser.add_argument("--viewer-ply", help="Also atomically publish final PLY here.")
    parser.add_argument("--resume", help="Resume a SplatFuse .pt checkpoint.")
    parser.add_argument("--overwrite", action="store_true", help="Back up and replace an existing run directory.")
    parser.add_argument("--allow-data-mismatch", action="store_true", help="Allow resume against a different fingerprint/split.")
    parser.add_argument("--allow-version-mismatch", action="store_true", help="Allow an unpinned gsplat API at your own risk.")
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda or cuda:1.")
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--downscale", type=int, default=1, help="Integer image reduction factor.")
    parser.add_argument("--max-resolution", type=int, default=1_600, help="Cap the longest image side; 0 disables cap.")
    parser.add_argument("--cache-images", type=int, default=8, help="CPU LRU image count; 0 disables cache.")
    parser.add_argument("--val-every", type=int, default=8, help="Hold out every Nth sorted frame; 0 disables.")
    parser.add_argument("--eval-max-images", type=int, default=8, help="Validation images per evaluation; 0 means all.")
    parser.add_argument("--max-initial-points", type=int, default=200_000)
    parser.add_argument("--max-gaussians", type=int, default=2_000_000)
    parser.add_argument("--point-error-max", type=float, default=math.inf)
    parser.add_argument("--init-opacity", type=float, default=0.1)
    parser.add_argument("--init-scale", type=float, default=1.0)
    parser.add_argument("--sh-degree", type=int, default=3, choices=range(4))
    parser.add_argument("--sh-degree-interval", type=int, default=1_000)
    parser.add_argument("--means-lr", type=float, default=1.6e-4)
    parser.add_argument("--means-lr-final-factor", type=float, default=0.01)
    parser.add_argument("--scales-lr", type=float, default=5e-3)
    parser.add_argument("--quats-lr", type=float, default=1e-3)
    parser.add_argument("--opacities-lr", type=float, default=5e-2)
    parser.add_argument("--sh0-lr", type=float, default=2.5e-3)
    parser.add_argument("--shn-lr", type=float, default=1.25e-4)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--opacity-regularization", type=float, default=0.0)
    parser.add_argument("--scale-regularization", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=0.0)
    parser.add_argument("--background", choices=("random", "white", "black"), default="random")
    parser.add_argument("--rasterize-mode", choices=("classic", "antialiased"), default="classic")
    parser.add_argument("--packed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=1e10)
    parser.add_argument("--refine-start", type=int, default=500)
    parser.add_argument("--refine-stop", type=int, default=15_000)
    parser.add_argument("--refine-every", type=int, default=100)
    parser.add_argument("--reset-opacity-every", type=int, default=3_000)
    parser.add_argument("--pause-refine-after-reset", type=int, default=0)
    parser.add_argument("--prune-opacity", type=float, default=0.005)
    parser.add_argument("--grow-gradient", type=float, default=0.0002)
    parser.add_argument("--grow-scale-3d", type=float, default=0.01)
    parser.add_argument("--prune-scale-3d", type=float, default=0.1)
    parser.add_argument("--absgrad", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--preview-every", type=int, default=500)
    parser.add_argument("--export-every", type=int, default=5_000)
    parser.add_argument("--doctor", action="store_true", help="Report readiness without training.")
    parser.add_argument("--inspect-data", action="store_true", help="Validate/load the dataset without CUDA.")
    parser.add_argument("--self-test", action="store_true", help="Run CPU-safe internal checks.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Select only TrainConfig fields from argparse's diagnostic flags."""

    names = {field.name for field in dataclasses.fields(TrainConfig)}
    values = {name: getattr(args, name) for name in names if hasattr(args, name)}
    return TrainConfig(**values)


def explicit_argument_destinations(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
) -> set[str]:
    """Return argparse destinations explicitly present on the command line."""

    destinations: set[str] = set()
    actions = parser._option_string_actions  # argparse exposes no public equivalent
    for token in argv:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        action = actions.get(option)
        if action is not None:
            destinations.add(action.dest)
    return destinations


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point with concise expected-error reporting."""

    parser = build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)
    configure_logging(args.verbose)
    if args.self_test:
        print(json.dumps(json_ready(run_self_test()), indent=2))
        return 0
    if not args.data and not args.doctor:
        parser.error("--data is required for training or --inspect-data")
    requested_data = args.data
    placeholder_data = requested_data or "."
    args.data = placeholder_data
    config = config_from_args(args)
    config._explicit_fields = explicit_argument_destinations(parser, raw_argv)
    try:
        if args.doctor:
            print(json.dumps(json_ready(doctor(requested_data, config)), indent=2))
            return 0
        validate_config(config)
        if args.inspect_data:
            print(json.dumps(json_ready(inspect_scene(config)), indent=2))
            return 0
        final = run_training(config)
        print(f"Training complete: {final}")
        return 0
    except KeyboardInterrupt:
        LOGGER.error("Interrupted.")
        return 130
    except UserFacingError as error:
        LOGGER.error("%s", error)
        return 2
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        LOGGER.error(
            "CUDA out of memory before a safe checkpoint boundary; no in-flight state was saved."
        )
        return 2
    except Exception:
        LOGGER.exception("Unexpected trainer failure")
        return 1
    finally:
        release_all_run_locks()


if __name__ == "__main__":
    raise SystemExit(main())
