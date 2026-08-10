"""CPU-safe regression tests for :mod:`trainer.train`."""

from __future__ import annotations

import json
import math
import random
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from trainer import train


def write_text_model(root: Path, image_count: int = 4, point_count: int = 5) -> None:
    """Create a minimal valid PINHOLE COLMAP text scene for tests."""

    sparse = root / "sparse"
    images = root / "images"
    sparse.mkdir(parents=True)
    images.mkdir(parents=True)
    (sparse / "cameras.txt").write_text(
        "# Camera list\n1 PINHOLE 16 12 14 15 8 6\n",
        encoding="utf-8",
    )
    image_lines = ["# Image list"]
    for index in range(image_count):
        name = f"frame {index:02d}.png"
        image_lines.extend(
            [
                f"{index + 1} 1 0 0 0 {index * 0.1} 0 0 1 {name}",
                "",
            ]
        )
        Image.new("RGB", (16, 12), (20 * index, 80, 160)).save(images / name)
    (sparse / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    points = []
    for index in range(point_count):
        x = float(index % 3)
        y = float(index // 3)
        points.append(f"{index + 1} {x} {y} 2 255 128 32 0.1 1 0 2 0")
    (sparse / "points3D.txt").write_text("\n".join(points) + "\n", encoding="utf-8")


def make_minimal_splats(
    *,
    scales: torch.Tensor | None = None,
    quats: torch.Tensor | None = None,
) -> torch.nn.ParameterDict:
    """Build one degree-zero Gaussian for focused export validation tests."""

    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.zeros(1, 3)),
            "scales": torch.nn.Parameter(
                torch.zeros(1, 3) if scales is None else scales.to(torch.float32)
            ),
            "quats": torch.nn.Parameter(
                torch.tensor([[1.0, 0.0, 0.0, 0.0]])
                if quats is None
                else quats.to(torch.float32)
            ),
            "opacities": torch.nn.Parameter(torch.zeros(1)),
            "sh0": torch.nn.Parameter(torch.zeros(1, 1, 3)),
            "shN": torch.nn.Parameter(torch.zeros(1, 0, 3)),
        }
    )


class GeometryTests(unittest.TestCase):
    def test_identity_quaternion(self) -> None:
        np.testing.assert_allclose(
            train.qvec_to_rotation([2.0, 0.0, 0.0, 0.0]),
            np.eye(3),
            atol=1e-7,
        )

    def test_quarter_turn_quaternion(self) -> None:
        half = math.sqrt(0.5)
        rotation = train.qvec_to_rotation([half, 0.0, 0.0, half])
        np.testing.assert_allclose(
            rotation @ np.asarray([1.0, 0.0, 0.0]),
            np.asarray([0.0, 1.0, 0.0]),
            atol=1e-6,
        )

    def test_scale_intrinsics_follows_colmap_corner_coordinates(self) -> None:
        intrinsic = np.asarray([[100.0, 0, 50.0], [0, 80.0, 40.0], [0, 0, 1]])
        scaled = train.scale_intrinsics(intrinsic, 0.5, 0.25)
        np.testing.assert_allclose(
            scaled,
            np.asarray([[50.0, 0, 25.0], [0, 20.0, 10.0], [0, 0, 1]]),
        )

    def test_camera_models(self) -> None:
        pinhole = train.CameraRecord(1, "PINHOLE", 10, 8, (9, 10, 4.5, 3.5))
        k, distortion, kind = train.camera_intrinsics(pinhole)
        self.assertEqual(kind, "none")
        self.assertEqual(k[0, 0], 9)
        self.assertFalse(distortion.any())
        radial = train.CameraRecord(2, "SIMPLE_RADIAL", 10, 8, (9, 4.5, 3.5, 0.1))
        _, distortion, kind = train.camera_intrinsics(radial)
        self.assertEqual(kind, "standard")
        self.assertAlmostEqual(distortion[0], 0.1)

    def test_unsupported_camera_is_explicit(self) -> None:
        camera = train.CameraRecord(1, "FOV", 10, 8, (9, 4.5, 3.5, 0.2))
        with self.assertRaisesRegex(train.UserFacingError, "not supported"):
            train.camera_intrinsics(camera)


class DiscoveryAndParsingTests(unittest.TestCase):
    def test_direct_sparse_export_beats_numbered_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_text_model(root)
            numbered = root / "sparse" / "0"
            numbered.mkdir()
            for stem in train.MODEL_STEMS:
                (numbered / f"{stem}.txt").write_bytes(b"x" * 1000)
            self.assertEqual(train.discover_colmap_model(root), root / "sparse")

    def test_largest_numbered_model_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sparse = root / "sparse"
            for name, size in (("0", 10), ("1", 20)):
                model = sparse / name
                model.mkdir(parents=True)
                (model / "cameras.bin").write_bytes(b"c")
                (model / "images.bin").write_bytes(b"i")
                (model / "points3D.bin").write_bytes(b"p" * size)
            self.assertEqual(train.discover_colmap_model(root), sparse / "1")

    def test_incomplete_binary_files_do_not_override_complete_text_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_text_model(root)
            sparse = root / "sparse"
            stale_binary = sparse / "cameras.bin"
            stale_binary.write_bytes(b"stale binary camera A")

            first = train.load_scene(train.TrainConfig(data=str(root), iterations=1))
            stale_binary.write_bytes(b"stale binary camera B")
            stale_changed = train.load_scene(
                train.TrainConfig(data=str(root), iterations=1)
            )
            self.assertEqual(first.fingerprint, stale_changed.fingerprint)

            cameras_text = sparse / "cameras.txt"
            cameras_text.write_text(
                cameras_text.read_text(encoding="utf-8").replace(
                    "14 15 8 6", "13 15 8 6"
                ),
                encoding="utf-8",
            )
            effective_input_changed = train.load_scene(
                train.TrainConfig(data=str(root), iterations=1)
            )
            self.assertNotEqual(first.fingerprint, effective_input_changed.fingerprint)
            self.assertEqual(effective_input_changed.cameras[1].params[0], 13.0)

    def test_mixed_model_components_are_not_a_complete_colmap_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_text_model(root)
            sparse = root / "sparse"
            (sparse / "cameras.txt").unlink()
            (sparse / "cameras.bin").write_bytes(b"camera")
            self.assertFalse(train.has_colmap_model(sparse))

    def test_text_scene_with_spaces_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_text_model(root)
            scene = train.load_scene(train.TrainConfig(data=str(root), iterations=1))
            self.assertEqual(len(scene.frames), 4)
            self.assertEqual(scene.frames[0].name, "frame 00.png")
            self.assertEqual(len(scene.points.xyz), 5)
            self.assertEqual(len(scene.fingerprint), 64)

    def test_unsafe_registered_image_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(train.UserFacingError, "Unsafe image path"):
            train.normalize_image_name("../secret.jpg")

    def test_images_text_requires_a_valid_points2d_line(self) -> None:
        malformed_records = {
            "missing": "1 1 0 0 0 0 0 0 1 frame.png\n",
            "wrong_arity": "1 1 0 0 0 0 0 0 1 frame.png\n10 20\n",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.txt"
            for name, payload in malformed_records.items():
                with self.subTest(name=name):
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaises(train.UserFacingError):
                        train.parse_images_text(path)

    def test_points_text_rejects_incomplete_track_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "points3D.txt"
            path.write_text(
                "\n".join(
                    f"{index} {index} 0 2 255 128 32 0.1 1"
                    for index in range(1, 5)
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(train.UserFacingError):
                train.parse_points_text(path)

    def test_images_text_rejects_nonfinite_translation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.txt"
            path.write_text(
                "1 1 0 0 0 nan 0 0 1 frame.png\n\n",
                encoding="utf-8",
            )
            with self.assertRaises(train.UserFacingError):
                train.parse_images_text(path)

    def test_split_is_deterministic_and_nonempty(self) -> None:
        frames = tuple(
            train.FrameRecord(index, 1, f"{index}.png", np.eye(4))
            for index in range(8)
        )
        training, validation = train.split_frames(frames, 3)
        self.assertEqual([item.image_id for item in validation], [0, 3, 6])
        self.assertEqual(len(training), 5)


class ImageAndInitializationTests(unittest.TestCase):
    def test_image_resize_and_intrinsics_stay_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_text_model(root)
            config = train.TrainConfig(data=str(root), iterations=1, downscale=2)
            scene = train.load_scene(config)
            image, intrinsic = train.ImageStore(scene, config).get(scene.frames[0])
            self.assertEqual(tuple(image.shape), (6, 8, 3))
            self.assertAlmostEqual(float(intrinsic[0, 0]), 7.0)
            self.assertAlmostEqual(float(intrinsic[0, 2]), 4.0)

    def test_image_aspect_ratio_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_text_model(root)
            Image.new("RGB", (16, 10), (20, 80, 160)).save(
                root / "images" / "frame 00.png"
            )
            config = train.TrainConfig(data=str(root), iterations=1)
            with self.assertRaises(train.UserFacingError):
                scene = train.load_scene(config)
                train.ImageStore(scene, config).get(scene.frames[0])

    def test_duplicate_points_get_finite_scales(self) -> None:
        points = np.asarray(
            [[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32
        )
        scales = train.estimate_log_scales(points, 1.0, 1.0)
        self.assertEqual(scales.shape, (4, 3))
        self.assertTrue(np.isfinite(scales).all())

    def test_initial_parameter_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_text_model(root)
            config = train.TrainConfig(data=str(root), iterations=1, sh_degree=3)
            scene = train.load_scene(config)
            splats = train.initialize_splats(scene, config, torch.device("cpu"))
            self.assertEqual(set(splats), {"means", "scales", "quats", "opacities", "sh0", "shN"})
            self.assertEqual(tuple(splats["shN"].shape), (5, 15, 3))
            self.assertTrue(all(bool(torch.isfinite(value).all()) for value in splats.values()))


class LossExportAndStateTests(unittest.TestCase):
    def test_identity_loss_and_ssim(self) -> None:
        image = torch.rand(1, 12, 16, 3, requires_grad=True)
        total, l1, score = train.image_loss(image, image, 0.2)
        self.assertAlmostEqual(float(total), 0.0, places=5)
        self.assertAlmostEqual(float(l1), 0.0, places=7)
        self.assertAlmostEqual(float(score), 1.0, places=5)
        total.backward()
        self.assertTrue(bool(torch.isfinite(image.grad).all()))

    def test_ply_schema_and_payload_size(self) -> None:
        splats = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.zeros(2, 3)),
                "scales": torch.nn.Parameter(torch.zeros(2, 3)),
                "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]]).repeat(2, 1)),
                "opacities": torch.nn.Parameter(torch.zeros(2)),
                "sh0": torch.nn.Parameter(torch.zeros(2, 1, 3)),
                "shN": torch.nn.Parameter(torch.zeros(2, 0, 3)),
            }
        )
        payload = train.encode_ply(splats)
        marker = b"end_header\n"
        offset = payload.index(marker) + len(marker)
        header = payload[:offset].decode("ascii")
        self.assertIn("element vertex 2", header)
        self.assertIn("property float opacity", header)
        self.assertEqual(len(payload) - offset, 2 * 14 * 4)

    def test_degree_three_ply_is_identical_across_multiple_chunks(self) -> None:
        count = 5
        generator = torch.Generator().manual_seed(91)
        quats = torch.randn(count, 4, generator=generator)
        quats[:, 0] += 2.0
        splats = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(
                    torch.arange(count * 3, dtype=torch.float32).reshape(count, 3)
                    / 10.0
                ),
                "scales": torch.nn.Parameter(
                    torch.linspace(-1.0, 1.0, count * 3).reshape(count, 3)
                ),
                "quats": torch.nn.Parameter(quats),
                "opacities": torch.nn.Parameter(torch.linspace(-2.0, 2.0, count)),
                "sh0": torch.nn.Parameter(
                    torch.randn(count, 1, 3, generator=generator)
                ),
                "shN": torch.nn.Parameter(
                    torch.randn(count, 15, 3, generator=generator)
                ),
            }
        )

        encoded = train.encode_ply(splats)
        rows, rest_fields = train.validate_ply_schema(splats)
        chunks = list(train.iter_ply_payload(splats, chunk_size=2))
        streamed = train.ply_header(rows, rest_fields) + b"".join(chunks)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(streamed, encoded)
        marker = b"end_header\n"
        data_offset = encoded.index(marker) + len(marker)
        self.assertEqual(len(encoded) - data_offset, count * 59 * 4)

    def test_nonfinite_ply_is_rejected(self) -> None:
        splats = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.tensor([[math.nan, 0.0, 0.0]])),
                "scales": torch.nn.Parameter(torch.zeros(1, 3)),
                "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]])),
                "opacities": torch.nn.Parameter(torch.zeros(1)),
                "sh0": torch.nn.Parameter(torch.zeros(1, 1, 3)),
                "shN": torch.nn.Parameter(torch.zeros(1, 0, 3)),
            }
        )
        with self.assertRaisesRegex(train.UserFacingError, "NaN/Inf"):
            train.encode_ply(splats)

    def test_zero_quaternion_ply_is_rejected(self) -> None:
        splats = make_minimal_splats(quats=torch.zeros(1, 4))
        with self.assertRaises(train.UserFacingError):
            train.encode_ply(splats)

    def test_nonfinite_activated_scale_ply_is_rejected(self) -> None:
        splats = make_minimal_splats(scales=torch.full((1, 3), 100.0))
        self.assertFalse(bool(torch.isfinite(torch.exp(splats["scales"])).all()))
        with self.assertRaises(train.UserFacingError):
            train.encode_ply(splats)

    def test_atomic_json_replaces_complete_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            train.atomic_write_json(path, {"value": 1})
            train.atomic_write_json(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 2})
            self.assertEqual(list(path.parent.glob("*.tmp-*")), [])

    def test_atomic_copy_replaces_destination_without_temp_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "published" / "scene.ply"
            payload = (bytes(range(251)) * 8_500) + b"complete"
            source.write_bytes(payload)
            destination.parent.mkdir()
            destination.write_bytes(b"stale")

            train.atomic_copy_file(source, destination)

            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(list(destination.parent.glob("*.tmp-*")), [])

    def test_resume_metrics_keep_only_monotonic_checkpoint_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            rows = [
                {"step": 1, "loss": 0.9},
                {"step": 2, "loss": 0.8},
                {"step": 3, "loss": 0.7},
                {"step": 2, "loss": 99.0},
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\ntruncated",
                encoding="utf-8",
            )

            kept = train.truncate_metrics_for_resume(path, completed_step=1)

            self.assertEqual(kept, 2)
            remaining = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(remaining, rows[:2])
            self.assertEqual(list(path.parent.glob("*.tmp-*")), [])

    def test_checkpoint_reconstructs_dynamic_parameters_and_adam(self) -> None:
        config = train.TrainConfig(data="unused", iterations=10, sh_degree=0)
        splats = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.randn(3, 3)),
                "scales": torch.nn.Parameter(torch.zeros(3, 3)),
                "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]]).repeat(3, 1)),
                "opacities": torch.nn.Parameter(torch.zeros(3)),
                "sh0": torch.nn.Parameter(torch.zeros(3, 1, 3)),
                "shN": torch.nn.Parameter(torch.zeros(3, 0, 3)),
            }
        )
        optimizers = train.make_optimizers(splats, config, 2.0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.9)
        splats["means"].sum().backward()
        optimizers["means"].step()
        optimizers["means"].zero_grad(set_to_none=True)
        scheduler.step()
        frame = train.FrameRecord(1, 1, "frame.png", np.eye(4))
        scene = train.SceneRecord(
            data_root=Path("."),
            model_path=Path("."),
            images_path=Path("."),
            cameras={},
            frames=(frame,),
            points=train.PointCloudRecord(
                xyz=np.zeros((4, 3), np.float32),
                rgb=np.zeros((4, 3), np.float32),
                errors=np.zeros(4, np.float32),
                track_lengths=np.zeros(4, np.int32),
            ),
            fingerprint="abc",
            scene_scale=2.0,
        )
        sampler = random.Random(7)
        payload = train.checkpoint_payload(
            4,
            splats,
            optimizers,
            scheduler,
            {"scene_scale": 2.0, "grad2d": torch.zeros(3)},
            config,
            scene,
            (frame,),
            (),
            sampler,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            train.atomic_torch_save(path, payload)
            loaded = train.load_checkpoint(path, torch.device("cpu"))
        restored = train.parameters_from_checkpoint(loaded, torch.device("cpu"))
        restored_optimizers = train.make_optimizers(restored, config, 2.0)
        for name, optimizer in restored_optimizers.items():
            optimizer.load_state_dict(loaded["optimizers"][name])
        self.assertEqual(loaded["completed_step"], 4)
        self.assertEqual(tuple(restored["means"].shape), (3, 3))
        self.assertTrue(restored_optimizers["means"].state)

    def test_checkpoint_cpu_cap_preserves_every_dynamic_row(self) -> None:
        count = 8
        config = train.TrainConfig(data="unused", iterations=10, sh_degree=3)
        splats = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(
                    torch.arange(count * 3, dtype=torch.float32).reshape(count, 3)
                ),
                "scales": torch.nn.Parameter(torch.zeros(count, 3)),
                "quats": torch.nn.Parameter(
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1)
                ),
                "opacities": torch.nn.Parameter(torch.linspace(-4.0, 3.0, count)),
                "sh0": torch.nn.Parameter(torch.zeros(count, 1, 3)),
                "shN": torch.nn.Parameter(torch.zeros(count, 15, 3)),
            }
        )
        optimizers = train.make_optimizers(splats, config, 1.0)
        sum(parameter.sum() for parameter in splats.values()).backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        payload = {
            "splats": splats.state_dict(),
            "optimizers": {
                name: optimizer.state_dict()
                for name, optimizer in optimizers.items()
            },
            "strategy_state": {
                "scene_scale": 1.0,
                "grad2d": torch.arange(count, dtype=torch.float32),
                "count": torch.arange(count, dtype=torch.float32) + 10.0,
            },
        }
        keep = torch.topk(
            torch.sigmoid(payload["splats"]["opacities"]),
            k=4,
            largest=True,
        ).indices.sort().values
        expected_splats = {
            name: value[keep].clone()
            for name, value in payload["splats"].items()
        }
        expected_optimizer_rows: dict[str, list[dict[str, torch.Tensor]]] = {}
        for name, optimizer_payload in payload["optimizers"].items():
            expected_optimizer_rows[name] = []
            for state in optimizer_payload["state"].values():
                expected_optimizer_rows[name].append(
                    {
                        key: (value[keep].clone() if value.ndim > 0 else value.clone())
                        for key, value in state.items()
                        if isinstance(value, torch.Tensor)
                    }
                )

        removed = train.cap_checkpoint_payload(payload, maximum=4)

        self.assertEqual(removed, 4)
        for name, expected in expected_splats.items():
            torch.testing.assert_close(payload["splats"][name], expected)
        torch.testing.assert_close(
            payload["strategy_state"]["grad2d"],
            torch.arange(count, dtype=torch.float32)[keep],
        )
        torch.testing.assert_close(
            payload["strategy_state"]["count"],
            (torch.arange(count, dtype=torch.float32) + 10.0)[keep],
        )
        for name, optimizer_payload in payload["optimizers"].items():
            for state, expected in zip(
                optimizer_payload["state"].values(),
                expected_optimizer_rows[name],
                strict=True,
            ):
                for key, expected_value in expected.items():
                    torch.testing.assert_close(state[key], expected_value)

        restored = train.parameters_from_checkpoint(payload, torch.device("cpu"))
        restored_optimizers = train.make_optimizers(restored, config, 1.0)
        for name, optimizer in restored_optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][name])
            self.assertIs(optimizer.param_groups[0]["params"][0], restored[name])
        train.validate_dynamic_state(
            restored,
            restored_optimizers,
            payload["strategy_state"],
        )

    def test_resume_restores_trajectory_config_but_keeps_overrides(self) -> None:
        saved = train.TrainConfig(
            data="original-scene",
            iterations=10_000,
            downscale=4,
            max_gaussians=750_000,
            means_lr=7.5e-5,
            background="white",
            grow_gradient=8e-4,
            seed=314,
        )
        omitted_overrides = train.TrainConfig(
            data="replacement-scene",
            iterations=40_000,
            downscale=2,
            max_gaussians=500_000,
        )

        restored_report = train.restore_checkpoint_config(
            omitted_overrides,
            vars(saved).copy(),
        )

        self.assertEqual(omitted_overrides.means_lr, saved.means_lr)
        self.assertEqual(omitted_overrides.background, saved.background)
        self.assertEqual(omitted_overrides.grow_gradient, saved.grow_gradient)
        self.assertEqual(omitted_overrides.seed, saved.seed)
        self.assertEqual(omitted_overrides.data, "replacement-scene")
        self.assertEqual(omitted_overrides.iterations, saved.iterations)
        self.assertEqual(omitted_overrides.downscale, saved.downscale)
        self.assertEqual(omitted_overrides.max_gaussians, saved.max_gaussians)
        self.assertIn("means_lr", restored_report["restored_fields"])
        self.assertIn("iterations", restored_report["restored_fields"])

        explicit_overrides = train.TrainConfig(
            data="replacement-scene",
            iterations=40_000,
            downscale=2,
            max_gaussians=500_000,
        )
        explicit_overrides._explicit_fields = train.explicit_argument_destinations(
            train.build_parser(),
            [
                "--iterations=40000",
                "--downscale",
                "2",
                "--max-gaussians",
                "500000",
            ],
        )
        override_report = train.restore_checkpoint_config(
            explicit_overrides,
            vars(saved).copy(),
        )

        self.assertEqual(explicit_overrides.iterations, 40_000)
        self.assertEqual(explicit_overrides.downscale, 2)
        self.assertEqual(explicit_overrides.max_gaussians, 500_000)
        self.assertIn("iterations", override_report["allowed_overrides"])
        self.assertIn("downscale", override_report["allowed_overrides"])
        self.assertIn("max_gaussians", override_report["allowed_overrides"])

    def test_local_gsplat_build_version_normalizes_to_release(self) -> None:
        self.assertEqual(
            train.package_base_version("1.5.3+pt24cu124"),
            train.EXPECTED_GSPLAT_VERSION,
        )
        self.assertEqual(train.package_base_version("1.5.3"), "1.5.3")
        self.assertEqual(
            train.package_base_version("1.5.3.post1+vendor"),
            "1.5.3.post1",
        )

    def test_malformed_checkpoint_envelope_is_rejected(self) -> None:
        malformed_payloads = {
            "missing_rng": {
                "format_version": train.FORMAT_VERSION,
                "completed_step": 0,
                "splats": {},
                "optimizers": {},
                "scheduler": {},
                "strategy_state": {},
                "config": {},
                "scene_fingerprint": "abc",
                "train_image_ids": [],
                "val_image_ids": [],
            },
            "wrong_container_types": {
                "format_version": train.FORMAT_VERSION,
                "completed_step": "zero",
                "splats": [],
                "optimizers": [],
                "scheduler": [],
                "strategy_state": [],
                "config": [],
                "scene_fingerprint": 123,
                "train_image_ids": "1",
                "val_image_ids": "2",
                "rng": [],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, payload in malformed_payloads.items():
                with self.subTest(name=name):
                    path = root / f"{name}.pt"
                    train.atomic_torch_save(path, payload)
                    with self.assertRaises(train.UserFacingError):
                        train.load_checkpoint(path, torch.device("cpu"))

    def test_rng_capture_and_restore(self) -> None:
        train.set_determinism(123)
        sampler = random.Random(456)
        state = train.capture_rng_state(sampler)
        expected = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
            sampler.random(),
        )
        train.restore_rng_state(state, sampler)
        actual = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
            sampler.random(),
        )
        self.assertEqual(expected, actual)

    def test_refinement_can_be_disabled(self) -> None:
        config = train.TrainConfig(data="unused", iterations=1, refine_stop=0)
        train.validate_config(config)

    def test_negative_artifact_interval_is_rejected(self) -> None:
        config = train.TrainConfig(data="unused", iterations=1, preview_every=-1)
        with self.assertRaisesRegex(train.UserFacingError, "cannot be negative"):
            train.validate_config(config)

    def test_seed_outside_numpy_range_is_rejected(self) -> None:
        for seed in (-1, 2**32):
            with self.subTest(seed=seed):
                config = train.TrainConfig(data="unused", iterations=1, seed=seed)
                with self.assertRaises(train.UserFacingError):
                    train.validate_config(config)

    def test_nonfinite_config_values_are_rejected(self) -> None:
        invalid_values = {
            "init_scale": math.nan,
            "means_lr": math.inf,
            "ssim_weight": math.nan,
            "near_plane": math.inf,
            "gradient_clip": math.nan,
            "point_error_max": math.nan,
        }
        for field, value in invalid_values.items():
            with self.subTest(field=field, value=value):
                config = train.TrainConfig(data="unused", iterations=1)
                setattr(config, field, value)
                with self.assertRaises(train.UserFacingError):
                    train.validate_config(config)

    def test_output_directory_cannot_overlap_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "dataset"
            data.mkdir()
            config = train.TrainConfig(
                data=str(data),
                output_dir=str(data / "training-output"),
                iterations=1,
            )
            with self.assertRaisesRegex(train.UserFacingError, "overlap"):
                train.prepare_output_directory(
                    config,
                    "scene",
                    protected_paths=(data,),
                )

    def test_overwrite_cannot_target_repository_source_subdirectory(self) -> None:
        repository = Path(train.__file__).resolve().parents[1]
        config = train.TrainConfig(
            data="unused",
            output_dir=str(repository / "viewer" / "src"),
            overwrite=True,
            iterations=1,
        )
        with mock.patch.object(
            train,
            "acquire_run_lock",
            side_effect=AssertionError("source check must happen before locking"),
        ):
            with self.assertRaisesRegex(train.UserFacingError, "source-tree"):
                train.prepare_output_directory(config, "scene")

    def test_branched_resume_requires_an_empty_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "source-run" / "checkpoints" / "step.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint placeholder")
            branch = root / "branch-run"
            branch.mkdir()
            (branch / "existing.txt").write_text("keep", encoding="utf-8")
            config = train.TrainConfig(
                data="unused",
                output_dir=str(branch),
                resume=str(checkpoint),
                iterations=1,
            )
            with self.assertRaisesRegex(train.UserFacingError, "branched resume"):
                train.prepare_output_directory(config, "scene")


if __name__ == "__main__":
    unittest.main()
