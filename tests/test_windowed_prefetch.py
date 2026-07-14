from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest import mock

import torch

from any4hdmi.dataset.fk_cache import FKCacheEntry
from any4hdmi.dataset.windowed import WindowedMotionDataset, _contiguous_id_runs


def _entry() -> FKCacheEntry:
    lengths = [3, 4]
    frames = sum(lengths)
    frame_values = torch.arange(frames, dtype=torch.float16).reshape(frames, 1, 1)
    vec3 = frame_values.expand(-1, 1, 3).clone()
    quat = torch.zeros((frames, 1, 4), dtype=torch.float16)
    quat[..., 0] = frame_values.squeeze(-1)
    return FKCacheEntry(
        cache_entry_dir=Path("/tmp/test-windowed-prefetch"),
        body_names=["pelvis"],
        joint_names=["joint"],
        motion_paths=[Path("m0.npz"), Path("m1.npz")],
        starts=[0, 3],
        ends=[3, 7],
        storage_fields={
            "motion_id": torch.tensor([0, 0, 0, 1, 1, 1, 1]),
            "step": torch.tensor([0, 1, 2, 0, 1, 2, 3]),
            "body_pos_w": vec3,
            "body_lin_vel_w": vec3 + 10,
            "body_quat_w": quat,
            "body_ang_vel_w": vec3 + 20,
            "joint_pos": frame_values.squeeze(-1),
            "joint_vel": frame_values.squeeze(-1) + 30,
        },
    )


class WindowedPrefetchTest(unittest.TestCase):
    def _dataset(self, device: str = "cpu") -> WindowedMotionDataset:
        dataset = WindowedMotionDataset.from_cache_entry(
            _entry(),
            num_envs=6,
            device=device,
            next_window_device=device,
            pin_window_load=device == "cuda",
        )
        dataset._drain_pending_loads(wait=True)
        return dataset

    def test_direct_pool_preserves_sparse_order_and_clamps_motion_end(self) -> None:
        dataset = self._dataset()
        dataset._load_next_windows(
            env_ids=torch.tensor([4, 1, 3]),
            motion_ids=torch.tensor([1, 0, 1]),
            source_starts=torch.tensor([2, 1, 0]),
        )

        self.assertEqual(dataset._next_window.body_pos_w.dtype, torch.float16)
        torch.testing.assert_close(
            dataset._next_window.body_pos_w[4, :4, 0, 0],
            torch.tensor([5, 6, 6, 6], dtype=torch.float16),
        )
        torch.testing.assert_close(
            dataset._next_window.body_pos_w[1, :4, 0, 0],
            torch.tensor([1, 2, 2, 2], dtype=torch.float16),
        )
        torch.testing.assert_close(
            dataset._next_window.body_pos_w[3, :5, 0, 0],
            torch.tensor([3, 4, 5, 6, 6], dtype=torch.float16),
        )
        torch.testing.assert_close(
            dataset._next_window.motion_id[[4, 1, 3], 0],
            torch.tensor([1, 0, 1]),
        )

    def test_promotion_converts_fp16_next_pool_to_fp32_current_pool(self) -> None:
        dataset = self._dataset()
        dataset._load_next_windows(
            torch.tensor([4, 1]),
            torch.tensor([1, 0]),
            torch.tensor([1, 0]),
        )
        dataset._copy_motion_rows(
            dataset._current_window,
            torch.tensor([4, 1]),
            dataset._next_window[torch.tensor([4, 1])],
        )
        self.assertEqual(dataset._current_window.body_pos_w.dtype, torch.float32)
        torch.testing.assert_close(
            dataset._current_window.body_pos_w[4, :4, 0, 0],
            torch.tensor([4, 5, 6, 6], dtype=torch.float32),
        )

    def test_future_completion_is_the_readiness_contract(self) -> None:
        dataset = self._dataset()
        started = threading.Event()
        release = threading.Event()

        def blocked_load(*_args) -> None:
            started.set()
            release.wait(timeout=5)

        dataset._load_next_windows = blocked_load
        dataset._schedule_next_window_prefetch(
            torch.tensor([2]), torch.tensor([0]), torch.tensor([0])
        )
        self.assertTrue(started.wait(timeout=5))
        self.assertFalse(dataset._pending_loads[0].future.done())
        release.set()
        dataset._drain_pending_loads(wait=True, env_ids=torch.tensor([2]))
        self.assertEqual(dataset._pending_loads, [])

    def test_worker_stream_is_synchronized_before_future_returns(self) -> None:
        dataset = self._dataset()
        stream = SimpleNamespace(synchronize=mock.Mock())
        dataset._prefetch_cuda_stream = stream
        dataset._load_window_batch_into_pool = mock.Mock()
        with mock.patch(
            "any4hdmi.dataset.windowed.torch.cuda.stream",
            return_value=nullcontext(),
        ):
            dataset._load_next_windows(
                torch.tensor([2]), torch.tensor([0]), torch.tensor([0])
            )
        dataset._load_window_batch_into_pool.assert_called_once()
        stream.synchronize.assert_called_once_with()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_future_readiness_smoke(self) -> None:
        dataset = self._dataset("cuda")
        dataset._schedule_next_window_prefetch(
            torch.tensor([5], device="cuda"),
            torch.tensor([1], device="cuda"),
            torch.tensor([2], device="cuda"),
        )
        dataset._drain_pending_loads(
            wait=True, env_ids=torch.tensor([5], device="cuda")
        )
        torch.testing.assert_close(
            dataset._next_window.body_pos_w[5, :4, 0, 0].cpu(),
            torch.tensor([5, 6, 6, 6], dtype=torch.float16),
        )

    def test_contiguous_id_runs_preserve_sparse_destinations(self) -> None:
        self.assertEqual(
            _contiguous_id_runs(torch.tensor([1, 2, 5, 9, 10, 11])),
            [(0, 2, 1, 3), (2, 3, 5, 6), (3, 6, 9, 12)],
        )

    def test_uniform_specs_reconstruct_motion_index_without_full_gpu_index(self) -> None:
        dataset = self._dataset()
        with mock.patch(
            "any4hdmi.dataset.windowed.torch.randint",
            return_value=torch.tensor([0, 2, 3, 6]),
        ):
            motion_ids, starts, lengths = dataset._draw_uniform_window_specs(4)
        torch.testing.assert_close(motion_ids, torch.tensor([0, 0, 1, 1]))
        torch.testing.assert_close(starts, torch.zeros(4, dtype=torch.long))
        torch.testing.assert_close(lengths, torch.full((4,), 512))


if __name__ == "__main__":
    unittest.main()
