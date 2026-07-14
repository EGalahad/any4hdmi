from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from any4hdmi import FullMotionDataset, MotionData, MotionDatasetRouter


def _full_dataset(offset: float) -> FullMotionDataset:
    scalar = torch.tensor([offset, offset + 1]).reshape(2, 1, 1)
    vec3 = scalar.expand(-1, 1, 3).clone()
    quat = torch.zeros((2, 1, 4))
    quat[..., 0] = 1
    return FullMotionDataset(
        body_names=["pelvis"],
        joint_names=["joint"],
        motion_paths=[Path(f"{offset}.npz")],
        starts=[0],
        ends=[2],
        data=MotionData(
            motion_id=torch.tensor([0, 0]),
            step=torch.tensor([0, 1]),
            body_pos_w=vec3,
            body_lin_vel_w=torch.zeros_like(vec3),
            body_quat_w=quat,
            body_ang_vel_w=torch.zeros_like(vec3),
            joint_pos=scalar.clone(),
            joint_vel=torch.zeros_like(scalar),
            batch_size=(2,),
            device=torch.device("cpu"),
        ),
        num_envs=4,
        output_float_dtype=torch.float32,
    )


def _windowed_stub(offset: float = 10.0):
    source = _full_dataset(offset)
    stub = SimpleNamespace(
        body_names=["pelvis"],
        joint_names=["joint"],
        motion_paths=[Path("windowed.npz")],
        starts=torch.tensor([0]),
        ends=torch.tensor([2]),
        lengths=torch.tensor([2]),
        num_motions=1,
        num_steps=2,
        sample_id_span=4,
        device=torch.device("cpu"),
    )
    stub.to = Mock(return_value=stub)
    stub.get_slice = Mock(
        side_effect=lambda motion_ids, starts, steps: source.get_slice(
            torch.zeros_like(motion_ids), starts, steps
        )
    )
    return stub


class MotionDatasetRouterTest(unittest.TestCase):
    def test_all_resident_children_share_one_fp16_store(self) -> None:
        first = _full_dataset(1)
        second = _full_dataset(3)
        router = MotionDatasetRouter([first, second]).to("cpu")

        self.assertIsNotNone(router._resident_data)
        self.assertEqual(router._resident_data.body_pos_w.dtype, torch.float16)
        self.assertEqual(
            first.data.body_pos_w.untyped_storage().data_ptr(),
            router._resident_data.body_pos_w.untyped_storage().data_ptr(),
        )
        result = router.get_slice(
            torch.tensor([1, 0]), torch.tensor([0, 0]), torch.tensor([0, 1])
        )
        torch.testing.assert_close(
            result.body_pos_w[:, :, 0, 0],
            torch.tensor([[3.0, 4.0], [1.0, 2.0]]),
        )
        self.assertEqual(result.body_pos_w.dtype, torch.float32)

    def test_mixed_runtime_routes_across_windowed_id_span(self) -> None:
        before = _full_dataset(1)
        windowed = _windowed_stub()
        after = _full_dataset(20)
        router = MotionDatasetRouter([before, windowed, after]).to("cpu")

        torch.testing.assert_close(router.route_offsets, torch.tensor([0, 1, 5]))
        torch.testing.assert_close(router.route_ends, torch.tensor([1, 5, 6]))
        motion_ids = torch.tensor([5, 4, 0, 1, 5, 0])
        starts = torch.tensor([0, 0, 1, 1, 1, 0])
        result = router.get_slice(motion_ids, starts, torch.tensor([0, 1]))
        torch.testing.assert_close(
            result.body_pos_w[:, :, 0, 0],
            torch.tensor(
                [
                    [20, 21],
                    [10, 11],
                    [2, 2],
                    [11, 11],
                    [21, 21],
                    [1, 2],
                ],
                dtype=torch.float32,
            ),
        )
        windowed.get_slice.assert_called_once()
        torch.testing.assert_close(
            windowed.get_slice.call_args.args[0], torch.tensor([3, 0])
        )

    def test_router_has_no_combined_frame_index_compatibility_api(self) -> None:
        router = MotionDatasetRouter([_full_dataset(1), _windowed_stub()])
        self.assertFalse(hasattr(router, "data"))

    def test_packed_sharded_child_keeps_local_runtime_motion_ids(self) -> None:
        sharded = _full_dataset(1)
        sharded.data.motion_id.fill_(2)
        sharded.motion_id_offset = 2
        router = MotionDatasetRouter([sharded]).to("cpu")
        result = router.get_slice(
            torch.tensor([0]), torch.tensor([0]), torch.tensor([0, 1])
        )
        torch.testing.assert_close(result.motion_id, torch.tensor([[0, 0]]))
        with patch(
            "any4hdmi.dataset.full.torch.randint", return_value=torch.tensor([0])
        ):
            sampled = sharded.sample_motion(
                torch.tensor([0]),
                terminated_t=torch.zeros(1, dtype=torch.long),
                rewind_mask=torch.zeros(1, dtype=torch.bool),
                rewind_steps=torch.zeros(1, dtype=torch.long),
            )
        torch.testing.assert_close(sampled.motion_id, torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
