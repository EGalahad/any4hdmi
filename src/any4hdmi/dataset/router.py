from __future__ import annotations

from collections.abc import Sequence

import torch

from any4hdmi.dataset.base import BaseDataset, MotionData
from any4hdmi.dataset.full import FullMotionDataset, convert_motion_data


_MOTION_DATA_FIELD_NAMES = (
    "motion_id",
    "step",
    "body_pos_w",
    "body_lin_vel_w",
    "body_quat_w",
    "body_ang_vel_w",
    "joint_pos",
    "joint_vel",
)
_FLOAT_MOTION_DATA_FIELD_NAMES = _MOTION_DATA_FIELD_NAMES[2:]


class MotionDatasetRouter:
    """Route runtime motion IDs and share one packed store across resident children."""

    def __init__(self, datasets: Sequence[BaseDataset]) -> None:
        if not datasets:
            raise ValueError("MotionDatasetRouter requires at least one dataset")
        self.datasets = list(datasets)
        self.body_names = list(self.datasets[0].body_names)
        self.joint_names = list(self.datasets[0].joint_names)
        for index, dataset in enumerate(self.datasets[1:], start=1):
            if list(dataset.body_names) != self.body_names:
                raise ValueError(f"datasets[{index}] body_names do not match")
            if list(dataset.joint_names) != self.joint_names:
                raise ValueError(f"datasets[{index}] joint_names do not match")

        self.device = torch.device(self.datasets[0].device)
        motion_offsets = []
        motion_ends = []
        route_offsets = []
        route_ends = []
        motion_offset = 0
        route_offset = 0
        step_offset = 0
        starts = []
        ends = []
        self.motion_paths = []
        for dataset in self.datasets:
            motion_offsets.append(motion_offset)
            motion_offset += int(dataset.num_motions)
            motion_ends.append(motion_offset)
            route_offsets.append(route_offset)
            route_offset += int(dataset.sample_id_span)
            route_ends.append(route_offset)
            starts.append(dataset.starts.to(self.device) + step_offset)
            ends.append(dataset.ends.to(self.device) + step_offset)
            step_offset += int(dataset.num_steps)
            self.motion_paths.extend(dataset.motion_paths)

        self.motion_id_offsets = torch.tensor(
            motion_offsets, device=self.device, dtype=torch.long
        )
        self.motion_id_ends = torch.tensor(
            motion_ends, device=self.device, dtype=torch.long
        )
        self.route_offsets = torch.tensor(
            route_offsets, device=self.device, dtype=torch.long
        )
        self.route_ends = torch.tensor(
            route_ends, device=self.device, dtype=torch.long
        )
        self.starts = torch.cat(starts)
        self.ends = torch.cat(ends)
        self.lengths = self.ends - self.starts
        self._resident_flags = [
            isinstance(dataset, FullMotionDataset) for dataset in self.datasets
        ]
        self._resident_dataset_mask = torch.tensor(
            self._resident_flags, device=self.device, dtype=torch.bool
        )
        self._resident_motion_id_offsets = torch.tensor(
            [getattr(dataset, "motion_id_offset", 0) for dataset in self.datasets],
            device=self.device,
            dtype=torch.long,
        )
        self._resident_data: MotionData | None = None
        self._resident_route_starts: torch.Tensor | None = None
        self._resident_route_ends: torch.Tensor | None = None

    @property
    def num_motions(self) -> int:
        return int(self.motion_id_ends[-1].item())

    @property
    def num_steps(self) -> int:
        return sum(int(dataset.num_steps) for dataset in self.datasets)

    def to(self, device: torch.device | str) -> MotionDatasetRouter:
        self.device = torch.device(device)
        resident_datasets = [
            dataset
            for dataset, resident in zip(self.datasets, self._resident_flags)
            if resident
        ]
        if resident_datasets:
            total_frames = sum(
                int(dataset.num_steps) for dataset in resident_datasets
            )
            packed_fields = {}
            for field_name in _MOTION_DATA_FIELD_NAMES:
                sources = [
                    getattr(dataset.data, field_name)
                    for dataset in resident_datasets
                ]
                output = torch.empty(
                    (total_frames, *sources[0].shape[1:]),
                    device=self.device,
                    dtype=(
                        torch.float16
                        if field_name in _FLOAT_MOTION_DATA_FIELD_NAMES
                        else sources[0].dtype
                    ),
                )
                frame_start = 0
                for dataset, source in zip(resident_datasets, sources):
                    frame_end = frame_start + int(dataset.num_steps)
                    output[frame_start:frame_end].copy_(source)
                    setattr(dataset.data, field_name, output[frame_start:frame_end])
                    frame_start = frame_end
                packed_fields[field_name] = output

            self._resident_data = MotionData(
                **packed_fields,
                device=self.device,
                batch_size=(total_frames,),
            )

        self.datasets = [dataset.to(self.device) for dataset in self.datasets]
        self.motion_id_offsets = self.motion_id_offsets.to(self.device)
        self.motion_id_ends = self.motion_id_ends.to(self.device)
        self.route_offsets = self.route_offsets.to(self.device)
        self.route_ends = self.route_ends.to(self.device)
        self.starts = self.starts.to(self.device)
        self.ends = self.ends.to(self.device)
        self.lengths = self.lengths.to(self.device)
        self._resident_dataset_mask = self._resident_dataset_mask.to(self.device)
        self._resident_motion_id_offsets = self._resident_motion_id_offsets.to(
            self.device
        )

        if resident_datasets:
            total_route_span = int(self.route_ends[-1].item())
            self._resident_route_starts = torch.full(
                (total_route_span,), -1, device=self.device, dtype=torch.long
            )
            self._resident_route_ends = torch.full_like(
                self._resident_route_starts, -1
            )
            frame_offset = 0
            for dataset_index, (dataset, resident) in enumerate(
                zip(self.datasets, self._resident_flags)
            ):
                if not resident:
                    continue
                route_ids = self.route_offsets[dataset_index] + torch.arange(
                    int(dataset.num_motions), device=self.device
                )
                self._resident_route_starts[route_ids] = (
                    dataset.starts + frame_offset
                )
                self._resident_route_ends[route_ids] = dataset.ends + frame_offset
                frame_offset += int(dataset.num_steps)
            print(
                "[any4hdmi][resident] packed"
                f" children={len(resident_datasets)} frames={total_frames}"
                " float_dtype=torch.float16"
            )
        return self

    def get_slice(
        self,
        motion_ids: torch.Tensor,
        starts: torch.Tensor,
        steps: torch.Tensor,
    ) -> MotionData:
        motion_ids = motion_ids.to(device=self.device, dtype=torch.long)
        starts = starts.to(device=self.device, dtype=torch.long)
        steps = steps.to(device=self.device, dtype=torch.long)
        if motion_ids.numel() == 0:
            return self.datasets[0].get_slice(motion_ids, starts, steps)

        dataset_ids = torch.bucketize(motion_ids, self.route_ends, right=True)
        parts = []
        positions = []
        if self._resident_data is not None:
            assert self._resident_route_starts is not None
            assert self._resident_route_ends is not None
            resident_positions = torch.nonzero(
                self._resident_dataset_mask[dataset_ids], as_tuple=False
            ).squeeze(-1)
            if resident_positions.numel():
                resident_ids = motion_ids[resident_positions]
                resident_starts = self._resident_route_starts[resident_ids]
                resident_ends = self._resident_route_ends[resident_ids]
                index = (
                    resident_starts + starts[resident_positions]
                ).unsqueeze(1) + steps.unsqueeze(0)
                index.clamp_max_(resident_ends.unsqueeze(1) - 1)
                index.clamp_min_(resident_starts.unsqueeze(1))
                resident_result = convert_motion_data(
                    self._resident_data[index],
                    float_dtype=torch.float32,
                )
                resident_result.motion_id = (
                    resident_result.motion_id
                    - self._resident_motion_id_offsets[
                        dataset_ids[resident_positions]
                    ].unsqueeze(1)
                )
                parts.append(resident_result)
                positions.append(resident_positions)

        for dataset_index, dataset in enumerate(self.datasets):
            if self._resident_data is not None and self._resident_flags[dataset_index]:
                continue
            member_positions = torch.nonzero(
                dataset_ids == dataset_index, as_tuple=False
            ).squeeze(-1)
            if not member_positions.numel():
                continue
            parts.append(
                dataset.get_slice(
                    motion_ids[member_positions] - self.route_offsets[dataset_index],
                    starts[member_positions],
                    steps,
                )
            )
            positions.append(member_positions)

        if not parts:
            raise IndexError("motion_ids do not route to any child dataset")
        if len(parts) == 1:
            return parts[0]
        merged_positions = torch.cat(positions)
        return torch.cat(parts, dim=0)[torch.argsort(merged_positions)]
