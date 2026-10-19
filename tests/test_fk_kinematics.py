"""Reference FK must not run collision detection or change pose/velocity outputs."""
from unittest.mock import patch

import mujoco
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw
from any4hdmi.fk.runner import FKRunner


def test_warp_fk_matches_full_forward_without_collision(tmp_path):
    path = tmp_path / 'model.xml'
    path.write_text('''<mujoco><worldbody>
    <body name="robot" pos="0 0 1"><freejoint/><geom type="sphere" size=".1"/>
      <body pos=".1 0 0"><joint name="hinge" axis="0 1 0"/><geom type="capsule" size=".05 .1"/></body>
    </body><body name="object" pos="1 0 1"><freejoint/><geom type="box" size=".1 .2 .3"/></body>
    </worldbody></mujoco>''')
    runner = FKRunner(path, batch_size=3, device='cpu')
    with wp.ScopedDevice('cpu'):
        runner._wp, runner._mjw = wp, mjw
        runner._warp_model = mjw.put_model(runner.model)
        runner._warp_data = mjw.make_data(runner.model, nworld=3)
        qpos = torch.from_numpy(np.tile(runner.model.qpos0, (3, 1))).float()
        qpos[:, 0] += torch.tensor([.1, .2, .3])
        qpos[:, 7] = torch.tensor([.3, -.5, .7])
        qvel = torch.linspace(-.2, .3, 3 * runner.model.nv).reshape(3, -1)
        wp.copy(runner._warp_data.qpos, wp.from_torch(qpos))
        wp.copy(runner._warp_data.qvel, wp.from_torch(qvel))
        mjw.fwd_position(runner._warp_model, runner._warp_data)
        mjw.fwd_velocity(runner._warp_model, runner._warp_data)
        expected = {name: wp.to_torch(getattr(runner._warp_data, name)).clone()
                    for name in ('xpos', 'xquat', 'cvel')}
        with patch.object(mjw, 'fwd_position', side_effect=AssertionError('collision invoked')):
            result = runner._forward_kinematics_warp(qpos, qvel, synchronize=True)
        torch.testing.assert_close(result['body_pos_w'], expected['xpos'], rtol=0, atol=0)
        torch.testing.assert_close(result['body_quat_w'], expected['xquat'], rtol=0, atol=0)
        torch.testing.assert_close(result['body_lin_vel_w'], expected['cvel'][..., 3:], rtol=0, atol=0)
        torch.testing.assert_close(result['body_ang_vel_w'], expected['cvel'][..., :3], rtol=0, atol=0)
