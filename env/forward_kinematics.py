

# MIT License

# Copyright (c) 2025 RL4VLA Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



import torch
import pytorch_kinematics as pk 
from ManiSkill.mani_skill.envs.sapien_env import BaseEnv
from ManiSkill.mani_skill.utils.structs.pose import Pose
from ManiSkill.mani_skill.agents.controllers.utils.kinematics import Kinematics
from ManiSkill.mani_skill.agents.utils import get_active_joint_indices

def forward_kinematics(env: BaseEnv, kinematics: Kinematics, qpos, world_frame:bool = False):
    """Transfer joint positions to ee pose in robot base frame (or world frame if specified)
    
    Args:
        env: ManiSkill environment
        kinematics: Kinematics object
        qpos: Joint positions [B, joint_dof]
        world_frame: Whether to convert to world frame
    Returns:
        ee_pose: End effector pose in robot base frame (or world frame if specified)
        [B, 7] in robot base frame or world frame if specified. 7=3+4 = position + quaternion. ( to convert to action
    """
    qpos = torch.as_tensor(qpos.squeeze())
    
    # Fill the qpos_fk with the active joint indices, nullify the rest. Here, qpos_fk means the ee_pose. 
    qpos_fk = torch.zeros(qpos.shape[0], env.agent.robot.max_dof, # 8
                        dtype=qpos.dtype, device=env.agent.robot.device)
    qpos_fk[:, get_active_joint_indices(env.agent.robot, env.agent.arm_joint_names)] = qpos.to(device=env.agent.robot.device)
    
    if kinematics.use_gpu_ik:
        qpos_fk = qpos_fk[..., kinematics.active_ancestor_joint_idxs]
        tf_matrix = kinematics.pk_chain.forward_kinematics(qpos_fk.float()).get_matrix()
        pos = tf_matrix[:, :3, 3]
        rot = pk.matrix_to_quaternion(tf_matrix[:, :3, :3])
        ee_pose = Pose.create_from_pq(pos, rot, device=kinematics.device)
    else:
        kinematics.pmodel.compute_forward_kinematics(qpos_fk[0].cpu().numpy())
        ee_pose = kinematics.pmodel.get_link_pose(kinematics.end_link_idx)
        ee_pose = Pose.create(ee_pose, device=kinematics.device)

    # Convert to world frame if specified
    if world_frame: 
        return ee_pose * env.agent.robot.root.pose
    return ee_pose