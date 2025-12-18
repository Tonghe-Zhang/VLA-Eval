# Robot Selection and Control Modes Guide

This guide explains how to use different robot arms (like WidowX, Panda, Fetch) with ManiSkill environments.

## Table of Contents
- [Quick Start](#quick-start)
- [Available Robots](#available-robots)
- [Control Modes](#control-modes)
- [Configuration Examples](#configuration-examples)
- [Common Issues](#common-issues)

---

## Quick Start

### Method 1: Use Pre-configured YAML files

```bash
# PushT with WidowX
bash test/eval_openpi.sh pusht_widowx

# Assembly with WidowX  
bash test/eval_openpi.sh peginsertion_widowx
bash test/eval_openpi.sh assemblingkits_widowx
```

### Method 2: Override via Command Line

```bash
# Override robot for any existing config
bash test/eval_openpi.sh pusht_config robot_uids=widowx250s control_mode=pd_ee_delta_pose

# Use WidowXAI robot
bash test/eval_openpi.sh pusht_config robot_uids=widowxai control_mode=pd_ee_delta_pose
```

### Method 3: Create Your Own Config

```yaml
# my_custom_config.yaml
env_id: "PushT-v1"
env_type: "tabletop-gripper"
robot_uids: "widowx250s"  # Specify robot here
control_mode: "pd_ee_delta_pose"  # Control mode compatible with robot
# ... rest of config
```

---

## Available Robots

### ManiSkill Supported Robots

| Robot UID | Full Name | DOF | Gripper | Use Case |
|-----------|-----------|-----|---------|----------|
| `panda` | Franka Emika Panda | 7 | Parallel | General manipulation |
| `panda_stick` | Panda with Stick | 7 | Tool | Pushing tasks (e.g., PushT) |
| `widowx250s` | WidowX 250S (Joint Control) | 6 | Parallel | Joint-space control only |
| `widowx250s_simpler` | WidowX 250S (SimplerEnv) | 6 | Parallel | **Real2Sim, EE control** ✅ |
| `widowxai` | WidowX AI | 6 | Parallel | Advanced WidowX |
| `widowxai_wristcam` | WidowX AI + Wrist Cam | 6 | Parallel | Vision-based manipulation |
| `fetch` | Fetch Mobile Manipulator | 7 | Parallel | Mobile manipulation |
| `ur_10e` | Universal Robots UR10e | 6 | N/A | Industrial tasks |
| `googlerobot` | Google Robot | 7 | Parallel | Research tasks |

### Robot Comparison

**Panda (Franka)**
- ✅ 7-DOF (more dexterous)
- ✅ Well-calibrated, accurate
- ✅ Default for most ManiSkill tasks
- ❌ Different from real-world WidowX setups

**WidowX250S**
- ✅ 6-DOF (simpler kinematics)
- ✅ Matches SimplerEnv configurations
- ✅ Good for Real2Sim transfer
- ✅ Lower cost hardware
- ❌ Slightly less dexterous than 7-DOF

---

## Control Modes

Different robots support different control modes. Here are the most common:

### For Panda Robot

| Control Mode | Action Space | Description |
|--------------|--------------|-------------|
| `pd_ee_pose` | (x, y, z, roll, pitch, yaw, gripper) | Absolute end-effector pose |
| `pd_ee_delta_pose` | (Δx, Δy, Δz, Δroll, Δpitch, Δyaw, Δgripper) | Delta pose (preferred) |
| `pd_joint_pos` | (q₁, q₂, ..., q₇, gripper) | Absolute joint positions |
| `pd_joint_delta_pos` | (Δq₁, Δq₂, ..., Δq₇, Δgripper) | Delta joint positions |

### For WidowX Robot

**⚠️ IMPORTANT: Use `widowx250s_simpler` for EE control!**

| Control Mode | Robot | Action Space | Description |
|--------------|-------|--------------|-------------|
| `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos` | `widowx250s_simpler` | (Δx, Δy, Δz, Δroll, Δpitch, Δyaw, Δgripper) | **SimplerEnv control (recommended)** ✅ |
| `pd_joint_pos` | `widowx250s` | (q₁, q₂, ..., q₆, gripper) | Absolute joint positions |
| `pd_joint_delta_pos` | `widowx250s` | (Δq₁, Δq₂, ..., Δq₆, Δgripper) | Delta joint positions |

### Control Mode Selection Guide

**Use `pd_ee_delta_pose` when:**
- Training policies in end-effector space
- Need intuitive Cartesian control
- Working with vision-based policies
- ✅ **Recommended for most use cases**

**Use `pd_ee_pose` when:**
- Need absolute positioning
- Have explicit pose targets
- ⚠️ Can be harder to learn due to absolute coordinates

**Use `pd_joint_*` when:**
- Have joint-space demonstrations
- Need precise joint control
- Working with joint-space policies

---

## Configuration Examples

### Example 1: Assembly with WidowX (SimplerEnv-style)

```yaml
env_id: "PegInsertionSide-v1"
robot_uids: "widowx250s_simpler"
control_mode: "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
proprioception_type: "qpos"  # 6 arm + 2 gripper = 8 dim
single_action_dim: 7  # 6 DOF + 1 gripper
```

### Example 2: Assembly with Panda (Default)

```yaml
env_id: "AssemblingKits-v1"
# robot_uids not specified -> uses environment default (panda)
control_mode: "pd_ee_pose"
proprioception_type: "qpos"  # 7 arm + 2 gripper = 9 dim
single_action_dim: 7  # 6 DOF + 1 gripper (7th joint often fixed)
```

### Example 3: SimplerEnv Tasks

```yaml
env_id: "PutCarrotOnPlateInScene-v1"
robot_uids: "widowx250s_simpler"  # Use _simpler variant!
control_mode: "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
proprioception_type: "qpos"  
single_action_dim: 7
```

### Example 4: Override Existing Config

```bash
# Take an existing Panda config and switch to WidowX SimplerEnv
bash test/eval_openpi.sh peginsertion_pi05 \
    robot_uids=widowx250s_simpler \
    control_mode=arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos \
    single_action_dim=7 \
    proprioception_type=qpos
```

---

## Action Space Dimensions

⚠️ **Important**: Action dimensions vary by robot!

### Panda (7-DOF)
- **EE Space**: 7 dimensions (3 position + 3 rotation + 1 gripper)
- **Joint Space**: 7 or 9 dimensions (7 arm joints + optional 2 gripper joints)
- **Proprioception (qpos)**: 9 dimensions (7 arm + 2 gripper)

### WidowX (6-DOF)
- **EE Space**: 7 dimensions (3 position + 3 rotation + 1 gripper)
- **Joint Space**: 6 or 8 dimensions (6 arm joints + optional 2 gripper joints)
- **Proprioception (qpos)**: 8 dimensions (6 arm + 2 gripper)

### PushT Special Case
- Uses `panda_stick` (no gripper control needed)
- **Action Space**: 6 dimensions (3 position + 3 rotation only)

---

## Common Issues

### Issue 1: Dimension Mismatch

**Error**: `Action shape mismatch: expected [B, 7] but got [B, 9]`

**Solution**: Update `single_action_dim` in your config:
```yaml
# For WidowX
single_action_dim: 7  # 6 DOF + 1 gripper

# For Panda
single_action_dim: 7  # or 9 depending on control mode
```

### Issue 2: Robot Not Supported

**Error**: `Robot 'widowx250s' not in SUPPORTED_ROBOTS`

**Solution**: Not all environments support all robots. Check environment source code or use default robot. For example, PushT only supports `panda_stick`.

To bypass (advanced):
```python
# In environment source code
SUPPORTED_ROBOTS = ["panda", "widowx250s", "fetch"]  # Add your robot
```

### Issue 3: Control Mode Not Available

**Error**: `Control mode 'pd_ee_delta_pose' not found`

**Solution**: Check available controllers for your robot:
```python
# Print available controllers
env = gym.make("PushT-v1", robot_uids="widowx250s")
print(env.agent._controller_configs.keys())
```

### Issue 4: Different Proprioception Dimensions

**Error**: Model expects different proprioception size

**Solution**: Update `proprioception_type` or retrain model:
```yaml
# Options:
proprioception_type: "qpos"      # Joint positions (most common)
proprioception_type: "ee_pose"   # End-effector pose (7-dim)
proprioception_type: "state"     # Full state (high-dim)
```

---

## Advanced: Custom Robot Support

If you want to add a completely custom robot:

1. **Create robot class** in `/Manip/ManiSkill/mani_skill/agents/robots/`
2. **Register robot**: Use `@register_agent()` decorator
3. **Define controllers**: Override `_controller_configs` property  
4. **Add to environment**: Update `SUPPORTED_ROBOTS` in task file
5. **Update configs**: Create YAML configs with new `robot_uids`

See ManiSkill documentation for details: https://maniskill.readthedocs.io/

---

## Summary

| Task Type | Recommended Robot | Control Mode | Config File |
|-----------|-------------------|--------------|-------------|
| SimplerEnv | `widowx250s` | `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos` | `carrot.yaml` |
| PushT | `panda_stick` | `pd_ee_delta_pose` | `pusht.yaml` |
| PushT (WidowX) | `widowx250s` | `pd_ee_delta_pose` | `pusht_widowx.yaml` |
| Assembly | `panda` or `widowx250s` | `pd_ee_delta_pose` | `*_widowx.yaml` |
| General | `panda` (default) | `pd_ee_delta_pose` | Most configs |

**Key Takeaways:**
- ✅ Always match `control_mode` to your robot
- ✅ Verify `single_action_dim` matches robot DOF
- ✅ Use WidowX for Real2Sim transfer
- ✅ Use Panda for general manipulation research
- ✅ Override via command line for quick experiments

---

**Questions or Issues?** Check ManiSkill docs or create an issue on GitHub.

