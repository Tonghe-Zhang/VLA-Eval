# Action Parameters Convention

## Summary

This document clarifies the naming convention for action-related parameters in the OpenPI model evaluation pipeline.

## Parameter Definitions

### `action_horizon` (Model OUTPUT)
- **What**: The number of action steps the model generates in a single forward pass
- **Role**: OUTPUT dimension of the model's action prediction
- **Typical Value**: 8 (in your config)
- **Model Output Shape**: `[batch, action_horizon, action_dim]`
- **Purpose**: Determines how many future actions the model predicts

### `action_replan_horizon` (Execution INPUT)
- **What**: The number of actions actually executed before replanning
- **Role**: INPUT to the execution loop (how many actions to use)
- **Typical Value**: 5 (in your config)
- **Execution Shape**: `[batch, action_replan_horizon, action_dim]`
- **Purpose**: Determines how often to replan (shorter = more reactive)

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Observation Collection                                       │
│    obs = env.get_obs()                                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Model Inference (OUTPUT)                                     │
│    actions = model.sample_actions(obs)                          │
│    Shape: [batch, action_horizon=8, action_dim]                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Action Slicing (OUTPUT → INPUT)                              │
│    actions = actions[:, :action_replan_horizon, :]              │
│    Shape: [batch, action_replan_horizon=5, action_dim]          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Action Execution (INPUT)                                     │
│    for i in range(action_replan_horizon):                       │
│        env.step(actions[:, i, :])                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         └─────► REPEAT (replan with new obs)
```

## Why This Convention?

### Model Predictive Control (MPC) Pattern
This follows the standard MPC pattern in robotics:
1. **Predict Long**: Model generates a longer action sequence (`action_horizon=8`)
2. **Execute Short**: Only execute the first few actions (`action_replan_horizon=5`)
3. **Replan Often**: Get new observations and replan frequently
4. **Stay Robust**: Corrects for disturbances and model errors

### Benefits
- **Robustness**: Frequent replanning handles disturbances
- **Efficiency**: Don't need to regenerate full trajectory every step
- **Flexibility**: Can adjust `action_replan_horizon` without retraining model

## Configuration Example

```yaml
# In model_cfg/pi0_maniskill.yaml
action_horizon: 8              # Model generates 8-step sequences
action_replan_horizon: 5       # Execute 5, then replan

# In model_cfg/pi05_maniskill.yaml
action_horizon: 8              # Model generates 8-step sequences
action_replan_horizon: 5       # Execute 5, then replan
```

## Code Implementation

### In `eval_openpi.py`:
```python
def get_action(self, obs, proprioception, language_instruction):
    # Model generates action_horizon actions (OUTPUT)
    actions = self.model.sample_actions(...)  
    # Shape: [batch, action_horizon=8, action_dim]
    
    # Slice to action_replan_horizon for execution (INPUT)
    actions = actions[:, :self.model.config.action_replan_horizon, :]
    # Shape: [batch, action_replan_horizon=5, action_dim]
    
    return actions
```

### In `eval_base.py`:
```python
def run_test(self):
    for step in range(num_steps):
        # Get actions for next action_replan_horizon steps
        actions = self.get_action(obs, ...)
        
        # Execute action_replan_horizon actions
        for i in range(self.action_replan_horizon):
            obs, reward, done, info = env.step(actions[:, i, :])
            # ... record data ...
```

## Model Info Logging

The following files are automatically saved to the log directory:
- `model_architecture.txt`: Full model architecture and config
- `model_config.json`: Model config in JSON format
- `hydra_model_config.yaml`: Hydra config used to load the model
- `model_parameters.txt`: Parameter counts by module

These files are saved to: `log/quick_test_maniskill/video_test/{env_id}/{timestamp}/`

## Legacy Parameters (Removed)

- **`action_chunk`**: Was replaced by clearer `action_horizon` and `action_replan_horizon` split
  - Previously ambiguous whether it meant model output or execution input
  - Now we have explicit parameters for each concept

## Consistency Check ✅

Your current setup is **consistent**:
- ✅ `action_horizon=8`: Model OUTPUT (generates 8 actions)
- ✅ `action_replan_horizon=5`: Execution INPUT (use 5 actions)
- ✅ Model generates more than it executes (8 > 5) ← correct MPC pattern
- ✅ Clear separation between model prediction and execution

## References

- Model Predictive Control (MPC): A standard control strategy in robotics
- Receding Horizon Control: Execute partial plan, replan with new observations
- OpenPI paper: Uses similar action chunking strategies

