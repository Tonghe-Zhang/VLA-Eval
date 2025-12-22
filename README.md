
# Environment Installation Walkthrough


## Directory structure

Our recommended directory structure:
--quick_test_maniskill
|_SimplerEnvPlus
|_openpi
|_PretrainedModels
    |__big_vision
    |__pi0_base
    |__pi05_base
    |__pi0_sft
    |__pi05_sft
        |_your/specific/checkpoint/name


## Openpi environment
First, setup your openpi environment by running [this shell script](./setup_openpi_env.sh)

## Then install maniskill environment
```bash
uv pip install hydra-core colorama peft maniskill
```

## Download the quick-test-maniskill repo


## Download the SimplerEnvPlus repo


## Test your Pi05 checkpoint in Simpler env
```
bash /home/tonghe/Project/Manip/quick_test_maniskill/test/eval_openpi.sh
```