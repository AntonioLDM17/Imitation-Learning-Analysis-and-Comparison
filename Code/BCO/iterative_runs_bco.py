import os
import sys
dict_demo_episodes_x_policy_epochs = {
    5: 212,
    10: 106,
    20: 71,
    50: 31,
    100: 17
}
# Code to run the script iteratively for different demo episodes
for demo_episodes, policy_epochs in dict_demo_episodes_x_policy_epochs.items():
    print(f"Running training with {demo_episodes} demonstration episodes and {policy_epochs} policy epochs...")
    os.system(f"python train_bco.py --env halfcheetah --pre_interactions 300000 --alpha 0.0 --policy_epochs {policy_epochs} --demo_episodes {demo_episodes} --seed 44")
    print(f"Training with {demo_episodes} demonstration episodes completed.\n")
