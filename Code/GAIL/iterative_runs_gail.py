import os
import sys
# Code to run the script iteratively for different demo episodes
demo_episodes_list = [1, 5, 10, 20, 50]
for demo_episodes in demo_episodes_list:
    print(f"Running training with {demo_episodes} demonstration episodes...")
    os.system(f"python train_gail.py --env halfcheetah --timesteps 2000000 --seed 42 --demo_episodes {demo_episodes}")
    print(f"Training with {demo_episodes} demonstration episodes completed.\n")