import os
import sys

# Code to run the script iteratively for different demo episodes
demo_episodes_list = [1, 5, 10, 20, 50]
for demo_episodes in demo_episodes_list:
    print(f"Running training with {demo_episodes} demonstration episodes...")
    os.system(f"python train_sqil.py --env HalfCheetah-v4 --demo_path ../data/demonstrations/{demo_episodes}/halfcheetah_demonstrations_{demo_episodes}.npy --seed 44 --episodes 2000 --demo_episodes {demo_episodes}")
    print(f"Training with {demo_episodes} demonstration episodes completed.\n")