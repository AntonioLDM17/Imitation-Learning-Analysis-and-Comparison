import os
import sys
import types
import argparse

# Create dummy modules for "mujoco_py" (to avoid compiling its extensions)
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
# You can add minimal attributes if needed; for now, we leave them empty.
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

import gymnasium as gym
import numpy as np

def inspect_env(env_name):
    # Create the environment
    env = gym.make(env_name)
    # Reset environment; gymnasium returns a tuple, where the first element is the observation.
    reset_result = env.reset()
    raw_obs = reset_result[0]
    print("Raw observation from env.reset():")
    print(raw_obs)
    print("Observation type:", type(raw_obs))
    print("Observation shape:", np.array(raw_obs).shape)
    
    # Sample a random action from the action space (flat, not batched)
    sample_action = env.action_space.sample()
    print("\nSample action from env.action_space:")
    print(sample_action)
    print("Action type:", type(sample_action))
    print("Action shape:", np.array(sample_action).shape)
    
    # Step the environment with the flat action (not batched)
    result = env.step(sample_action)
    print("\nOutputs from env.step(action):")
    # Gymnasium's step() may return either 4 or 5 elements:
    for i, output in enumerate(result):
        if isinstance(output, np.ndarray):
            print(f"Output {i} is a numpy array with shape: {output.shape}")
        else:
            print(f"Output {i} (type {type(output)}): {output}")
    
    env.close()

def inspect_demonstrations(demo_path):
    # Load demonstration data (assuming .npy file saved with allow_pickle=True)
    demonstrations = np.load(demo_path, allow_pickle=True)
    print("\nLoaded demonstrations type:", type(demonstrations))
    
    if isinstance(demonstrations, np.ndarray):
        print("Demonstrations numpy array shape:", demonstrations.shape)
        first_demo = demonstrations[0]
    else:
        first_demo = demonstrations

    try:
        # Check if the first element is iterable (e.g., a tuple of transitions)
        _ = iter(first_demo)
        print("\nFirst demonstration is iterable. Type:", type(first_demo))
        print("Length of first demonstration:", len(first_demo))
        for idx, item in enumerate(first_demo):
            if hasattr(item, "shape"):
                print(f"Item {idx} shape: {np.array(item).shape}")
            else:
                print(f"Item {idx} type: {type(item)}")
    except TypeError:
        # In case it's not directly iterable, it might be a trajectory object with attributes.
        print("\nFirst demonstration is not directly iterable.")
        if hasattr(first_demo, "obs"):
            print("Trajectory attribute 'obs' length:", len(first_demo.obs))
            print("Shape of first observation:", np.array(first_demo.obs[0]).shape)
        else:
            print("No iterable attributes found in the demonstration object.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect environment and demonstration shapes.")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4",
                        help="Environment name to inspect (e.g., HalfCheetah-v4 or CartPole-v1)")
    parser.add_argument("--demo_path", type=str, default="demonstrations/halfcheetah_demonstrations.npy",
                        help="Path to the demonstration file (e.g., demonstrations/halfcheetah_demonstrations.npy)")
    args = parser.parse_args()
    
    print("=== Inspecting Environment ===")
    inspect_env(args.env)
    
    print("\n=== Inspecting Demonstrations ===")
    inspect_demonstrations(args.demo_path)
