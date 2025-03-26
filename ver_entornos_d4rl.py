import d4rl
import gym

print("Entornos registrados con D4RL:")
for env_spec in gym.envs.registry.values():
    if 'cheetah' in env_spec.id.lower():
        print(env_spec.id)