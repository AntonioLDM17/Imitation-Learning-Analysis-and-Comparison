import gymnasium as gym
import numpy as np
import torch
import argparse
import os
import sys, types
from sqil_agent import SQILAgent

# Create dummy modules for "mujoco_py" (to avoid compiling its extensions)
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

def main():
    parser = argparse.ArgumentParser(description="Evaluate a SQIL trained model")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4",
                        help="Nombre del entorno Gym (ej: HalfCheetah-v4 o CartPole-v1)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Número de episodios para la evaluación")
    parser.add_argument("--model_dir", type=str, default="models",
                        help="Directorio donde están guardados los modelos (sqil_actor.pth, sqil_critic.pth)")
    args = parser.parse_args()

    # Configurar semilla
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Crear entorno
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    
    # Asumir que el espacio de acción es continuo (Box)
    if isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
        action_range = float(env.action_space.high[0])
    else:
        raise NotImplementedError("Esta evaluación está diseñada para entornos continuos.")

    # Instanciar el agente SQIL (los demás parámetros se pueden ajustar según los usados en entrenamiento)
    agent = SQILAgent(state_dim, action_dim, action_range=action_range, batch_size=256)

    # Cargar pesos entrenados
    actor_path = os.path.join(args.model_dir, "sqil_actor.pth")
    critic_path = os.path.join(args.model_dir, "sqil_critic.pth")
    if not os.path.exists(actor_path) or not os.path.exists(critic_path):
        raise FileNotFoundError("No se encontraron los archivos de modelo en el directorio especificado.")
    
    agent.actor.load_state_dict(torch.load(actor_path, map_location=torch.device("cpu")))
    agent.critic.load_state_dict(torch.load(critic_path, map_location=torch.device("cpu")))
    print(f"Modelos cargados desde {args.model_dir}")

    # Evaluación: ejecutar el agente en n episodios y medir recompensa acumulada
    rewards = []
    for ep in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + ep)  # Cambiar la semilla por episodio para mayor variabilidad
        done = False
        ep_reward = 0
        while not done:
            # Seleccionar acción (se utiliza la función select_action definida en el agente)
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            state = next_state
            if done or truncated:
                break
        rewards.append(ep_reward)
        print(f"Episodio {ep}: Recompensa acumulada = {ep_reward:.2f}")
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"\nEvaluación sobre {args.episodes} episodios:")
    print(f"  Recompensa media: {mean_reward:.2f}")
    print(f"  Desviación estándar: {std_reward:.2f}")
    env.close()

if __name__ == "__main__":
    print("Uso de ejemplo:")
    print("python evaluate_sqil.py --env HalfCheetah-v4 --seed 42 --episodes 50 --model_dir models")
    main()
