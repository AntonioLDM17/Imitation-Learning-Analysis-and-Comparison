import gymnasium as gym
import numpy as np
import torch
import time
from sqil_agent import SQILAgent
import os, sys, types, argparse
from torch.utils.tensorboard import SummaryWriter

# Create dummy modules for "mujoco_py" (to avoid compiling its extensions)
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

def load_demonstrations(demo_path):
    """
    Se espera que las demostraciones sean un archivo NumPy (.npy) que contenga una lista
    de objetos TrajectoryWithRew.
    """
    demos = np.load(demo_path, allow_pickle=True)
    return demos

def extract_transitions_from_trajectory(traj):
    """
    Dado un objeto TrajectoryWithRew, extrae transiciones en forma de tuplas:
    (estado, acción, estado_siguiente, done).
    Se asume que:
      - 'obs' es un array de observaciones,
      - 'acts' es un array de acciones,
      - 'terminal' indica si la trayectoria termina.
    Se generan transiciones para cada par consecutivo de observaciones. 
    La transición final se marca como done si traj.terminal es True.
    """
    transitions = []
    obs = traj.obs
    acts = traj.acts
    n = len(obs)
    for i in range(n - 1):
        done = (i == n - 2) and getattr(traj, "terminal", False)
        transitions.append((obs[i], acts[i], obs[i + 1], done))
    return transitions

def main():
    parser = argparse.ArgumentParser(description="Entrenar SQIL basado en DI‑engine (SAC) en PyTorch")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4",
                        help="Nombre del entorno Gym (por ejemplo, HalfCheetah-v4 o CartPole-v1)")
    parser.add_argument("--demo_path", type=str, default=None,
                        help="Ruta al archivo de demostraciones (.npy)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--episodes", type=int, default=1000, help="Número de episodios de entrenamiento")
    parser.add_argument("--max_steps", type=int, default=1000, help="Número máximo de pasos por episodio")
    parser.add_argument("--batch_size", type=int, default=256, help="Tamaño de batch para actualizaciones")
    parser.add_argument("--update_every", type=int, default=50, help="Número de pasos entre actualizaciones")
    args = parser.parse_args()
    
    # Configurar semilla
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Crear SummaryWriter para TensorBoard
    writer = SummaryWriter("logs/sqil")
    
    # Crear entorno
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    
    # Asumir que el espacio de acción es continuo (Box)
    if isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
        action_range = float(env.action_space.high[0])
    else:
        raise NotImplementedError("Esta implementación de SQIL está diseñada para entornos continuos.")
    
    agent = SQILAgent(state_dim, action_dim, action_range=action_range, batch_size=args.batch_size)
    
    # Cargar demostraciones si se proporciona un archivo
    if args.demo_path is not None:
        demos = load_demonstrations(args.demo_path)
        print(f"Cargando {len(demos)} demostraciones desde {args.demo_path}")
        for traj in demos:
            transitions = extract_transitions_from_trajectory(traj)
            for transition in transitions:
                state, action, next_state, done = transition
                agent.store_demo(state, action, next_state, done)
    else:
        print("No se proporcionó un archivo de demostraciones. ¡IMPORTANTE: SQIL requiere demostraciones expertas!")
    
    total_steps = 0
    start_time = time.time()
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed)
        episode_reward = 0
        for step in range(args.max_steps):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            # Se ignora la recompensa del entorno; se asigna 0 a las transiciones del agente
            agent.store_agent(state, action, next_state, float(done or truncated))
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            # Actualizar el agente cada 'update_every' pasos
            if total_steps % args.update_every == 0:
                losses = agent.update()
                if losses is not None:
                    critic_loss, actor_loss, alpha_loss = losses
                    print(f"Paso {total_steps}: Loss Critic={critic_loss:.4f}, Actor={actor_loss:.4f}, Alpha={alpha_loss:.4f}")
                    writer.add_scalar("Loss/Critic", critic_loss, total_steps)
                    writer.add_scalar("Loss/Actor", actor_loss, total_steps)
                    writer.add_scalar("Loss/Alpha", alpha_loss, total_steps)
            
            if done or truncated:
                break
        print(f"Episodio {episode} terminado. Recompensa acumulada: {episode_reward}")
        writer.add_scalar("Reward/Episode", episode_reward, episode)
    elapsed = time.time() - start_time
    print(f"Entrenamiento completado en {elapsed:.2f} segundos.")
    
    # Guardar los modelos entrenados
    torch.save(agent.actor.state_dict(), "models/sqil_actor.pth")
    torch.save(agent.critic.state_dict(), "models/sqil_critic.pth")
    env.close()
    writer.close()

if __name__ == "__main__":
    print("Example usage:")
    print("python train_sqil.py --env HalfCheetah-v4 --demo_path ../data/demonstrations/halfcheetah_demonstrations.npy --seed 42 --episodes 1000")
    print(" To see the training process, run:")
    print("tensorboard --logdir logs/sqil")
    main()
