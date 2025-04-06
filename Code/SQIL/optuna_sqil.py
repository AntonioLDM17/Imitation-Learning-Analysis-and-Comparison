import os
import argparse
import time
import numpy as np
import gymnasium as gym
import torch
import optuna

from sqil_agent import SQILAgent

# Create dummy modules for "mujoco_py" (to avoid compiling its extensions)
import sys, types
dummy = types.ModuleType("mujoco_py")
dummy.builder = types.ModuleType("mujoco_py.builder")
dummy.locomotion = types.ModuleType("mujoco_py.locomotion")
sys.modules["mujoco_py"] = dummy
sys.modules["mujoco_py.builder"] = dummy.builder
sys.modules["mujoco_py.locomotion"] = dummy.locomotion

from torch.utils.tensorboard import SummaryWriter

# Parámetros fijos para el entorno y episodios de entrenamiento/evaluación
ENV_NAME = "HalfCheetah-v4"
TRAIN_EPISODES = 500
EVAL_EPISODES = 50

def objective(trial: optuna.Trial):
    # Sugerir hiperparámetros
    actor_lr = trial.suggest_loguniform("actor_lr", 1e-5, 1e-3)
    critic_lr = trial.suggest_loguniform("critic_lr", 1e-5, 1e-3)
    alpha_lr  = trial.suggest_loguniform("alpha_lr", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    update_every = trial.suggest_categorical("update_every", [10, 50, 100])
    max_steps = trial.suggest_categorical("max_steps", [500, 1000, 1500])
    
    gamma = 0.99
    tau = 0.005
    target_entropy = None  # Se calculará automáticamente en SQILAgent
    demo_buffer_capacity = 100000
    agent_buffer_capacity = 100000
    hidden_dim = 256

    # Configurar semilla (variamos la semilla entre trials)
    seed = 42 + trial.number
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Crear un SummaryWriter específico para este trial
    writer = SummaryWriter(log_dir=f"logs/optuna_trial_{trial.number}")
    
    # Crear entorno
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
        action_range = float(env.action_space.high[0])
    else:
        raise NotImplementedError("Este script está diseñado para entornos continuos.")
    
    # Instanciar el agente SQIL con los hiperparámetros sugeridos
    agent = SQILAgent(
        state_dim, action_dim, action_range=action_range,
        actor_hidden=hidden_dim, critic_hidden=hidden_dim,
        actor_lr=actor_lr, critic_lr=critic_lr, alpha_lr=alpha_lr,
        gamma=gamma, tau=tau, target_entropy=target_entropy,
        demo_buffer_capacity=demo_buffer_capacity,
        agent_buffer_capacity=agent_buffer_capacity,
        batch_size=batch_size
    )
    
    # Cargar demostraciones (ajustar la ruta según la estructura del proyecto)
    demo_path = os.path.join("..", "data", "demonstrations", "halfcheetah_demonstrations.npy")
    if not os.path.exists(demo_path):
        raise FileNotFoundError(f"No se encontró el archivo de demostraciones en {demo_path}")
    demos = np.load(demo_path, allow_pickle=True)
    print(f"Trial {trial.number}: Cargando {len(demos)} demostraciones desde {demo_path}")
    # Extraer transiciones de cada trayectoria y almacenarlas en el buffer de demostraciones
    for traj in demos:
        obs = traj.obs
        acts = traj.acts
        n = len(obs)
        for i in range(n - 1):
            done = (i == n - 2) and getattr(traj, "terminal", False)
            agent.store_demo(obs[i], acts[i], obs[i + 1], done)
    print(f"Trial {trial.number}: Buffer de demostraciones cargado con {len(agent.demo_buffer)} transiciones.")
    writer.add_text("Demo/Info", f"Buffer de demostraciones: {len(agent.demo_buffer)} transiciones.", global_step=0)
    
    # Entrenar el agente durante TRAIN_EPISODES episodios
    total_steps = 0
    reward_history = []
    for episode in range(1, TRAIN_EPISODES + 1):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            # SQIL ignora la recompensa ambiental; se asigna 0 a las transiciones del agente
            agent.store_agent(state, action, next_state, float(done or truncated))
            state = next_state
            episode_reward += reward
            total_steps += 1
            if total_steps % update_every == 0:
                losses = agent.update()
                if losses is not None:
                    critic_loss, actor_loss, alpha_loss = losses
                    writer.add_scalar("Loss/Critic", critic_loss, total_steps)
                    writer.add_scalar("Loss/Actor", actor_loss, total_steps)
                    writer.add_scalar("Loss/Alpha", alpha_loss, total_steps)
            if done or truncated:
                break
        reward_history.append(episode_reward)
        writer.add_scalar("Train/EpisodeReward", episode_reward, episode)
        # Debug: cada 50 episodios, imprimir y loguear la recompensa media de esos episodios
        if episode % 50 == 0:
            avg_reward = np.mean(reward_history[-50:])
            print(f"Trial {trial.number} - Episodio {episode}: Recompensa media de los últimos 50 episodios = {avg_reward:.2f}")
            writer.add_text("Train/Debug", f"Episodio {episode}: Recompensa media últimos 50 = {avg_reward:.2f}", global_step=episode)
    
    # Guardar el modelo (actor) con un nombre que incluya los hiperparámetros
    model_name = f"sqil_actor_lr{actor_lr:.0e}_critic_lr{critic_lr:.0e}_alpha_lr{alpha_lr:.0e}_bs{batch_size}_upd{update_every}_ms{max_steps}.pth"
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    actor_save_path = os.path.join(model_dir, model_name)
    torch.save(agent.actor.state_dict(), actor_save_path)
    print(f"Trial {trial.number}: Modelo guardado en {actor_save_path}")
    writer.add_text("Model/Info", f"Modelo guardado en {actor_save_path}", global_step=TRAIN_EPISODES)
    
    # Evaluar el modelo entrenado en EVAL_EPISODES episodios
    eval_rewards = []
    for ep in range(1, EVAL_EPISODES + 1):
        state, _ = env.reset(seed=seed + 1000 + ep)
        ep_reward = 0
        while True:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            state = next_state
            if done or truncated:
                break
        eval_rewards.append(ep_reward)
        writer.add_scalar("Eval/EpisodeReward", ep_reward, ep)
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    print(f"Trial {trial.number}: Evaluación en {EVAL_EPISODES} episodios: Recompensa media = {mean_reward:.2f} (std: {std_reward:.2f})")
    writer.add_text("Eval/Info", f"Recompensa media: {mean_reward:.2f}, std: {std_reward:.2f}", global_step=TRAIN_EPISODES)
    
    env.close()
    writer.close()
    return mean_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for SQILAgent")
    parser.add_argument("--n_trials", type=int, default=20, help="Número de trials de Optuna")
    args = parser.parse_args()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)
    
    print("Mejor trial:")
    trial = study.best_trial
    print(f"Valor objetivo: {trial.value:.2f}")
    print("Mejores hiperparámetros:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
