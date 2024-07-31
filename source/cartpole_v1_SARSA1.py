# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 14:24:13 2024

@author: JHH2
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# Configuraciones del entorno y del algoritmo
env = gym.make('CartPole-v1')
n_actions = env.action_space.n
n_bins = (10, 10, 10, 10)  # Número N limites o valores discretos para cada dimensión (car_position, car_velocity, pole_angle, pole_velocity)
alpha = 0.1  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento
epsilon = 1.0  # Tasa de exploración inicial
epsilon_decay = 0.995  # Decaimiento
epsilon_min = 0.01  # Mínimo valor de epsilon
n_episodes = 100000  # Número de episodios
goal_avg_reward = 500  # Recompensa media objetivo para considerar el problema resuelto
max_steps = 10000  # Cantidad máxima de pasos por episodio
window_size = 100  # Ventana para calcular la media móvil

# Límites para la discretización
state_bins = [
    np.linspace(-4.8, 4.8, n_bins[0] - 1),  # car_position
    np.linspace(-3.0, 3.0, n_bins[1] - 1),  # car_velocity
    np.linspace(-0.418, 0.418, n_bins[2] - 1),  # pole_angle (~24° en radianes)
    np.linspace(-3.0, 3.0, n_bins[3] - 1)   # pole_velocity
]

# Función para discretizar los estados
def discretize_state(state):
    car_position, car_velocity, pole_angle, pole_velocity = state
    state_discrete = [
        np.digitize(car_position, state_bins[0]),
        np.digitize(car_velocity, state_bins[1]),
        np.digitize(pole_angle, state_bins[2]),
        np.digitize(pole_velocity, state_bins[3])
    ]
    return tuple(state_discrete)

# Inicialización de la tabla Q
q_table = np.zeros(n_bins + (n_actions,))

# Función para elegir una acción basado en epsilon-greedy
def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    return np.argmax(q_table[state])

file_csv = open('salida_sarsa.csv', 'w')
file_csv.writelines(f'Episodio,RecompensaxTotalxEpisodio,RecompensaPromedio100Epidosios\n')
print('Resultados para SARSA')
print(f'Alpha = {alpha}\nGamma = {gamma}\nEpsilon = {epsilon}\nEpsilon Decay = {epsilon_decay}\nEpisodios = {n_episodes}\nPasos x episodio = {max_steps}')

# Entrenamiento con SARSA
rewards_per_episode = []
mean_durations = []
mean_duration = 0
t_ini = time.time()

for episode in range(n_episodes):
    state, _ = env.reset()
    state = discretize_state(state)
    action = choose_action(state)
    total_reward = 0
    
    for t in range(max_steps):
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize_state(next_state)
        next_action = choose_action(next_state)
        
        # Actualización de la tabla Q con SARSA
        q_target = reward + gamma * q_table[next_state][next_action]
        q_table[state][action] += alpha * (q_target - q_table[state][action])
        
        state = next_state
        action = next_action
        total_reward += reward
        
        if terminated or truncated:
            break
    
    rewards_per_episode.append(total_reward)
    
    # Calcular la duración media de los últimos 100 episodios
    if episode >= window_size:
        mean_duration = np.mean(rewards_per_episode[-window_size:])
        mean_durations.append(mean_duration)
    else:
        mean_durations.append(np.mean(rewards_per_episode))
    
    # Decaimiento de epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    # Verificar si se ha resuelto el problema
    if episode >= window_size and mean_duration >= goal_avg_reward:
        print(f'Problema resuelto en {episode + 1} episodios con una duración media de {mean_duration:.2f}')
        file_csv.writelines(f'{episode + 1},{total_reward},{mean_duration:.2f}\n')
        break
    
    file_csv.writelines(f'{episode + 1},{total_reward},{mean_duration:.2f}\n')

t_fin = time.time()    
file_csv.close()

print('Resultados para SARSA')
print(f'Alpha = {alpha}\nGamma = {gamma}\nEpsilon = {epsilon}\nEpsilon Decay = {epsilon_decay}\nEpisodios = {n_episodes}\nPasos x episodio = {max_steps}')
print(f'Tiempo realizado : {t_fin - t_ini} segundos')

# Graficar las recompensas obtenidas por episodio
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(rewards_per_episode)
plt.grid(True)
plt.xlabel('Episodio')
plt.ylabel('Recompensa Total')
plt.title('Recompensas por Episodio')

# Graficar la duración media de los últimos 100 episodios
plt.subplot(1, 2, 2)
plt.plot(mean_durations)
plt.xlabel('Episodio')
plt.ylabel('Duración Media (100 episodios)')
plt.title('Duración Media en 100 Episodios Consecutivos')
plt.grid(True)
plt.tight_layout()
plt.show()