# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 06:26:33 2024

@author: JHH2
"""
import warnings
warnings.filterwarnings("ignore")

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

# Configuración del entorno y parámetros de SARSA
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Definir el rango de bins para la discretización
num_bins = 10
bins = [
    np.linspace(-4.8, 4.8, num_bins),  # cart position
    np.linspace(-4, 4, num_bins),      # cart velocity
    np.linspace(-0.418, 0.418, num_bins),  # pole angle
    np.linspace(-4, 4, num_bins)       # pole velocity
]

# Inicializar la tabla Q con dimensiones adecuadas
q_table = np.zeros([num_bins] * state_size + [action_size])

# Hiperparámetros
learning_rate = 0.1
discount_rate = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 1000
max_steps = 5000

# Función para discretizar el espacio de estados
def discretize_state(state):
    indices = []
    for i in range(state_size):
        indices.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(indices)

# Almacena las recompensas por episodio para graficar
rewards_per_episode = []

# SARSA
for episode in range(episodes):
    state = discretize_state(env.reset()[0])
    total_reward = 0
    consecutive_steps = 0
    
    # Selección de la acción inicial
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state])
    
    for step in range(max_steps):
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)
        
        # Selección de la siguiente acción
        if random.uniform(0, 1) < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(q_table[next_state])
        
        # Actualización de Q-Table
        q_table[state][action] = q_table[state][action] + learning_rate * (
            reward + discount_rate * q_table[next_state][next_action] - q_table[state][action])
        
        state = next_state
        action = next_action
        total_reward += reward
        
        # Verificar si el poste se ha mantenido en posición por más de 500 pasos consecutivos
        if abs(state[2]) < np.pi / 20:
            consecutive_steps += 1
        else:
            consecutive_steps = 0
        
        if consecutive_steps >= 500:
            done = True
        
        if done:
            break

    if episode % 20 == 0:
        print(f'Episodio {episode}, Estado final: {state}, Recompensa total: {total_reward}')    
        
    # Reducción del valor de epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    rewards_per_episode.append(total_reward)
    
    #if episode % 100 == 0:
    #    print(f'Episodio: {episode}, Recompensa total: {total_reward}, Epsilon: {epsilon}')

# Graficar las recompensas por episodio
plt.figure(figsize=(18, 8))
plt.plot(rewards_per_episode)
plt.xlabel('Episodio')
plt.ylabel('Recompensa total')
plt.title('Recompensa total por episodio')
plt.grid(True)
plt.show()

# Simulación con el agente entrenado
initial_state = [0, 0, np.pi/4, 0]
env.env.state = initial_state
state = discretize_state(env.env.state)

positions = []
angles = []
prev_angle = initial_state[2]

for _ in range(max_steps):
    action = np.argmax(q_table[state])
    next_state, reward, done, _, _ = env.step(action)
    state = discretize_state(next_state)
    
    positions.append(next_state[0])
    angles.append(next_state[2])
    
    if abs(next_state[2]) < np.pi / 20 or abs(next_state[2]) > np.pi / 2:
        break
    
    # Cambio de acción si el ángulo está aumentando en valor absoluto
    if abs(next_state[2]) > abs(prev_angle):
        action = 1 - action  # Cambiar acción (0 -> 1 o 1 -> 0)
    
    prev_angle = next_state[2]

# Gráficos de resultados
time = np.linspace(0, len(positions)/50, len(positions))  # 50 steps por segundo

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, positions, label='Posición del carro')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición (m)')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, angles, label='Ángulo del poste')
plt.xlabel('Tiempo (s)')
plt.ylabel('Ángulo (rad)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Cerrar el entorno
env.close()
