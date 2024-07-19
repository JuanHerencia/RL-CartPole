# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 06:51:14 2024

@author: JHH2
"""
import warnings
warnings.filterwarnings("ignore")

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# Configuración del entorno y parámetros
env = gym.make('CartPole-v1')

# Modificar parámetros físicos
env.env.masscart = 1.0  # Peso del carrito (en kg)
env.env.masspole = 0.1  # Peso del poste (en kg)
env.env.gravity = 9.8   # Gravedad (en m/s^2)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Definir el rango de bins para la discretización
num_bins = 20
bins = [
    np.linspace(-4.8, 4.8, num_bins),  # cart position
    np.linspace(-4, 4, num_bins),      # cart velocity
    np.linspace(-0.418, 0.418, num_bins),  # pole angle
    np.linspace(-4, 4, num_bins)       # pole velocity
]

# Inicializar las tablas Q para Q-Learning y SARSA
q_table_q_learning = np.zeros([num_bins] * state_size + [action_size])
q_table_sarsa = np.zeros([num_bins] * state_size + [action_size])

# Hiperparámetros
learning_rate = 0.8
discount_rate = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 10000
max_steps = 5000

# Función para discretizar el espacio de estados
def discretize_state(state):
    indices = []
    for i in range(state_size):
        indices.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(indices)

# Función epsilon-greedy para seleccionar una acción
def epsilon_greedy_action(q_table, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

# Almacena las recompensas por episodio para graficar
rewards_q_learning = []
rewards_sarsa = []

print(f'Alpha = {learning_rate}\nGamma = {discount_rate}\nEpsilon = {epsilon}\nEpsilon Decay = {epsilon_decay}\nEpisodios = {episodes}\nPasos x episodio = {max_steps}')
print('Q-Learning')
inicio = time.time()
# Implementación de Q-Learning
for episode in range(episodes):
    state = discretize_state(env.reset()[0])
    total_reward = 0
    
    for step in range(max_steps):
        action = epsilon_greedy_action(q_table_q_learning, state, epsilon)
        
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)
        
        # Actualización de Q-Table para Q-Learning
        q_table_q_learning[state][action] = q_table_q_learning[state][action] + learning_rate * (
            reward + discount_rate * np.max(q_table_q_learning[next_state]) - q_table_q_learning[state][action])
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    # Reducción del valor de epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    rewards_q_learning.append(total_reward)
    
    if total_reward >= 500:
       print(f'Episodio {episode}, Recompensa total: {total_reward}')
    
    #if episode % 100 == 0:
    #    print(f'[Q-Learning] Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}')

fin = time.time()
tiempo_ql = fin - inicio

# Reset epsilon for SARSA
epsilon = 1.0

print('SARSA')
inicio = time.time()
# Implementación de SARSA
for episode in range(episodes):
    state = discretize_state(env.reset()[0])
    total_reward = 0
    action = epsilon_greedy_action(q_table_sarsa, state, epsilon)
    
    for step in range(max_steps):
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)
        next_action = epsilon_greedy_action(q_table_sarsa, next_state, epsilon)
        
        # Actualización de Q-Table para SARSA
        q_table_sarsa[state][action] = q_table_sarsa[state][action] + learning_rate * (
            reward + discount_rate * q_table_sarsa[next_state][next_action] - q_table_sarsa[state][action])
        
        state = next_state
        action = next_action
        total_reward += reward
        
        if done:
            break
    
    # Reducción del valor de epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    rewards_sarsa.append(total_reward)
    
    if total_reward >= 500:
       print(f'Episodio {episode}, Recompensa total: {total_reward}')
    
    #if episode % 100 == 0:
    #    print(f'[SARSA] Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}')
fin = time.time()        
tiempo_sarsa = fin - inicio

print(f'Tiempo Q-Learning = {tiempo_ql} segundos')
print(f'Tiempo SARSA      = {tiempo_sarsa} segundos')

# Gráficos de las recompensas por episodio para Q-Learning y SARSA
plt.figure(figsize=(18, 8))
# Mostrar al 75% de altura de la máx recompensa
plt.subplot(2, 1, 1)
plt.text(8000,0.70*max(max(rewards_sarsa),max(rewards_q_learning)), f'Alpha = {learning_rate}\nGamma = {discount_rate}\nEpsilon = {epsilon}\nEpsilon Decay = {epsilon_decay}\nEpisodios = {episodes}\nPasos x episodio = {max_steps}')
plt.plot(rewards_q_learning, label='Q-Learning')
plt.xlabel('Episodios')
plt.ylabel('Recompensa')
plt.title('Recompensa Total por episodio (Q-Learning)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(rewards_sarsa, label='SARSA', color = 'b')
plt.xlabel('Episodios')
plt.ylabel('Recompensa')
plt.title('Recompensa Total por episodio (SARSA)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Simulación con el agente entrenado para Q-Learning
initial_state = [0, 0, np.pi/4, 0]
env.env.state = initial_state
state = discretize_state(env.env.state)

positions_q_learning = []
angles_q_learning = []
prev_angle = initial_state[2]

for _ in range(max_steps):
    action = np.argmax(q_table_q_learning[state])
    next_state, reward, done, _, _ = env.step(action)
    state = discretize_state(next_state)
    
    positions_q_learning.append(next_state[0])
    angles_q_learning.append(next_state[2])
    
    if abs(next_state[2]) < np.pi / 20 or abs(next_state[2]) > np.pi / 2:
        break
    
    # Cambio de acción si el ángulo está aumentando en valor absoluto
    if abs(next_state[2]) > abs(prev_angle):
        action = 1 - action  # Cambiar acción (0 -> 1 o 1 -> 0)
    
    prev_angle = next_state[2]

# Sumarizar las recompensas
total_reward_q_learning = sum(rewards_q_learning)
total_reward_sarsa = sum(rewards_sarsa)

print(f'Sumarización de Recompensas:\nTotal Recompensa (Q-Learning): {total_reward_q_learning}\nTotal Recompensa (SARSA): {total_reward_sarsa}')

# Cerrar el entorno
env.close()
