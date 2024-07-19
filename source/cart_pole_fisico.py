# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 02:36:57 2024

@author: JHERENCIA
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time

# Parámetros del sistema
m_cart = 1.0  # masa del carro
m_pole = 0.3  # masa del poste
l = 0.5       # longitud del poste (desde el pivote hasta el centro de masa)
g = 9.81      # aceleración debido a la gravedad

# Fuerza inicial para comenzar el equilibrio
F_initial = 12.75 # 12.76 Una centécima menos pierde el equilibrio

# Estado inicial: [posición del carro, velocidad del carro, ángulo del poste, velocidad angular del poste]
initial_state = [0, 0, np.pi/4, 0]

# Definir las ecuaciones diferenciales del sistema
def cart_pole_dynamics(state, t, F):
    x, x_dot, theta, theta_dot = state
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    total_mass = m_cart + m_pole
    pole_mass_length = m_pole * l

    theta_ddot = (g * sin_theta + cos_theta * (-F - pole_mass_length * theta_dot**2 * sin_theta) / total_mass) / (l * (4/3 - m_pole * cos_theta**2 / total_mass))
    x_ddot = (F + pole_mass_length * (theta_dot**2 * sin_theta - theta_ddot * cos_theta)) / total_mass
    
    return [x_dot, x_ddot, theta_dot, theta_ddot]

# Tiempo de simulación
t = np.linspace(0, 20, 1000)
dt = t[1] - t[0]  # Paso de tiempo

# Solucionar las ecuaciones diferenciales
states = [initial_state]
F = F_initial

inicio = time.time()
for i in range(len(t) - 1):
    current_state = states[-1]
    next_state = odeint(cart_pole_dynamics, current_state, [t[i], t[i+1]], args=(F,))[1]
    states.append(next_state)
    
    if i % 50 == 0:
        print(f'iteración {i}-{next_state}')
    # Cambio de fuerza según el ángulo
    if abs(next_state[2]) > abs(current_state[2]):
        F = -F
fin_ = time.time()

states = np.array(states)

# Extraer las soluciones
x = states[:, 0]
theta = states[:, 2]

# Verificar condiciones de terminación
for i, angle in enumerate(theta):
    if abs(angle) < np.pi / 20 or abs(angle) > np.pi / 2:
        t = t[:i+1]
        x = x[:i+1]
        theta = theta[:i+1]
        print(f'Resultado esperado en Iteración {i}\nPosicion del carro = {round(x[-1],2)} metros\nAngulo de poste = {round(theta[-1],3)} radianes ')
        break

print(f'Tiempo de 1000 episodios      = {1000*(fin_ - inicio)} segundos')

# Gráficos de resultados
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, x, label='Posición del carro')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición (m)')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, theta, label='Ángulo del poste')
plt.xlabel('Tiempo (s)')
plt.ylabel('Ángulo (rad)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
