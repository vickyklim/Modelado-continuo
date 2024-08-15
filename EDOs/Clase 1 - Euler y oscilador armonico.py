# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

##############################################################################
"""
def Euler(f,t0,tn,y0,h):
    N = int((tn-t0)/h)
    t = np.linspace(t0,tn,N+1)
    # t = np.arange(t0,tn,h)
    y = np.zeros(N+1)
    
    for i in range(1,N+1):
        y[i] = y[i-1]+h*f()
"""
    
##############################################################################
# Euler para una ecuación diferencial ordinaria

def euler(t0,tf,y0,h,f,l):
    t = np.arange(t0,tf+h,h)
    y = [y0]
    for i in range(len(t)-1):
        y.append(y[-1]+h*f(t[i],y[-1],l))
    return t,y

def f_exp(t,y,l):
    return l*y

"""
La solucion teorica de la ecuacion diferencial 
y' = l*y
es y = A*e^(l*t) con A dada por la condicion inicial y0.
"""

# Uso t0 = 0, tf = 2, y0 = 1, l = 2
sol_h1 = euler(0,2,1,0.1,f_exp,2)
sol_h001 = euler(0,2,1,0.001,f_exp,2)

plt.plot(sol_h1[0], sol_h1[1], c = 'r')
plt.plot(sol_h001[0], sol_h001[1], c = 'b')
# Ploteo y = y0 * e^(2*t)
plt.plot(sol_h001[0], np.exp(2*sol_h001[0]), c = 'k', linestyle = '--')
plt.show()

# Uso t0 = 0, tf = 2, y0 = 1, l = -10
sol_h1 = euler(0,2,1,0.1,f_exp,-10)
sol_h001 = euler(0,2,1,0.001,f_exp,-10)

plt.plot(sol_h1[0], sol_h1[1], c = 'r')
plt.plot(sol_h001[0], sol_h001[1], c = 'b')
# Ploteo y = y0 * e^(2*t)
plt.plot(sol_h001[0], np.exp(-10*sol_h001[0]), c = 'k', linestyle = '--')
plt.show()

##############################################################################
# Euler para el oscilador armonico

"""
La ecuacion diferencial es
y'' = -y
La convertimos en un sistema de ecuaciones de primer orden con el cambio de 
variables y' = u, y entonces y'' = u' = -y.
y' = u
u' = -y
Resolvemos con un metodo de Euler para las dos variables a la vez con dos
condiciones iniciales y0, u0.
y es la posicion del oscilador, u es la velocidad
"""

def euler_2(t0,tf,y0, u0,h,f1,f2):
    t = np.arange(t0,tf+h,h)
    y = [y0]
    u = [u0]
    for i in range(len(t)-1):
        y_aux = y[-1]
        y.append(y[-1]+h*f1(t[i],y[-1],u[-1]))
        u.append(u[-1]+h*f2(t[i],y_aux,u[-1]))
    return t,y,u

def f1(t,y,u):
    return u

def f2(t,y,u):
    return -y

def energia(y,u):
    e = []
    for i in range(len(y)):
        e.append((y[i]**2+u[i]**2)/2)
    return e

# sol_osc_h1 = (t,y,u)
sol_osc_h1 = euler_2(0,15,1,0,0.1,f1,f2)
sol_osc_h001 = euler_2(0,15,1,0,0.001,f1,f2)

plt.plot(sol_osc_h1[0], sol_osc_h1[1], c = 'b', label = 'Posicion')
plt.plot(sol_osc_h1[0], sol_osc_h1[2], c = 'cyan', label = 'Velocidad')
plt.xlabel('Tiempo')
plt.title('h = 0.1 - oscilador armónico')
plt.legend()
plt.show()

plt.plot(sol_osc_h001[0], sol_osc_h001[1], c = 'r', label = 'Posicion')
plt.plot(sol_osc_h001[0], sol_osc_h001[2], c = 'orangered', label = 'Velocidad')
plt.xlabel('Tiempo')
plt.title('h = 0.001 - oscilador armónico')
plt.legend()
plt.show()

plt.plot(sol_osc_h1[1], sol_osc_h1[2], c = 'b', label = 'h = 0.1')
plt.plot(sol_osc_h001[1], sol_osc_h001[2], c = 'r', label = 'h = 0.001')
plt.xlabel('Posición')
plt.ylabel('Velocidad')
plt.title('Diagrama de fases - oscilador armónico')
plt.legend()
plt.show()

plt.plot(sol_osc_h1[0], energia(sol_osc_h1[1],sol_osc_h1[2]), c = 'b', label = 'h = 0.1')
plt.plot(sol_osc_h001[0], energia(sol_osc_h001[1],sol_osc_h001[2]), c = 'r', label = 'h = 0.001')
plt.xlabel('Tiempo')
plt.ylabel('Energía total')
plt.title('Energía total - oscilador armónico')
plt.legend()
plt.show()

##############################################################################
# Oscilador amortiguado

"""
La ecuacion diferencial es
y'' = -y - gama*y'
La convertimos en un sistema de ecuaciones de primer orden con el cambio de 
variables y' = u, y entonces y'' = g - gama * u**2
y' = u
u' = -y - gama * u
Resolvemos con un metodo de Euler para las dos variables a la vez con dos
condiciones iniciales y0, u0.
y es la posicion del oscilador, u es la velocidad
"""

def f_amort(t,y,u,gama):
    return -y - gama * u

def euler_amort(t0,tf,y0,u0,h,f1,f2,gama):
    t = np.arange(t0,tf+h,h)
    y = [y0]
    u = [u0]
    for i in range(len(t)-1):
        y_aux = y[-1]
        y.append(y[-1]+h*f1(t[i],y[-1],u[-1]))
        u.append(u[-1]+h*f_amort(t[i],y_aux,u[-1],gama))
    return t,y,u

sol_osc_h001_gama01 = euler_amort(0,15,1,0,0.1,f1,f_amort,0.1)
sol_osc_h001_gama05 = euler_amort(0,15,1,0,0.1,f1,f_amort,0.5)
sol_osc_h001_gama1 = euler_amort(0,15,1,0,0.1,f1,f_amort,1)
sol_osc_h001_gama2 = euler_amort(0,15,1,0,0.1,f1,f_amort,2)
sol_osc_h001_gama10 = euler_amort(0,15,1,0,0.1,f1,f_amort,10)

plt.plot(sol_osc_h001_gama01[0], sol_osc_h001_gama01[1], c = 'b', label = 'Gamma = 0.1')
plt.plot(sol_osc_h001_gama05[0], sol_osc_h001_gama05[1], c = 'cyan', label = 'Gamma = 0.5')
plt.plot(sol_osc_h001_gama1[0], sol_osc_h001_gama1[1], c = 'g', label = 'Gamma = 1')
plt.plot(sol_osc_h001_gama2[0], sol_osc_h001_gama2[1], c = 'r', label = 'Gamma = 2')
plt.plot(sol_osc_h001_gama10[0], sol_osc_h001_gama10[1], c = 'gold', label = 'Gamma = 10')
plt.legend()
plt.title('Oscilador armónico amortiguado')
plt.xlabel('Tiempo')
plt.ylabel('Posición')
plt.show()

plt.plot(sol_osc_h001_gama01[0], energia(sol_osc_h001_gama01[1],
                                         sol_osc_h001_gama01[2]), 
         c = 'b', label = 'Gamma = 0.1')
plt.plot(sol_osc_h001_gama05[0], energia(sol_osc_h001_gama05[1],
                                         sol_osc_h001_gama05[2]), 
         c = 'cyan', label = 'Gamma = 0.5')
plt.plot(sol_osc_h001_gama1[0], energia(sol_osc_h001_gama1[1],
                                         sol_osc_h001_gama1[2]), 
         c = 'g', label = 'Gamma = 1')
plt.plot(sol_osc_h001_gama2[0], energia(sol_osc_h001_gama2[1],
                                         sol_osc_h001_gama2[2]), 
         c = 'r', label = 'Gamma = 2')
plt.plot(sol_osc_h001_gama10[0], energia(sol_osc_h001_gama10[1],
                                         sol_osc_h001_gama10[2]), 
         c = 'gold', label = 'Gamma = 10')
plt.xlabel('Tiempo')
plt.ylabel('Energía total')
plt.title('Energía total - oscilador armónico amortiguado')
plt.legend()
plt.show()