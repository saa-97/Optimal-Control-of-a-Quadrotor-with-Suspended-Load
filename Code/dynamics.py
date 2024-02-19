'''
    This file contains the dynamics of the quadrotor and the gradient of the dynamics.
    The dynamics are discretized using Forward Euler discretization.
    
'''


import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

ns = 8 # states
ni = 2 # inputs

dt = 1e-2 # discretization stepsize - Forward Euler
tf = 15 
TT = int((tf/dt)) # number of time steps

MM = 0.028
mm = 0.04
JJ = 0.001
gg = 9.81
LL = 0.2
ll = 0.05

KKeq = (MM + mm) * gg

x_dot = np.zeros(ns) 

def dynamics(xx,uu):

    xp,yp,alpha,theta,vx,vy,w_alpha,w_theta = xx            # Assigning variables to states
    Fs,Fd = uu                                              # Assigning variables to inputs

    # Discretized dynamics

    xpp = xp + dt * vx
    ypp = yp + dt * vy
    alphapp = alpha + dt * w_alpha
    thetapp = theta + dt * w_theta

    vxpp = vx + dt * (mm*LL*w_alpha**2*np.sin(alpha)-Fs*(np.sin(theta)-(mm/MM)*np.sin(alpha-theta)*np.cos(alpha)))/(MM+mm)
    vypp = vy + dt * (-mm*LL*w_alpha**2*np.cos(alpha) + Fs*(np.cos(theta)+(mm/MM)*(np.sin(alpha-theta)*np.sin(alpha))) - (MM+mm)*gg)/(MM+mm)
    w_alphapp = w_alpha + dt*(-Fs*np.sin(alpha-theta))/(MM*LL)
    w_thetapp = w_theta + dt*(ll*Fd)/JJ

    xxp = np.array([xpp,ypp,alphapp,thetapp,vxpp,vypp,w_alphapp,w_thetapp])         # The discretized states are stored in xxp

    # Gradient of the dynamics

    fx = np.array([
        [1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,0,1,0,dt*(mm*LL*w_alpha**2*np.cos(alpha) - Fs*(-np.cos(alpha-theta)*np.cos(alpha)+np.sin(alpha-theta)*np.sin(alpha))*(mm/MM))/(MM+mm), dt*(mm*LL*w_alpha**2*np.sin(alpha) + Fs*(np.cos(alpha-theta)*np.sin(alpha)+np.sin(alpha-theta)*np.cos(alpha))*(mm/MM))/(mm+MM),dt*(-Fs*np.cos(alpha-theta))/(MM*LL),0],
        [0,0,0,1,dt*(-Fs*(np.cos(theta)+(mm/MM)*np.cos(alpha-theta)*np.cos(alpha)))/(mm+MM),dt*(-Fs*(np.sin(theta)+(mm/MM)*np.cos(alpha-theta)*np.sin(alpha)))/(mm+MM),dt*(Fs*np.cos(alpha-theta))/(MM*LL),0],
        [dt,0,0,0,1,0,0,0],
        [0,dt,0,0,0,1,0,0],
        [0,0,dt,0,dt*(2*mm*LL*w_alpha*np.sin(alpha))/(mm+MM),dt*(-2*mm*LL*w_alpha*np.cos(alpha))/(mm+MM),1,0],
        [0,0,0,dt,0,0,0,1]
    ])

    fu = np.array([
        [0,0,0,0,dt*(-np.sin(theta)+(mm/MM)*np.sin(alpha-theta)*np.cos(alpha))/(mm+MM),dt*(np.cos(theta)+(mm/MM)*np.sin(alpha-theta)*np.sin(alpha))/(MM+mm),dt*(-np.sin(alpha-theta))/(MM*LL),0],
        [0,0,0,0,0,0,0,dt*ll/JJ]
    ])

    return xxp, fx, fu

    