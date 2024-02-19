'''
This file contains the desired trajectories for the quadrotor to follow.
We have defined a step trajectory, and a spline trajectory.
'''

import numpy as np
import matplotlib.pyplot as plt
import dynamics as dyn
from scipy.interpolate import CubicSpline

tf = dyn.tf

def desired_traj(reference,TT,x_eq1,x_eq2):
    x_trial = np.zeros((dyn.ns,TT))
    if reference == "step":
        for tt in range(TT-1):
            if tt<TT/2:
                x_trial[0,tt+1] = x_eq1[0]
                x_trial[1,tt+1] = x_eq1[1]
                x_trial[2,tt+1] = x_eq1[2]
                x_trial[3,tt+1] = x_eq1[3]

                x_trial[4,tt+1] = x_eq1[4]
                x_trial[5,tt+1] = x_eq1[5]
                x_trial[6,tt+1] = x_eq1[6]
                x_trial[7,tt+1] = x_eq1[7]

            else:
                x_trial[0,tt+1] = x_eq2[0]
                x_trial[1,tt+1] = x_eq2[1]
                x_trial[2,tt+1] = x_eq2[2]
                x_trial[3,tt+1] = x_eq2[3]

                x_trial[4,tt+1] = x_eq2[4]
                x_trial[5,tt+1] = x_eq2[5]
                x_trial[6,tt+1] = x_eq2[6]
                x_trial[7,tt+1] = x_eq2[7]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        plt.suptitle('Reference step trajectory - Task 1',fontsize=16)

        ax1.plot(np.linspace(0, tf, TT), x_trial[0, :], 'r--')
        ax1.set_ylabel('X position', fontsize=14)  

        ax2.plot(np.linspace(0, tf, TT), x_trial[1, :], 'r--')
        ax2.set_ylabel('Y position', fontsize=14)

        plt.xlabel('Time')
        plt.show()
    
    if reference == "spline":
        x0 = x_eq1[0]
        xf = x_eq2[0]

        y0 = x_eq1[1]
        yf = x_eq2[1]
        
        t = np.linspace(0,tf,TT)        

        x_control = np.array([x0,x0,x0,x0,x0,x0,x0+(xf-x0)*0.2,x0+(xf-x0)*0.8,xf+(xf-x0)*0.0,xf,xf,xf,xf,xf,xf,xf])
        y_control = np.array([y0,y0,y0,y0,y0,y0,y0+(yf-y0)*0.2,y0+(yf-y0)*0.8,yf+(yf-y0)*0.0,yf,yf,yf,yf,yf,yf,yf])
        t_control = np.linspace(0, tf, len(x_control))

        x_spline = CubicSpline(t_control, x_control, bc_type='natural')
        y_spline = CubicSpline(t_control, y_control, bc_type='natural')

        x_trial[0,:] = x_spline(t)
        x_trial[1,:] = y_spline(t)
        
        plt.figure('Reference Trajectory Plot')
        plt.plot(t, x_trial[0, :], 'blue', label='spline')
        plt.xlabel('Time',fontsize=14)
        plt.ylabel('$(x,y)$ position',fontsize=14)
        plt.plot(t_control, x_control, 'o',color='red', label='control points')
        plt.legend(loc='best',fontsize=14)
        plt.suptitle('Reference spline trajectory - Task 2',fontsize=16)
        plt.show()

    if reference == "loop":

        radius = 2  
        center_x = 0  
        center_y = 0  
        for tt in range(TT-1):
            if tt < 0.2*TT:
                x_trial[0,tt] = 0
                x_trial[1,tt] = 0
                x_trial[3,tt] = 0 

            if tt > 0.2*TT and tt < 0.7*TT:
                alpha = 0.01 * (tt)
                x_trial[0,tt] = radius * np.sin(alpha)
                x_trial[1,tt] = radius * np.cos(alpha) + 2*radius               

            else:
                x_trial[0,tt] = x_trial[0,tt - 1]
                x_trial[1,tt] = x_trial[1,tt - 1]

            x_trial[3,tt] = (2*np.pi/TT)*tt
        
        plt.figure('Reference Trajectory Plot')
        plt.plot(x_trial[0,:],x_trial[1,:]) 
        plt.show()  

    return x_trial
