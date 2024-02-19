'''
    This file contains functions to plot the results of the simulation.
'''

import matplotlib.pyplot as plt
import dynamics as dyn
import numpy as np
import solver as newton

ns = dyn.ns
ni = dyn.ni
term_cond = newton.term_cond

def plot_states_and_inputs(x1, x_des,u_plot, tt_hor,task):
    fig, axs = plt.subplots(5, 2, sharex='all', figsize=(16, 10))  

    if task == 1:
        plt.suptitle('Tracking performance of task 1', fontsize=16)  
    elif task == 2:
        plt.suptitle('Tracking performance of task 2', fontsize=16)
    elif task == 3:
        plt.suptitle('Tracking performance of task 3', fontsize=16)
    elif task == 4:
        plt.suptitle('Tracking performance of task 4', fontsize=16)

    state_labels = ['$x_p$', '$y_p$', '$alpha$', '$theta$', '$v_x$', '$v_y$', '$omega_alpha$', '$omega_theta$']
    input_labels = ['$F_s$', '$F_d$']

    for i in range(8):
        axs[i//2, i%2].plot(tt_hor, x1[i,:], linewidth=2, label='Actual')
        axs[i//2, i%2].plot(tt_hor, x_des[i,:], linewidth=2, linestyle='--', color='r', label='Reference')
        axs[i//2, i%2].grid()
        axs[i//2, i%2].set_ylabel(state_labels[i])

    for i in range(2):
        axs[4, i].plot(tt_hor, u_plot[i,:], linewidth=2)
        axs[4, i].grid()
        axs[4, i].set_ylabel(input_labels[i])

    axs[0, 0].legend()

    plt.tight_layout()
    plt.show()

def reference_traj(xx_star,uu_star,tt_hor,task):
    
    fig, axs = plt.subplots(5, 2, sharex='all', figsize=(15, 10))  
    if task == 3:
        plt.suptitle('Reference trajectory for task 3 - LQR', fontsize=16)  
    elif task == 4:
        plt.suptitle('Reference trajectory for task 4 - MPC', fontsize=16)

    state_labels = ['$x_p$', '$y_p$', '$alpha$', '$theta$', '$v_x$', '$v_y$', '$omega_alpha$', '$omega_theta$']
    input_labels = ['$F_s$', '$F_d$']

    for i in range(8):
        axs[i//2, i%2].plot(tt_hor, xx_star[i,:], linewidth=2, label='Actual')
        # axs[i//2, i%2].plot(tt_hor, x_des[i,:], linewidth=2, linestyle='--', color='r', label='Reference')
        axs[i//2, i%2].grid()
        axs[i//2, i%2].set_ylabel(state_labels[i])

    for i in range(2):
        axs[4, i].plot(tt_hor, uu_star[i,:], linewidth=2)
        axs[4, i].grid()
        axs[4, i].set_ylabel(input_labels[i])

    axs[0, 0].legend()  

    plt.tight_layout()
    plt.show()

def state_errors(xx,xx_ref,tt_hor,task):

    state_err = (xx - xx_ref)

    labels = ['$\u0394 x_{p}$', '$\u0394 y_{p}$', '$\u0394 alpha$', '$\u0394 theta$', 
            '$\u0394 v_{x}$', '$\u0394 v_{y}$', '$\u0394 w_{alpha}$', '$\u0394 w_{theta}$']

    fig, axs = plt.subplots(4, 2, figsize=(16, 20))  

    for i in range(8):
        row = i // 2  
        col = i % 2 

        axs[row, col].plot(tt_hor, state_err[i,:], linewidth=2)
        axs[row, col].grid()
        axs[row, col].set_ylabel(labels[i])

    plt.suptitle("State error for task " + str(task), fontsize=16) 
    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    plt.show()

def cost(kk,JJ):
    plt.figure('Cost')
    plt.rcParams.update({'font.size': 16})
    plt.suptitle('Cost')
    plt.plot(np.arange(kk), JJ[:kk])
    plt.xlabel('$k$')
    plt.ylabel('$J(\\mathbf{u}^k)$')
    plt.yscale('log')
    plt.grid()
    plt.show(block=False)

def descent(kk,descent):
    # kk=len(descent)
    plt.figure('Descent direction')
    plt.rcParams.update({'font.size': 16})
    plt.suptitle('Descent direction')
    plt.plot(np.arange(kk+1), descent[:kk+1])
    plt.axhline(y=term_cond, color='r', linestyle='--', label='Termination Condition')
    plt.xlabel('$k$')
    plt.ylabel('||$\\nabla J(\\mathbf{u}^k)||$')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show(block=False)

def report_states(x1,x_des,u_plot,tt_hor,task):
    state_labels = ['$x_{p}$', '$y_{p}$', '$alpha$', '$theta$', '$v_{x}$', '$v_{y}$', '$omega_{alpha}$', '$omega_{theta}$']
    input_labels = ['$F_s$', '$F_d$']

    for i in range(0, 8, 2):  
        fig, axs = plt.subplots(2, 1, figsize=(16, 10))  
        plt.suptitle("Plot of states " + state_labels[i] + " and " + state_labels[i+1] + " for task " + str(task), fontsize=16)  # Add title to the plot
        axs[0].plot(tt_hor, x1[i,:], linewidth=2, label='Actual')
        axs[0].plot(tt_hor, x_des[i,:], linewidth=2, linestyle='--', color='r', label='Reference')
        axs[0].grid()
        axs[0].set_ylabel(state_labels[i])
        axs[0].legend()

        axs[1].plot(tt_hor, x1[i+1,:], linewidth=2, label='Actual')
        axs[1].plot(tt_hor, x_des[i+1,:], linewidth=2, linestyle='--', color='r', label='Reference')
        axs[1].grid()
        axs[1].set_ylabel(state_labels[i+1])
        axs[1].legend()

    fig, axs = plt.subplots(2, 1, figsize=(16, 10))  
    for i in range(2):
        plt.suptitle("Plot of inputs " + input_labels[i-1] + " and " + input_labels[i] + " for task " + str(task), fontsize=16)  # Add title to the plot
        axs[i].plot(tt_hor, u_plot[i,:], linewidth=2)
        axs[i].grid()
        axs[i].set_ylabel(input_labels[i])

    plt.show()

def report_error_plots(xx,xx_ref,tt_hor,task):

    state_err = (xx - xx_ref)

    labels = ['$\u0394 x_{p}$', '$\u0394 y_{p}$', '$\u0394 alpha$', '$\u0394 theta$', 
            '$\u0394 v_{x}$', '$\u0394 v_{y}$', '$\u0394 w_{alpha}$', '$\u0394 w_{theta}$']

    for i in range(0,8,2):
        fig,axs = plt.subplots(2,1,figsize=(16,10))
        plt.suptitle("State error of " + labels[i] + " and " + labels[i+1] + " for task " + str(task), fontsize=16) 
        axs[0].plot(tt_hor, state_err[i,:], linewidth=2)
        axs[0].grid()
        axs[0].set_ylabel(labels[i])

        axs[1].plot(tt_hor, state_err[i+1,:], linewidth=2)
        axs[1].grid()
        axs[1].set_ylabel(labels[i+1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    plt.show()