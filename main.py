'''
    > OPTIMAL CONTROL OF A QUADROTOR WITH SUSPENDED LOAD
    > Authors: Nadia, Heba Al Abed, Sharjeel Ashraf Ansari
    > Course: Optimal Control

'''
import numpy as np
import matplotlib.pyplot as plt
import dynamics as dyn
import cost as cst
import solver
import desired_traj as desired_traj
import plot as plot
import solver_ltv_LQR as LQR
import MPC
import animation
from termcolor import colored

################################################
# SELECTION OF TASKS AND SETTING PERTURBATION
################################################

task1 = False
task2 = True
task3 = True            # Task 2 must be true to run task 3
task4 = True           # Task 2 must be true to run task 4
task5 = True            # Task 2 and 3 must be true to run task 5

perturbation = 'none' # Choose between 'none', 'small', 'large'

################################################
# INITIALIZE THE VARIABLES
################################################

TT = dyn.TT
ns = dyn.ns
ni = dyn.ni
max_iters = int(3e1)
tf = dyn.tf

x_eq1 = np.array([0,0,0,0,0,0,0,0])             # Selection of the two equilibrium points for the trajectory
x_eq2 = np.array([2,2,0,0,0,0,0,0])

print("\n")
print(colored("OPTIMAL CONTROL OF A QUADROTOR WITH SUSPENDED LOAD", "red", attrs=["bold", "underline"]).center(80))

################################################
# TASK 1 - STEP TRAJECTORY TRACKING
################################################

if task1 == True:
    task = 1
    print("\n")
    print(colored("STARTING TASK 1 - STEP TRAJECTORY TRACKING", "blue", attrs=["bold", "underline"]).center(80))
    print("\n","- Initializing the step trajectory - ","\n")

    xx_ref = desired_traj.desired_traj("step",TT,x_eq1,x_eq2)       # defining a step trajectory between the two equilibrium points

    f_s = np.ones((1,TT))*(dyn.MM + dyn.mm) * dyn.gg
    f_i = np.zeros((1,TT))
    uu_ref = np.vstack([f_s, f_i])                                  

    xx_init = np.zeros((ns,TT))
    init_inp = uu_ref

    x0 = xx_ref[:,0]

    xx = np.zeros((ns,TT,max_iters))
    uu = np.zeros((ni,TT,max_iters+1))

    # for tt in range(TT):                        
    #     xx[:,tt,0] = xx_ref[:,-1]
    #     uu[:,tt,0] = uu_ref[:,0]
    # xx[0,:,0] = np.ones((1,TT))
    # xx[1,:,0] = np.ones((1,TT))

    uu[:,:,0] = np.vstack([f_s, f_i])
    for tt in range(TT-1):               
        xx[:,tt+1,0] = dyn.dynamics(xx[:,tt,0],uu[:,tt,0])[0]

    xx,uu,descent,JJ,kk = solver.Newton(xx,uu,xx_ref,uu_ref,x0,max_iters,task)

    xx_star = xx[:,:,kk]
    uu_star = uu[:,:,kk]
    uu_star[:,-1] = uu_star[:,-2] # for plotting purposes

    plot.descent(kk,descent)
    plot.cost(kk,JJ)
    # plot.report_states(xx_star,xx_ref,uu_star,np.linspace(0,tf,TT),task)            # for report plots
    plot.plot_states_and_inputs(xx_star, xx_ref, uu_star, np.linspace(0,tf,TT),task)

################################################
# TASK 2 - SMOOTH TRAJECTORY TRACKING
################################################
    
if task2 == True:
    task = 2 
    print("\n")
    print(colored("STARTING TASK 2 - SMOOTH TRAJECTORY TRACKING", "blue", attrs=["bold", "underline"]).center(80))
    print("\n","- Initializing the smooth trajectory - ","\n")

    xx_ref = desired_traj.desired_traj("spline",TT,x_eq1,x_eq2)             # defining a smooth trajectory between the two equilibrium points

    f_s = np.ones((1,TT))*(dyn.MM + dyn.mm) * dyn.gg
    f_i = np.zeros((1,TT))
    uu_ref = np.vstack([f_s, f_i])

    xx_init = np.zeros((ns,TT))
    init_inp = uu_ref

    x0 = xx_ref[:,0]

    xx = np.zeros((ns,TT,max_iters))
    uu = np.zeros((ni,TT,max_iters+1))

    # for tt in range(TT):
    #     xx[:,tt,0] = np.copy(xx_ref[:,-1])
    #     uu[:,tt,0] = np.copy(uu_ref[:,0])
    # xx[0,:,0] = np.ones((1,TT))
    # xx[1,:,0] = np.ones((1,TT))

    uu[:,:,0] = np.vstack([f_s, f_i])
    for tt in range(TT-1):               
        xx[:,tt+1,0] = dyn.dynamics(xx[:,tt,0],uu[:,tt,0])[0]
    
    xx,uu,descent,JJ,kk = solver.Newton(xx,uu,xx_ref,uu_ref,x0,max_iters,task)

    xx_star = xx[:,:,kk]
    uu_star = uu[:,:,kk]
    uu_star[:,-1] = uu_star[:,-2] # for plotting purposes
    
    plot.descent(kk,descent)
    plot.cost(kk,JJ)
    # plot.report_states(xx_star,xx_ref,uu_star,np.linspace(0,tf,TT),task)            # for report plots
    plot.plot_states_and_inputs(xx_star, xx_ref, uu_star, np.linspace(0,tf,TT),task)


################################################
# TASK 3 - TRAJECTORY TRACKING via LQR
################################################
    
if task3 == True & task2 == True:
    task = 3
    print("\n")
    print(colored("STARTING TASK 3 - TRAJECTORY TRACKING via LQR", "blue", attrs=["bold", "underline"]).center(80))
    print("\n","- Initializing the reference trajectory for task 3 - ","\n")

    plot.reference_traj(xx_star,uu_star,np.linspace(0,tf,TT),task)

    print("Perturbation has been set to: ", perturbation,"\n")              # Perturbation can be adjusted at the beginning of the code

    xx_reg,uu_reg = LQR.LQR(xx_star,uu_star,perturbation)

    plot.plot_states_and_inputs(xx_reg,xx_star,uu_reg,np.linspace(0,tf,TT),task)
    plot.state_errors(xx_reg,xx_star,np.linspace(0,tf,TT),task)
    # plot.report_states(xx_reg,xx_star,uu_reg,np.linspace(0,tf,TT),task)            # for report plots
    # plot.report_error_plots(xx_reg,xx_star,np.linspace(0,tf,TT),task)


#########################################s#######
# TASK 4 - TRAJECTORY TRACKING via MPC
################################################

if task4 == True & task2 == True:
    task = 4 
    print("\n")
    print(colored("STARTING TASK 4 - TRAJECTORY TRACKING via MPC", "blue", attrs=["bold", "underline"]).center(80))
    print("\n","- Initializing the reference trajectory for task 4 - ","\n")

    plot.reference_traj(xx_star,uu_star,np.linspace(0,tf,TT),task)

    Tsim = TT
    T_pred = 100

    AA = np.zeros((ns,ns,TT-1))
    BB = np.zeros((ns,ni,TT-1))

    QQ = 0.1*np.diag([25,25,1,1,0.1,0.1,0.1,0.1])
    RR = np.diag([0.01,0.1])
    QQf = QQ*100

    for tt in range (TT-1):
        fx,fu = dyn.dynamics(xx_star[:,tt],uu_star[:,tt])[1:]

        AA[:,:,tt] = fx.T
        BB[:,:,tt] = fu.T
    
    xx_real_mpc = np.ones((ns,TT))
    uu_real_mpc = np.ones((ni,TT))

    xx0 = xx_star[:,0]
    if perturbation == 'none':
        xx0 = xx_star[:,0]
    elif perturbation == 'small':
        xx0 = xx_star[:,0] + np.array((0.2,0.2,0.1,0.1,0.1,0.1,0.1,0.1))
    elif perturbation == 'large':
        xx0 = xx_star[:,0] + np.array((0.8,0.8,0.1,0.1,0.1,0.1,0.1,0.1))

    print ("Perturbation has been set to: ", perturbation,"\n")

    xx_real_mpc[:,0] = xx0.squeeze()

    for tt in range(TT-1):
        xx_t_mpc = xx_real_mpc[:,tt]                    # set initial condition for the MPC

        if tt%5 == 0:                                   # print every 5 time instants
            print('MPC:\t t = {}'.format(tt))

        uu_real_mpc[:,tt] = MPC.linear_mpc(AA,BB,QQ,RR,QQf,xx_t_mpc,T_pred,xx_star[:,tt:],uu_star[:,tt:],tt)

        xx_real_mpc[:,tt+1] = dyn.dynamics(xx_real_mpc[:,tt],uu_real_mpc[:,tt])[0]

    plot.plot_states_and_inputs(xx_real_mpc,xx_star,uu_real_mpc,np.linspace(0,tf,TT),task)
    plot.state_errors(xx_real_mpc,xx_star,np.linspace(0,tf,TT),task)
    # plot.report_states(xx_real_mpc,xx_star,uu_real_mpc,np.linspace(0,tf,TT),task)            # for report plots
    # plot.report_error_plots(xx_real_mpc,xx_star,np.linspace(0,tf,TT),task)

################################################
# TASK 5 - ANIMATION
################################################
    
if task5 == True and task3 == True:

    print("\n")
    print(colored("STARTING TASK 5 - ANIMATION", "blue", attrs=["bold", "underline"]).center(80))

    animation.animate(xx_reg, xx_ref)