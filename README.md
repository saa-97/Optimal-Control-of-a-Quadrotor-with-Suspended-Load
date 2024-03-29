# Optimal Control of a Quadrotor with Suspended Load

The project focuses on the development of an optimal trajectory for a planar quadrotor carrying a suspended load. Various approaches for solving the trajectory tracking problem are explored, beginning with the application of Newton’s method for optimal control to obtain an optimal state trajectory. Subsequently, the obtained trajectory is utilized in Linear Quadratic Regulator (LQR) and Model Predictive Control (MPC) techniques to generate optimal inputs for guiding the quadrotor along the predefined trajectory. A comparative analysis is conducted to assess and contrast the effectiveness of both LQR and MPC in terms of optimal input generation for the desired trajectory.

The following contains the details about the code and the instructions on how to run it.

Below is a description of all the python files:
1) main.py - This is the main file of the code. From here, different tasks can be run by setting them to true.
2) dynamics.py - This file contains the discretized dynamics and the gradients of the dynamics for the linearization matrices
3) desired_traj.py – This file is used to generate the desired step and spline trajectories between our equilibrium points
4) plot.py – This file is used to plot the results of the simulation
5) cost.py -  This file contains the stage cost and the terminal cost function. It also contains the weight matrices for task 1 and task2 as well as the gradients of the cost functions.
6) solver.py – This file is the main solver for the Newton’s method. 
7) solver_ltv_LQR.py – This file contains the LQR function that is used for task 3.
8) MPC.py – This file contains the code for the MPC for task 4.
9) animation.py - This file contains the animation code required to animate the results of task 3.

The code can be run from the main.py file by following these steps:
1) Select the tasks to be run from the “SELECTION OF TASKS” section. In order to run tasks 3, 4 and 5, make sure that task 2 is set to TRUE.
2) Selection of perturbation magnitude for task 3 and 4 can also be set from their respective code blocks. Choose between ‘none’, ‘small’ and ‘large’ perturbation magnitudes.
3) In order to enable the visualization of Armijo plots, set visu_armijo to TRUE in the solver.py file. Frequency of plots can be set by print_iter parameter. 
