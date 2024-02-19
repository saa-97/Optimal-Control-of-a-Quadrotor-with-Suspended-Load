'''
    This file contains the code for MPC required to run task 4.
'''

import cvxpy as cp
import dynamics as dyn
TT = dyn.TT

def linear_mpc(AA, BB, QQ, RR, QQf, xxt, T_pred, xx_star, uu_star,tt):
    """
        Linear MPC solver - Constrained LQR

        Given a measured state xxt measured at t
        gives back the optimal input to be applied at t

        Args
          - AA, BB: linear dynamics
          - QQ,RR,QQf: cost matrices
          - xxt: initial condition (at time t)
          - T: time (prediction) horizon

        Returns
          - u_t: input to be applied at t

    """

    xxt = xxt.squeeze()
    
    ns = dyn.ns
    ni = dyn.ni

    xx_mpc = cp.Variable((ns, T_pred))
    uu_mpc = cp.Variable((ni, T_pred))
    cost = 0
    constr = []
    pred_time = min(T_pred-1,TT-tt-1)

    for t in range(pred_time):
        AAt = AA[:, :, t+tt]
        BBt = BB[:, :, t+tt]
        stateerror = xx_mpc[:,t] - xx_star[:,t]
        inputerror = uu_mpc[:,t] - uu_star[:,t]
        cost += cp.quad_form(stateerror, QQ) + cp.quad_form(inputerror, RR)
        constr += [                                                                     
            xx_mpc[:,t+1] - xx_star[:,t+1] == AAt@stateerror + BBt@inputerror,          # Dynamic constraints
            uu_mpc[:,t] <= 3,                                                           # Input constraints       
            uu_mpc[:,t] >= -3,
            xx_mpc[:,pred_time] == xx_star[:,pred_time],                                # Terminal constraint i.e. state at the end of the horizon is equal to the reference state
        ]

    # sums problem objectives and concatenates constraints.
    cost += cp.quad_form(xx_mpc[:,-1] - xx_star[:,-1], QQf)
    constr += [xx_mpc[:,0] == xxt]

    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve()

    if problem.status == "infeasible":
    # Otherwise, problem.value is inf or -inf, respectively.
        print("Infeasible problem! CHECK YOUR CONSTRAINTS!!!")

    return uu_mpc[:,0].value