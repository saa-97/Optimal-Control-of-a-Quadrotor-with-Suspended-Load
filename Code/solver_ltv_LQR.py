'''
    This file contains the LQR solver for our system.
    The LQR solver is used in task 3.
'''

import numpy as np
import dynamics as dyn

ns = dyn.ns
ni = dyn.ni
TT = dyn.TT

def LQR(xx_ref,uu_ref,perturbation):

    # LQR parameters
    QQ_reg = 0.1*np.diag([25,25,1,1,1,1,1,1])
    QQ_reg_T = QQ_reg
    RR_reg = np.diag([1,1])

    AA = np.zeros((ns,ns,TT-1))
    BB = np.zeros((ns,ni,TT-1))

    PP = np.zeros((ns,ns,TT))
    KK = np.zeros((ni,ns,TT))

    xx = np.zeros((ns,TT))
    Fs = np.ones((1,TT))*(dyn.MM+dyn.mm)*dyn.gg
    Fd = np.zeros((1,TT))
    uu = np.vstack((Fs,Fd))

    for tt in range (TT-1):
        AA[:,:,tt] = dyn.dynamics(xx_ref[:,tt],uu_ref[:,tt])[1].T
        BB[:,:,tt] = dyn.dynamics(xx_ref[:,tt],uu_ref[:,tt])[2].T

    # Solving Riccatti equation
    for tt in reversed(range(TT-1)):
        QQt = QQ_reg
        RRt = RR_reg
        AAt = AA[:,:,tt]
        BBt = BB[:,:,tt]
        PPtp = PP[:,:,tt+1]

        PP[:,:,tt] = QQt + AAt.T@PPtp@AAt - (AAt.T@PPtp@BBt)@np.linalg.inv((RRt + BBt.T@PPtp@BBt))@(BBt.T@PPtp@AAt)

    for tt in range(TT-1):
        QQt = QQ_reg
        RRt = RR_reg
        AAt = AA[:,:,tt]
        BBt = BB[:,:,tt]
        PPtp = PP[:,:,tt+1]

        KK[:,:,tt] = -np.linalg.inv(RRt + BBt.T@PPtp@BBt)@(BBt.T@PPtp@AAt)

    # Applying the perturbation
    if perturbation == 'none':
        xx[:,0] = xx_ref[:,0]
    elif perturbation == 'small':
        xx[:,0] = xx_ref[:,0] + np.array((0.2,0.2,2.0,0.1,0.1,0.1,0.1,0.1))
    elif perturbation == 'large':
        xx[:,0] = xx_ref[:,0] + np.array((0.8,0.8,0.1,0.1,0.1,0.1,0.1,0.1))
 
    for tt in range(TT-1):
        uu[:,tt] = uu_ref[:,tt] + KK[:,:,tt]@(xx[:,tt] - xx_ref[:,tt])
        xx[:,tt+1] = dyn.dynamics(xx[:,tt],uu[:,tt])[0]

    return xx,uu  