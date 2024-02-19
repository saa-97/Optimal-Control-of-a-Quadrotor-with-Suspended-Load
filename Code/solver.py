'''
    This file contains the Newton method and the LQR function.
'''


import numpy as np
import cost as cst
import dynamics as dyn

import matplotlib.pyplot as plt

ns = dyn.ns
ni = dyn.ni

TT = dyn.TT

stepsize_0 = 1
armijo_maxiters = 20
cc = 0.5
beta = 0.7
visu_armijo = False             # Set to True to visualize the Armijo stepsize selection plots
print_iter = 1                  # Select the frequency of printing the Armijo plots

term_cond = 1e-6

def Newton (xx,uu,xx_ref,uu_ref,x0,max_iters,task):

    # Initialize all the arrays

    AA = np.zeros((ns,ns,TT))
    BB = np.zeros((ns,ni,TT))
    qq = np.zeros((ns,TT))
    rr = np.zeros((ni,TT))

    xx0 = x0

    QQ = np.zeros((ns,ns,TT))
    RR = np.zeros((ni,ni,TT))
    SS = np.zeros((ni,ns,TT))

    lmbd = np.zeros((ns,TT,max_iters))

    dJ = np.zeros((ni,TT,max_iters))

    JJ = np.zeros(max_iters)
    descent = np.zeros(max_iters)
    descent_arm = np.zeros(max_iters)

    deltaxx = np.zeros((ns,TT,max_iters))
    deltauu = np.zeros((ni,TT,max_iters))

    # Start the iterations

    for kk in range (max_iters-1):
        JJ[kk] = 0

        for tt in range(TT-1):
            temp_cost = cst.stagecost(xx[:,tt,kk],uu[:,tt,kk],xx_ref[:,tt],uu_ref[:,tt],task)['cost_t']
            JJ[kk] += temp_cost

        temp_cost = cst.terminalcost(xx[:,-1,kk],xx_ref[:,-1],task)['cost_T']
        JJ[kk] += temp_cost

        # Descent direction calculation

        lmbd_temp = cst.terminalcost(xx[:,TT-1,kk],xx_ref[:,TT-1],task)['DLx']
        lmbd[:,TT-1,kk] = lmbd_temp.squeeze()

        for tt in reversed(range(TT-1)):
            
            at = cst.stagecost(xx[:,tt,kk],uu[:,tt,kk],xx_ref[:,tt],uu_ref[:,tt],task)['DLx']
            bt = cst.stagecost(xx[:,tt,kk],uu[:,tt,kk],xx_ref[:,tt],uu_ref[:,tt],task)['DLu']

            qq[:,tt] = at.squeeze()
            rr[:,tt] = bt.squeeze()

            fx, fu = dyn.dynamics(xx[:,tt,kk],uu[:,tt,kk])[1:]         
            AA[:,:,tt] = fx.T
            BB[:,:,tt] = fu.T  

            lmbd_temp = AA[:,:,tt].T @ lmbd[:,tt+1,kk] + at
            dJ_temp = BB[:,:,tt].T @ lmbd[:,tt+1,kk] + bt 

            lmbd[:,tt,kk] = lmbd_temp.squeeze()
            dJ[:,tt,kk] = dJ_temp.squeeze()

        
        for tt in range(TT-1):
            QQ[:,:,tt] = cst.stagecost(xx[:,tt,kk],uu[:,tt,kk],xx_ref[:,tt],uu_ref[:,tt],task)['DLxx']
            RR[:,:,tt] = cst.stagecost(xx[:,tt,kk],uu[:,tt,kk],xx_ref[:,tt],uu_ref[:,tt],task)['DLuu']
            SS[:,:,tt] = cst.stagecost(xx[:,tt,kk],uu[:,tt,kk],xx_ref[:,tt],uu_ref[:,tt],task)['DLux']

        qqT = cst.terminalcost(xx[:,-1,kk],xx_ref[:,-1],task)['DLx']
        QQT = cst.terminalcost(xx[:,-1,kk],xx_ref[:,-1],task)['DLxx']

        # Computing the descent direction

        deltaxx[:,:,kk],deltauu[:,:,kk],KK,sigma = LQR(AA,BB,QQ,RR,SS,QQT,TT,xx0,qq,rr,qqT)
        
        for tt in reversed(range(TT-1)):            
            descent[kk] += deltauu[:,tt,kk].T @ deltauu[:,tt,kk]
            descent_arm[kk] += dJ[:,tt,kk].T @ deltauu[:,tt,kk]

        # Computing the stepsize using Armijo 
                      
        stepsizes = []
        costs_armijo = []

        stepsize = stepsize_0

        for ii in range(armijo_maxiters):

            xx_temp = np.zeros((ns,TT))
            uu_temp = np.zeros((ni,TT))

            xx_temp[:,0] = x0

            for tt in range(TT-1):
                uu_temp[:,tt] = uu[:,tt,kk] + KK[:,:,tt]@(xx_temp[:,tt]-xx[:,tt,kk]) + stepsize*sigma[:,tt]    
                xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt],uu_temp[:,tt])[0]

            JJ_temp = 0 

            for tt in range(TT-1):
                temp_cost = cst.stagecost(xx_temp[:,tt],uu_temp[:,tt],xx_ref[:,tt],uu_ref[:,tt],task)['cost_t']
                JJ_temp += temp_cost

            temp_cost = cst.terminalcost(xx_temp[:,-1],xx_ref[:,-1],task)['cost_T']
            JJ_temp += temp_cost

            stepsizes.append(stepsize)
            costs_armijo.append(np.min([JJ_temp,100*JJ[kk]]))

            if JJ_temp > JJ[kk] + cc*stepsize*descent_arm[kk]:
                stepsize = beta*stepsize

            else:
                print('Armijo stepsize = {:.3e}'.format(stepsize))
                break

        if visu_armijo and kk%print_iter==0:
            steps = np.linspace(0,stepsize_0,int(2e1))
            costs = np.zeros(len(steps))

            for ii in range(len(steps)):
                step = steps[ii]

                xx_temp = np.zeros((ns,TT))
                uu_temp = np.zeros((ni,TT))

                xx_temp[:,0] = x0

                for tt in range(TT-1):
                    uu_temp[:,tt] = uu[:,tt,kk] + KK[:,:,tt]@(xx_temp[:,tt]-xx[:,tt,kk]) + step*sigma[:,tt]  
                    xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt],uu_temp[:,tt])[0]

                JJ_temp = 0

                for tt in range(TT-1):
                    temp_cost = cst.stagecost(xx_temp[:,tt],uu_temp[:,tt],xx_ref[:,tt],uu_ref[:,tt],task)['cost_t']
                    JJ_temp += temp_cost

                temp_cost = cst.terminalcost(xx_temp[:,-1],xx_ref[:,-1],task)['cost_T']
                JJ_temp += temp_cost

                costs[ii] = np.min([JJ_temp,100*JJ[kk]])

            plt.figure(1)
            plt.clf()

            plt.suptitle("Armijo stepsize selection rule, iteration kk: " + str(kk), fontsize=16)
            plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - stepsize*d^k)$')
            plt.plot(steps, JJ[kk] + descent_arm[kk]*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
            plt.plot(steps, JJ[kk] + cc*descent_arm[kk]*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
            plt.scatter(stepsizes, costs_armijo, marker='*') 

            plt.grid()
            plt.xlabel('stepsize')
            plt.legend(fontsize=14)        
            plt.draw()

            plt.show()


        xx_temp = np.zeros((ns,TT))
        uu_temp = np.zeros((ni,TT))
        xx_temp[:,0] = xx0

        for tt in range(TT-1):
            uu_temp[:,tt] = uu[:,tt,kk] + KK[:,:,tt]@(xx_temp[:,tt]-xx[:,tt,kk]) + stepsizes[-1]*sigma[:,tt]
            xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt],uu_temp[:,tt])[0]
       
        xx[:,:,kk+1] = xx_temp
        uu[:,:,kk+1] = uu_temp

        print('Iter = {}\t Descent = {:.3e}\t Cost = {:.3e}'.format(kk,descent[kk], JJ[kk]))

        if descent[kk] < term_cond:
            print(' - Termination condition reached. Stopping the algorithm - ')
            break
        
    return xx,uu,descent,JJ,kk


def LQR (AAin,BBin,QQin,RRin,SSin,QQfin,TT,x0,qqin = None, rrin = None, qqfin = None):

    try:
        # check if matrix is (.. x .. x TT) - 3 dimensional array 
        ns, lA = AAin.shape[1:]
    except:
        # if not 3 dimensional array, make it (.. x .. x 1)
        AAin = AAin[:,:,None]
        ns, lA = AAin.shape[1:]

    try:  
        ni, lB = BBin.shape[1:]
    except:
        BBin = BBin[:,:,None]
        ni, lB = BBin.shape[1:]

    try:
        nQ, lQ = QQin.shape[1:]
    except:
        QQin = QQin[:,:,None]
        nQ, lQ = QQin.shape[1:]

    try:
        nR, lR = RRin.shape[1:]
    except:
        RRin = RRin[:,:,None]
        nR, lR = RRin.shape[1:]

    try:
        nSi, nSs, lS = SSin.shape
    except:
        SSin = SSin[:,:,None]
        nSi, nSs, lS = SSin.shape

    # Check dimensions consistency -- safety
    if nQ != ns:
        print("Matrix Q does not match number of states")
        exit()
    if nR != ni:
        print("Matrix R does not match number of inputs")
        exit()
    if nSs != ns:
        print("Matrix S does not match number of states")
        exit()
    if nSi != ni:
        print("Matrix S does not match number of inputs")
        exit()

    if lA < TT:
        AAin = AAin.repeat(TT, axis=2)
    if lB < TT:
        BBin = BBin.repeat(TT, axis=2)
    if lQ < TT:
        QQin = QQin.repeat(TT, axis=2)
    if lR < TT:
        RRin = RRin.repeat(TT, axis=2)
    if lS < TT:
        SSin = SSin.repeat(TT, axis=2)

    # # Check for affine terms

    # augmented = False

    # if qqin is not None or rrin is not None or qqfin is not None:
    #     augmented = True
    #     print("Augmented term!")

    KK = np.zeros((ni,ns,TT))
    sigma = np.zeros((ni,TT))
    PP = np.zeros((ns,ns,TT))
    pp = np.zeros((ns,TT))

    QQ = QQin
    RR = RRin
    SS = SSin
    QQf = QQfin

    qq = qqin
    rr = rrin
    qqf = qqfin

    AA = AAin
    
    BB = BBin

    xx = np.zeros((ns,TT))
    uu = np.zeros((ni,TT))

    xx[:,0] = x0

    PP[:,:,-1] = QQf
    pp[:,-1] = qqf

    # Solving the Riccati equation backwards in time

    for tt in reversed(range(TT-1)):
        QQt = QQ[:,:,tt]
        qqt = qq[:,tt][:,None]
        RRt = RR[:,:,tt]
        rrt = rr[:,tt][:,None]
        SSt = SS[:,:,tt]
        AAt = AA[:,:,tt]
        BBt = BB[:,:,tt]    
        PPtp = PP[:,:,tt+1]
        pptp = pp[:,tt+1][:,None]

        MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
        mmt = rrt + BBt.T @ pptp
        
        PPt = AAt.T @ PPtp @ AAt - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ (BBt.T@PPtp@AAt + SSt) + QQt
        ppt = AAt.T @ pptp - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ mmt + qqt

        PP[:,:,tt] = PPt
        pp[:,tt] = ppt.squeeze()

    # Evaluate gain KK forward in time
        
    for tt in range(TT-1):
        QQt = QQ[:,:,tt]
        qqt = qq[:,tt][:,None]
        RRt = RR[:,:,tt]
        rrt = rr[:,tt][:,None]
        SSt = SS[:,:,tt]
        AAt = AA[:,:,tt]
        BBt = BB[:,:,tt]

        PPtp = PP[:,:,tt+1]
        pptp = pp[:,tt+1][:,None]

        MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
        mmt = rrt + BBt.T @ pptp

        KK[:,:,tt] = -MMt_inv@(BBt.T@PPtp@AAt + SSt)
        sigma_t = -MMt_inv@mmt

        sigma[:,tt] = sigma_t.squeeze()

    for tt in range(TT-1):

        uu[:,tt] = KK[:,:,tt]@xx[:,tt] + sigma[:,tt]
        xx_p = AA[:,:,tt]@xx[:,tt] + BB[:,:,tt]@uu[:,tt]

        xx[:,tt+1] = xx_p

    return xx,uu,KK,sigma















