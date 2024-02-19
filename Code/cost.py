'''
  This file contains the cost functions for the optimal control problem.
'''

import numpy as np
import dynamics as dyn

ns = dyn.ns 
ni = dyn.ni

# The cost function parameters for task 1
QQ_task1 = 0.000001*np.diag([2500,2500,25,0.001,0.001,0.001,25,0.001])
RR_task1 = 0.1*np.array([[1,0],[0,1]])
QQT_task1 = QQ_task1

# The cost function parameters for task 2
QQ_task2 = 0.000001*np.diag([2500,25000,0.001,0.001,0.001,0.001,0.001,0.001])
RR_task2 = 0.1*np.array([[1,0],[0,1]])
QQT_task2 = QQ_task2

RR = 0.01*np.array([[1,0],[0,1]])

def stagecost(xx,uu,xx_ref,uu_ref,task):
  if task == 1:
    QQ = QQ_task1
    RR = RR_task1
  elif task == 2:
    QQ = QQ_task2
    RR = RR_task2

  state_err = (xx - xx_ref)
  input_err = (uu - uu_ref)

  ll = 0.5*state_err.T @ QQ @ state_err + 0.5*input_err.T @ RR @ input_err  
  DLx = QQ @ state_err 
  DLu = RR @ input_err

  DLxx = QQ
  DLuu = RR
  DLux = np.array(np.zeros((ni, ns)))
  
  ll = ll.squeeze()

  out = {
        'cost_t': ll,
        'DLx': DLx,
        'DLu': DLu,
        'DLxx': DLxx,
        'DLux': DLux,
        'DLuu': DLuu
    }

  return out


def terminalcost(xx_T,xx_T_ref,task):

  if task == 1:
    QQT = QQT_task1
  elif task == 2:
    QQT = QQT_task2


  state_err = (xx_T - xx_T_ref)

  llT = 0.5*state_err.T @ QQT @ state_err  

  DLx = QQT @ state_err
  DLxx = QQT

  llT = llT.squeeze()

  out = {
    'cost_T': llT,
    'DLx': DLx,
    'DLxx': DLxx,
    }

  return out