import pandas as pd
import numpy as np

from datetime import datetime
import pandas as pd

from scipy import optimize
from scipy import integrate

N0 = 0
t = 0
SIR0 = 0


def SIR_modelling(y_data):
    global SIR0, t, N0

    ydata = np.array(y_data)
    t = np.arange(len(ydata))

    N0 = 10000000
    I0=ydata[0]
    S0=N0-I0
    R0=0
    SIR0 = (S0,I0,R0)


    popt = [0.2, 0.1]
    fit_odeint(t, *popt)

    popt, pcov = optimize.curve_fit(fit_odeint, t, ydata)
    perr = np.sqrt(np.diag(pcov))

    print('standard deviation errors : ',str(perr), ' start infect:',ydata[0])
    print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])

    fitted=fit_odeint(t, *popt)

    return t, fitted

def SIR_model_t(SIR,t,beta,gamma):
    ''' Simple SIR model
        S: susceptible population
        t: time step, mandatory for integral.odeint
        I: infected people
        R: recovered people
        beta:

        overall condition is that the sum of changes (differnces) sum up to 0
        dS+dI+dR=0
        S+I+R= N (constant size of population)

    '''

    S,I,R=SIR
    dS_dt=-beta*S*I/N0
    dI_dt=beta*S*I/N0-gamma*I
    dR_dt=gamma*I
    return dS_dt,dI_dt,dR_dt

def fit_odeint(x, beta, gamma):
    return integrate.odeint(SIR_model_t, SIR0, t, args = (beta, gamma))[:,1]
