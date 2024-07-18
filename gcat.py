import numpy as np
import time

# define globals
global MAXITER,TOL,thetainv,thetainv
MAXITER=100
TOL=1E-12
thetainv=2/(1+5**0.5)

def CalcLogL(param,logLonly=False):
    # calculate log-likelihood constant
    cons=2*n*np.log(2*np.pi)
    # get a1, a2, b1, b2, gc from param
    a1=param[:,0]
    a2=param[:,1]
    b1=param[:,2]
    b2=param[:,3]
    gc=param[:,4]
    # calculate st dev Y1, Y2, delta, rho, and 1-rho^2
    sig1=np.exp((x*b1[None,:]).sum(axis=1))
    sig2=np.exp((x*b2[None,:]).sum(axis=1))
    delta=np.exp((x*gc[None,:]).sum(axis=1))
    rho=(delta-1)/(delta+1)
    unexp=(1-(rho**2))
    # calculate errors and rescaled errors (i.e. error/stdev)
    e1=y1-(x*a1[None,:]).sum(axis=1)
    e2=y2-(x*a2[None,:]).sum(axis=1)
    r1=(e1/sig1)
    r2=(e2/sig2)
    # calculate log|V| and quadratic term
    logdetV=2*n*np.log(2)+((x*((2*b1+2*b2+gc)[None,:])).sum())\
        -2*((np.log(delta+1)).sum())
    quadratic=(((r1**2)+(r2**2)-2*(rho*r1*r2))/(1-(rho**2))).sum()
    # calculate logL/n
    logL=-0.5*(cons+logdetV+quadratic)/n
    if not(logLonly):
        # get gradient of logL per observation 
        ga1=((x*((((r1/sig1)-(rho*r2/sig1))/unexp)[:,None])).sum(axis=0))/n
        ga2=((x*((((r2/sig2)-(rho*r1/sig2))/unexp)[:,None])).sum(axis=0))/n
        gb1=((x*(((((r1**2)-(rho*r1*r2))/unexp)[:,None])-1)).sum(axis=0))/n
        gb2=((x*(((((r2**2)-(rho*r1*r2))/unexp)[:,None])-1)).sum(axis=0))/n
        ggc=((x*((0.5*rho-0.5*((((delta**2)-1)/(4*delta))*(r1**2+r2**2)\
                               -(((delta**2+1)/(2*delta))\
                                 *r1*r2)))[:,None])).sum(axis=0))/n
        # stack gradients into grand vector
        g=np.vstack((ga1,ga2,gb1,gb2,ggc)).T
        # initialise hessian
        H=np.zeros((k,5,k,5))
        # get entries hessian
        H[:,0,:,0]=-(x.T@(x*((1/(unexp*(sig1**2)))[:,None])))/n
        H[:,1,:,1]=-(x.T@(x*((1/(unexp*(sig2**2)))[:,None])))/n
        H[:,0,:,1]=-(x.T@(x*((-rho/(unexp*sig1*sig2))[:,None])))/n
        H[:,0,:,2]=-(x.T@(x*(((1/(sig1*unexp))*(rho*r2-2*r1))[:,None])))/n
        H[:,1,:,3]=-(x.T@(x*(((1/(sig2*unexp))*(rho*r1-2*r2))[:,None])))/n
        H[:,0,:,3]=-(x.T@(x*(((1/(sig1*unexp))*(rho*r2))[:,None])))/n
        H[:,1,:,2]=-(x.T@(x*(((1/(sig2*unexp))*(rho*r1))[:,None])))/n
        H[:,0,:,4]=-(x.T@(x*(((1/(sig1*unexp))*(rho*r1\
                                           -((1+(rho**2))*(r2/2))))[:,None])))/n
        H[:,1,:,4]=-(x.T@(x*(((1/(sig2*unexp))*(rho*r2\
                                           -((1+(rho**2))*(r1/2))))[:,None])))/n
        H[:,2,:,2]=-(x.T@(x*(((1/unexp)*(2*(r1**2)-rho*r1*r2))[:,None])))/n
        H[:,3,:,3]=-(x.T@(x*(((1/unexp)*(2*(r2**2)-rho*r1*r2))[:,None])))/n
        H[:,2,:,3]=-(x.T@(x*(((1/unexp)*(-rho*r1*r2))[:,None])))/n
        H[:,2,:,4]=-(x.T@(x*(((0.5*r1*r2)-((rho/unexp)*(r1**2)))[:,None])))/n
        H[:,3,:,4]=-(x.T@(x*(((0.5*r1*r2)-((rho/unexp)*(r2**2)))[:,None])))/n
        H[:,4,:,4]=-(x.T@(x*(((((((delta**2)+1)/(8*delta))*((r1**2)+(r2**2)))\
                          -((((delta**2)-1)/(4*delta))*(r1*r2)))\
                         -((1-rho**2)/4))[:,None])))/n
        for i in range(4):
            for j in range(i+1,5):
                H[:,j,:,i]=H[:,i,:,j]
        return logL,g,H
    else:
        return logL

def Newton(param):
    # set iteration counter to zero and convergence to false
    i=0
    converged=False
    # while not converged and MAXITER not reached
    while not(converged) and i<MAXITER:
        # calculate log-likelihood, its gradient, and Hessian
        (logL,g,H)=CalcLogL(param)
        # calculate convergence criterion: mean squared gradient
        msg=(g**2).mean()
        # if convergence criterion met
        if msg<TOL:
            # set convergence to true and calculate sampling variance
            converged=True
        else:
            # get Newton-Raphson update
            update=-((np.linalg.inv(H.reshape((k*5,k*5)))\
                      @(g.reshape((k*5,1))))).reshape((k,5))
            # perform golden section to get new parameters estimates
            (param,j)=GoldenSection(param,update)
            # update iteration counter
            i+=1
            # print update
            print('Newton iteration '+str(i)+': logL='+str(logL)+'; '+str(j)\
                  +' line-search steps')
    return param,logL,g,H

def GoldenSection(param1,update):
    param2=param1+(1-thetainv)*update
    param3=param1+thetainv*update
    param4=param1+update
    logL2=CalcLogL(param2,logLonly=True)
    logL3=CalcLogL(param3,logLonly=True)
    # set iteration counter to zero and convergence to false
    i=0
    converged=False
    # while not converged and MAXITER not reached
    while not(converged) and i<MAXITER:
        if logL2>logL3:
            param4=param3
            param3=param2
            param2=thetainv*param1+(1-thetainv)*param4
            logL3=logL2
            logL2=CalcLogL(param2,logLonly=True)
        else:
            param1=param2
            param2=param3
            param3=thetainv*param4+(1-thetainv)*param1
            logL2=logL3
            logL3=CalcLogL(param3,logLonly=True)
        if ((param2-param3)**2).mean()<TOL:
            converged=True
        i+=1
    return param2,i

def SimulateDataOneSNP():
    # define globals for data
    global y1,y2,x,n,k
    # set seed and random-number generator
    S=192398123
    rng=np.random.default_rng(S)
    # set constants
    PLOIDY=2
    MAF=0.5
    n=5*(10**4)
    k=2
    # set true values
    a1t=np.array((0.5,0.3))
    a2t=np.array((0.3,0.5))
    b1t=np.array((0.2,0.1))
    b2t=np.array((0.4,0.2))
    gct=np.array((0.1,0.05))
    # draw genotypes single SNP
    g=rng.binomial(PLOIDY,MAF,size=n)
    # set matrix of regressors
    x=np.hstack((np.ones((n,1)),g[:,None]))
    # get SNP-dependent standard deviations and correlations
    std1=np.exp((x*b1t[None,:]).sum(axis=1))
    std2=np.exp((x*b2t[None,:]).sum(axis=1))
    delta=np.exp((x*gct[None,:]).sum(axis=1))
    rho=(delta-1)/(delta+1)
    # get two nid sources of noise
    eta1=rng.normal(size=n)
    eta2=rng.normal(size=n)
    # cast to correlated sources in accordance with implied var-covar
    eps1=eta1*std1
    eps2=(rho*eta1 + ((1-(rho**2))**0.5)*eta2)*std2
    # get outcomes
    y1=(x*a1t[None,:]).sum(axis=1)+eps1
    y2=(x*a2t[None,:]).sum(axis=1)+eps2

def TestOneSNP():
    print('Simulating data')
    SimulateDataOneSNP()
    print('Initialising parameters')
    param=np.zeros((k,5))
    print('Starting estimation')
    t0=time.time()
    (param,logL,g,H)=Newton(param)
    paramVar=-np.linalg.inv(H.reshape((k*5,k*5)))/n
    paramSE=((np.diag(paramVar))**0.5).reshape((k,5))
    print('Estimation finished')
    print('Time elapsed in estimation = '+str(time.time()-t0)+' sec')
    return param,paramSE
