import numpy as np
import time

# define globals
global MAXITER,TOL,PLOIDY,thetainv,thetainv
MAXITER=100
TOL=1E-12
PLOIDY=2
thetainv=2/(1+5**0.5)

def CalcLogL(param,logLonly=False):
    # calculate log-likelihood constant
    cons=2*n*np.log(2*np.pi)
    # calculate st dev Y1, Y2, delta, rho, and 1-rho^2
    sig1=np.exp((x*param[None,:,2]).sum(axis=1))
    sig2=np.exp((x*param[None,:,3]).sum(axis=1))
    delta=np.exp((x*param[None,:,4]).sum(axis=1))
    rho=(delta-1)/(delta+1)
    unexp=(1-(rho**2))
    # calculate errors and rescaled errors (i.e. error/stdev)
    e1=y1-(x*param[None,:,0]).sum(axis=1)
    e2=y2-(x*param[None,:,1]).sum(axis=1)
    r1=(e1/sig1)
    r2=(e2/sig2)
    # calculate log|V| and quadratic term
    logdetV=2*n*np.log(2)\
        +((x*(((2*param[:,2]+2*param[:,3]+param[:,4]))[None,:])).sum())\
        -2*((np.log(delta+1)).sum())
    quadratic=(((r1**2)+(r2**2)-2*(rho*r1*r2))/(1-(rho**2))).sum()
    # calculate logL/n
    logL=-0.5*(cons+logdetV+quadratic)/n
    if logLonly:
        return logL
    else:
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

def SimulateData():
    # define globals for data and true parameters
    global y1,y2,x,paramtrue
    # set seed and random-number generator
    S=192398123
    rng=np.random.default_rng(S)
    # draw regressors
    x=np.hstack((np.ones((n,1)),rng.normal(size=(n,k-(m+1)))))
    # draw genotypes
    g=rng.binomial(PLOIDY,MAF,size=(n,m))
    # set combined matrix
    x=np.hstack((x,g))
    # draw true effects
    paramtrue=rng.normal(scale=0.1,size=(k,5))
    # get SNP-dependent standard deviations and correlations
    std1=np.exp((x*paramtrue[None,:,2]).sum(axis=1))
    std2=np.exp((x*paramtrue[None,:,3]).sum(axis=1))
    delta=np.exp((x*paramtrue[None,:,4]).sum(axis=1))
    rho=(delta-1)/(delta+1)
    # get two nid sources of noise
    eta1=rng.normal(size=n)
    eta2=rng.normal(size=n)
    # cast to correlated sources in accordance with implied var-covar
    eps1=eta1*std1
    eps2=(rho*eta1+((1-(rho**2))**0.5)*eta2)*std2
    # get outcomes
    y1=(x*paramtrue[None,:,0]).sum(axis=1)+eps1
    y2=(x*paramtrue[None,:,1]).sum(axis=1)+eps2

def Test():
    global n,k,m,MAF
    n=int(5e4)
    k=12
    m=10
    MAF=0.5
    print('Simulating data')
    SimulateData()
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

