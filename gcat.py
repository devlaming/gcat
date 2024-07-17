import numpy as np

# define globals
global MAXITER,TOL,thetainv,thetainv,n,k
MAXITER=100
TOL=1E-12
thetainv=2/(1+5**0.5)
n=5*(10**4)
k=2

def CalcLogL(param,logLonly=False):
    # calculate log-likelihood constant
    cons=2*n*np.log(2*np.pi)
    # get a1, a2, b1, b2, gc from param
    a1=param[0*k:1*k]
    a2=param[1*k:2*k]
    b1=param[2*k:3*k]
    b2=param[3*k:4*k]
    gc=param[4*k:5*k]
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
        g=np.concatenate((ga1,ga2,gb1,gb2,ggc))
        # get hessian
        ha1a1=-(x.T@(x*((1/(unexp*(sig1**2)))[:,None])))/n
        ha2a2=-(x.T@(x*((1/(unexp*(sig2**2)))[:,None])))/n
        ha1a2=-(x.T@(x*((-rho/(unexp*sig1*sig2))[:,None])))/n
        ha1b1=-(x.T@(x*(((1/(sig1*unexp))*(rho*r2-2*r1))[:,None])))/n
        ha1b2=-(x.T@(x*(((1/(sig1*unexp))*(rho*r2))[:,None])))/n
        ha2b1=-(x.T@(x*(((1/(sig2*unexp))*(rho*r1))[:,None])))/n
        ha2b2=-(x.T@(x*(((1/(sig2*unexp))*(rho*r1-2*r2))[:,None])))/n
        ha1gc=-(x.T@(x*(((1/(sig1*unexp))*(rho*r1\
                                           -((1+(rho**2))*(r2/2))))[:,None])))/n
        ha2gc=-(x.T@(x*(((1/(sig2*unexp))*(rho*r2\
                                           -((1+(rho**2))*(r1/2))))[:,None])))/n
        hb1b1=-(x.T@(x*(((1/unexp)*(2*(r1**2)-rho*r1*r2))[:,None])))/n
        hb2b2=-(x.T@(x*(((1/unexp)*(2*(r2**2)-rho*r1*r2))[:,None])))/n
        hb1b2=-(x.T@(x*(((1/unexp)*(-rho*r1*r2))[:,None])))/n
        hb1gc=-(x.T@(x*(((0.5*r1*r2)-((rho/unexp)*(r1**2)))[:,None])))/n
        hb2gc=-(x.T@(x*(((0.5*r1*r2)-((rho/unexp)*(r2**2)))[:,None])))/n
        hgcgc=-(x.T@(x*(((((((delta**2)+1)/(8*delta))*((r1**2)+(r2**2)))\
                          -((((delta**2)-1)/(4*delta))*(r1*r2)))\
                         -((1-rho**2)/4))[:,None])))/n
        # stack hessians into grand matrix
        H1=np.hstack((ha1a1,ha1a2,ha1b1,ha1b2,ha1gc))
        H2=np.hstack((ha1a2.T,ha2a2,ha2b1,ha2b2,ha2gc))
        H3=np.hstack((ha1b1.T,ha2b1.T,hb1b1,hb1b2,hb1gc))
        H4=np.hstack((ha1b2.T,ha2b2.T,hb1b2.T,hb2b2,hb2gc))
        H5=np.hstack((ha1gc.T,ha2gc.T,hb1gc.T,hb2gc.T,hgcgc))
        H=np.vstack((H1,H2,H3,H4,H5))
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
            paramVar=-np.linalg.inv(H)/n
            paramSE=(np.diag(paramVar))**0.5
        else:
            # get Newton-Raphson update
            update=-np.linalg.inv(H)@g
            # perform golden section to get new parameters estimates
            param=GoldenSection(param,update)
            # update iteration counter
            i+=1
    return param,paramSE,paramVar,logL,g

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
    return param2

def SimulateData():
    # define globals for data
    global y1,y2,x
    # set seed and random-number generator
    S=192398123
    rng=np.random.default_rng(S)
    # set constants
    PLOIDY=2
    MAF=0.5
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

def TryNewton():
    SimulateData()
    param=np.zeros(5*k)
    (param,paramSE,paramVar,logL,g)=Newton(param)
    return param,paramSE,paramVar,logL,g

(param,paramSE,paramVar,logL,g)=TryNewton()

