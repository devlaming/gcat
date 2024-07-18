import numpy as np
import time
from tqdm import tqdm

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
    sig1=np.exp((xs*param[None,:,2]).sum(axis=1))
    sig2=np.exp((xs*param[None,:,3]).sum(axis=1))
    delta=np.exp((xs*param[None,:,4]).sum(axis=1))
    rho=(delta-1)/(delta+1)
    unexp=(1-(rho**2))
    # calculate errors and rescaled errors (i.e. error/stdev)
    e1=y1-(xs*param[None,:,0]).sum(axis=1)
    e2=y2-(xs*param[None,:,1]).sum(axis=1)
    r1=(e1/sig1)
    r2=(e2/sig2)
    # calculate log|V| and quadratic term
    logdetV=2*n*np.log(2)\
        +((xs*(((2*param[:,2]+2*param[:,3]+param[:,4]))[None,:])).sum())\
        -2*((np.log(delta+1)).sum())
    quadratic=(((r1**2)+(r2**2)-2*(rho*r1*r2))/(1-(rho**2))).sum()
    # calculate logL/n
    logL=-0.5*(cons+logdetV+quadratic)/n
    if logLonly:
        return logL
    else:
        # get gradient of logL per observation 
        ga1=((xs*((((r1/sig1)-(rho*r2/sig1))/unexp)[:,None])).sum(axis=0))/n
        ga2=((xs*((((r2/sig2)-(rho*r1/sig2))/unexp)[:,None])).sum(axis=0))/n
        gb1=((xs*(((((r1**2)-(rho*r1*r2))/unexp)[:,None])-1)).sum(axis=0))/n
        gb2=((xs*(((((r2**2)-(rho*r1*r2))/unexp)[:,None])-1)).sum(axis=0))/n
        ggc=((xs*((0.5*rho-0.5*((((delta**2)-1)/(4*delta))*(r1**2+r2**2)\
                               -(((delta**2+1)/(2*delta))\
                                 *r1*r2)))[:,None])).sum(axis=0))/n
        # stack gradients into grand vector
        grad=np.vstack((ga1,ga2,gb1,gb2,ggc)).T
        # initialise hessian
        H=np.zeros((ks,5,ks,5))
        # get entries hessian
        H[:,0,:,0]=-(xs.T@(xs*((1/(unexp*(sig1**2)))[:,None])))/n
        H[:,1,:,1]=-(xs.T@(xs*((1/(unexp*(sig2**2)))[:,None])))/n
        H[:,0,:,1]=-(xs.T@(xs*((-rho/(unexp*sig1*sig2))[:,None])))/n
        H[:,0,:,2]=-(xs.T@(xs*(((1/(sig1*unexp))*(rho*r2-2*r1))[:,None])))/n
        H[:,1,:,3]=-(xs.T@(xs*(((1/(sig2*unexp))*(rho*r1-2*r2))[:,None])))/n
        H[:,0,:,3]=-(xs.T@(xs*(((1/(sig1*unexp))*(rho*r2))[:,None])))/n
        H[:,1,:,2]=-(xs.T@(xs*(((1/(sig2*unexp))*(rho*r1))[:,None])))/n
        H[:,0,:,4]=-(xs.T@(xs*(((1/(sig1*unexp))*(rho*r1\
                                           -((1+(rho**2))*(r2/2))))[:,None])))/n
        H[:,1,:,4]=-(xs.T@(xs*(((1/(sig2*unexp))*(rho*r2\
                                           -((1+(rho**2))*(r1/2))))[:,None])))/n
        H[:,2,:,2]=-(xs.T@(xs*(((1/unexp)*(2*(r1**2)-rho*r1*r2))[:,None])))/n
        H[:,3,:,3]=-(xs.T@(xs*(((1/unexp)*(2*(r2**2)-rho*r1*r2))[:,None])))/n
        H[:,2,:,3]=-(xs.T@(xs*(((1/unexp)*(-rho*r1*r2))[:,None])))/n
        H[:,2,:,4]=-(xs.T@(xs*(((0.5*r1*r2)-((rho/unexp)*(r1**2)))[:,None])))/n
        H[:,3,:,4]=-(xs.T@(xs*(((0.5*r1*r2)-((rho/unexp)*(r2**2)))[:,None])))/n
        H[:,4,:,4]=-(xs.T@(xs*(((((((delta**2)+1)/(8*delta))*((r1**2)+(r2**2)))\
                          -((((delta**2)-1)/(4*delta))*(r1*r2)))\
                         -((1-rho**2)/4))[:,None])))/n
        for i in range(4):
            for j in range(i+1,5):
                H[:,j,:,i]=H[:,i,:,j]
        return logL,grad,H

def Newton(param,silent=False):
    # set iteration counter to zero and convergence to false
    i=0
    converged=False
    # while not converged and MAXITER not reached
    while not(converged) and i<MAXITER:
        # calculate log-likelihood, its gradient, and Hessian
        (logL,grad,H)=CalcLogL(param)
        # calculate convergence criterion: mean squared gradient
        msg=(grad**2).mean()
        # if convergence criterion met
        if msg<TOL:
            # set convergence to true and calculate sampling variance
            converged=True
        else:
            # get Newton-Raphson update
            update=-((np.linalg.inv(H.reshape((ks*5,ks*5)))\
                      @(grad.reshape((ks*5,1))))).reshape((ks,5))
            # perform golden section to get new parameters estimates
            (param,j)=GoldenSection(param,update)
            # update iteration counter
            i+=1
            # print update if not silent
            if not(silent):
                print('Newton iteration '+str(i)+': logL='+str(logL)+'; '\
                      +str(j)+' line-search steps')
    return param,logL,grad,H

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

def GCAT():
    # consider globals for xs and ks
    global xs,ks
    print('2. ESTIMATION MODEL WITHOUT SNPS')
    print('Initialising parameters')
    param0=np.zeros((k,5))
    (param0,logL0,grad0,H0)=Newton(param0)
    param0Var=-np.linalg.inv(H0.reshape((k*5,k*5)))/n
    param0SE=((np.diag(param0Var))**0.5).reshape((k,5))
    print('3. ESTIMATION MODELS WITH SNPS')
    snp=np.zeros((m,5))
    snpSE=np.zeros((m,5))
    snpLRT=np.zeros(m)
    for j in tqdm(range(m)):
        param1=np.vstack((param0.copy(),np.zeros((1,5))))
        xs=np.hstack((x,g[:,j][:,None]))
        ks=k+1
        (param1,logL1,grad1,H1)=Newton(param1,silent=True)
        param1Var=-np.linalg.inv(H1.reshape(((k+1)*5,(k+1)*5)))/n
        param1SE=((np.diag(param1Var))**0.5).reshape(((k+1),5))
        snp[j,:]=param1[-1,:]
        snpSE[j,:]=param1SE[-1,:]
        snpLRT[j]=2*n*(logL1-logL0)
    snpWald=(snp/snpSE)**2
    return param0,param0SE,snp,snpSE,snpWald,snpLRT

def SimulateData():
    # define globals for data and true parameters
    global y1,y2,x,xs,g,paramtrue
    # set seed and random-number generator
    S=192398123
    rng=np.random.default_rng(S)
    # draw regressors
    x=np.hstack((np.ones((n,1)),rng.normal(size=(n,k-1))))
    xs=x
    # draw genotypes
    g=rng.binomial(PLOIDY,MAF,size=(n,m))
    # set combined matrix
    X=np.hstack((x,g))
    # draw true effects
    paramtrue=rng.normal(scale=0.1,size=(k+m,5))
    # get SNP-dependent standard deviations and correlations
    std1=np.exp((X*paramtrue[None,:,2]).sum(axis=1))
    std2=np.exp((X*paramtrue[None,:,3]).sum(axis=1))
    delta=np.exp((X*paramtrue[None,:,4]).sum(axis=1))
    rho=(delta-1)/(delta+1)
    # get two nid sources of noise
    eta1=rng.normal(size=n)
    eta2=rng.normal(size=n)
    # cast to correlated sources in accordance with implied var-covar
    eps1=eta1*std1
    eps2=(rho*eta1+((1-(rho**2))**0.5)*eta2)*std2
    # get outcomes
    y1=(X*paramtrue[None,:,0]).sum(axis=1)+eps1
    y2=(X*paramtrue[None,:,1]).sum(axis=1)+eps2

def Test():
    print('1. SIMULATING DATA')
    global n,k,ks,m,MAF
    n=int(5e4)
    k=2
    ks=k
    m=10
    MAF=0.5
    SimulateData()
    (param0,param0SE,snp,snpSE,snpWald,snpLRT)=GCAT()
    return param0,param0SE,snp,snpSE,snpWald,snpLRT

(param0,param0SE,snp,snpSE,snpWald,snpLRT)=Test()
