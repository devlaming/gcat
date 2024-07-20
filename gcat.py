import numpy as np
import warnings
from tqdm import tqdm

# define globals
global MAXITER,TOL,TAUMAF,MINVAR,MINEVALH,thetainv
MAXITER=50
TOL=1E-12
TAUMAF=0.05
MINVAR=0.01
MINEVALMH=1E-6
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
        # define gradient per observation as 3d array
        G=np.zeros((ks,5,n))
        G[:,0,:]=((xs*((((r1/sig1)-(rho*r2/sig1))/unexp)[:,None]))/n).T
        G[:,1,:]=((xs*((((r2/sig2)-(rho*r1/sig2))/unexp)[:,None]))/n).T
        G[:,2,:]=((xs*(((((r1**2)-(rho*r1*r2))/unexp)[:,None])-1))/n).T
        G[:,3,:]=((xs*(((((r2**2)-(rho*r1*r2))/unexp)[:,None])-1))/n).T
        G[:,4,:]=((xs*((0.5*rho-0.5*((((delta**2)-1)/(4*delta))*(r1**2+r2**2)\
                                    -(((delta**2+1)/(2*delta))\
                                      *r1*r2)))[:,None]))/n).T
        # calculate gradient by summing gradient per obs along observations
        grad=G.sum(axis=2)
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
        return logL,grad,H,G

def Newton(param,silent=False):
    # set iteration counter to zero and convergence to false
    i=0
    converged=False
    # while not converged and MAXITER not reached
    while not(converged) and i<MAXITER:
        # calculate log-likelihood, its gradient, and Hessian
        (logL,grad,H,G)=CalcLogL(param)
        # unpack Hessian to matrix
        UH=H.reshape((ks*5,ks*5))
        # take average of UH and UH.T for numerical stability
        UH=(UH+UH.T)/2
        # get eigenvalue decomposition of minus unpackage Hessian
        (D,P)=np.linalg.eigh(-UH)
        if (D<MINEVALMH).sum()>0:
            print('bended '+str((D<MINEVALMH).sum())+' eigenvalues of Hessian for numerical stability')
            D[D<MINEVALMH]=MINEVALMH
        # get Newton-Raphson update vector
        update=P@((((grad.reshape((ks*5,1))).T@P)/D).T)
        # calculate convergence criterion
        msg=(update*grad.reshape((ks*5,1))).sum()
        # if convergence criterion met
        if msg<TOL:
            # set convergence to true and calculate sampling variance
            converged=True
        else:
            # perform golden section to get new parameters estimates
            (param,j)=GoldenSection(param,update.reshape((ks,5)))
            # update iteration counter
            i+=1
            # print update if not silent
            if not(silent):
                print('Newton iteration '+str(i)+': logL='+str(logL)+'; '\
                      +str(j)+' line-search steps')
    return param,logL,grad,H,G,converged

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
        #if mid-left val >= mid-right val: set mid-right as right
        if logL2>=logL3: 
            param4=param3
            param3=param2
            param2=thetainv*param1+(1-thetainv)*param4
            logL3=logL2
            logL2=CalcLogL(param2,logLonly=True)
        #if mid-right val > mid-left val: set mid-left as left
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

def InitialiseParams():
    invXTX=np.linalg.inv((x.T@x)/n)
    a1=invXTX@((x.T@y1)/n)
    a2=invXTX@((x.T@y2)/n)
    e1=y1-x@a1
    e2=y2-x@a2
    z1=0.5*np.log(e1**2)
    z2=0.5*np.log(e2**2)
    b1=invXTX@((x.T@z1)/n)
    b2=invXTX@((x.T@z2)/n)
    gc=np.zeros(k)
    param0=np.vstack((a1,a2,b1,b2,gc)).T
    return param0

def GCAT():
    # consider globals for xs and ks
    global xs,ks
    print('2. ESTIMATION MODEL WITHOUT SNPS')
    print('Initialising parameters')
    param0=InitialiseParams()
    (param0,logL0,grad0,H0,G0,converged0)=Newton(param0)
    if not(converged0):
        raise RuntimeError('Estimates baseline model (without SNPs) not converged')
    invH0=np.linalg.inv(H0.reshape((k*5,k*5)))
    param0Var=-invH0/n
    param0SE=((np.diag(param0Var))**0.5).reshape((k,5))
    GGT0=(G0.reshape((k*5,n)))@((G0.reshape((k*5,n))).T)
    param0RobustVar=invH0@GGT0@invH0
    param0RobustSE=((np.diag(param0RobustVar))**0.5).reshape((k,5))
    print('3. ESTIMATION MODELS WITH SNPS')
    snp=np.empty((m,5))*np.nan
    snpSE=np.empty((m,5))*np.nan
    snpRobustSE=np.empty((m,5))*np.nan
    snpLRT=np.empty((m))*np.nan
    for j in tqdm(range(m)):
        param1=np.vstack((param0.copy(),np.zeros((1,5))))
        xs=np.hstack((x,g[:,j][:,None]))
        ks=k+1
        (param1,logL1,grad1,H1,G1,converged1)=Newton(param1,silent=True)
        if converged1:
            invH1=np.linalg.inv(H1.reshape(((k+1)*5,(k+1)*5)))
            param1Var=-invH1/n
            param1SE=((np.diag(param1Var))**0.5).reshape(((k+1),5))
            GGT1=(G1.reshape(((k+1)*5,n)))@((G1.reshape(((k+1)*5,n))).T)
            param1RobustVar=invH1@GGT1@invH1
            param1RobustSE=((np.diag(param1RobustVar))**0.5).reshape((k+1,5))
            snp[j,:]=param1[-1,:]
            snpSE[j,:]=param1SE[-1,:]
            snpRobustSE[j,:]=param1RobustSE[-1,:]
            snpLRT[j]=2*n*(logL1-logL0)
        else:
            warnings.warn('Model for SNP '+str(j)+' did not converge')
    snpWald=(snp/snpSE)**2
    snpRobustWald=(snp/snpRobustSE)**2
    return param0,param0SE,snp,snpSE,snpRobustSE,snpWald,snpRobustWald,snpLRT

def SimulateData(seed,h2y1,h2y2,h2sig1,h2sig2,h2rho):
    # define globals for data and true parameters
    global k,ks,y1,y2,x,xs,g,paramtrue
    # set random-number generator
    rng=np.random.default_rng(seed)
    # draw allele frequencies between (MAFTAU,1-MAFTAU)
    f=TAUMAF+(1-2*TAUMAF)*rng.uniform(size=m)
    # initialise genotype matrix and empirical AF
    g=np.zeros((n,m))
    eaf=np.zeros(m)
    # set number of SNPs not done drawing yet to m
    notdone=(np.ones(m)==1)
    mnotdone=notdone.sum()
    # while number of SNPs not done is at least 1
    while mnotdone>0: 
        # draw as many biallelic SNPs
        u=rng.uniform(size=(n,mnotdone))
        thisg=np.ones((n,mnotdone))
        thisg[u<(((1-f[notdone])**2)[None,:])]=0
        thisg[u>((1-(f[notdone]**2))[None,:])]=2
        g[:,notdone]=thisg
        # calculate empirical AF
        eaf[notdone]=thisg.mean(axis=0)/2
        # find SNPs with insufficient variation
        notdone=2*(eaf*(1-eaf))<MINVAR
        mnotdone=notdone.sum()
    # standardise genotype matrix
    g=(g-2*(eaf[None,:]))/(((2*eaf*(1-eaf))**0.5)[None,:])
    # draw effects (prior to scaling)
    alpha1=rng.normal(size=m)
    alpha2=rng.normal(size=m)
    beta1=rng.normal(size=m)
    beta2=rng.normal(size=m)
    gamma=rng.normal(size=m)
    # rescale betas and gammas to yield desired h2, assuming intercept has coefficient 1
    beta1=(beta1/(((beta1**2).sum())**0.5))*((h2sig1/(1-h2sig1))**0.5)
    beta2=(beta2/(((beta2**2).sum())**0.5))*((h2sig2/(1-h2sig2))**0.5)
    gamma=(gamma/(((gamma**2).sum())**0.5))*((h2rho/(1-h2rho))**0.5)
    # calculate standard deviations and correlations
    sig1=np.exp(1+((g*beta1[None,:]).sum(axis=1)))
    sig2=np.exp(1+((g*beta2[None,:]).sum(axis=1)))
    delta=np.exp(1+((g*gamma[None,:]).sum(axis=1)))
    rho=(delta-1)/(delta+1)
    # calculate average variance of both traits
    vare1=(sig1**2).mean()
    vare2=(sig2**2).mean()
    # rescale alphas to yield desired h2
    alpha1=(alpha1/(((alpha1**2).sum())**0.5))*(((h2y1*vare1)/(1-h2y1))**0.5)
    alpha2=(alpha2/(((alpha2**2).sum())**0.5))*(((h2y2*vare2)/(1-h2y2))**0.5)
    # draw noise factors
    eta1=rng.normal(size=n)
    eta2=rng.normal(size=n)
    # scale and mix noise to achieve desired standard deviations and correlations 
    e1=sig1*eta1
    e2=((rho*eta1)+(((1-(rho**2))**0.5)*eta2))*sig2
    # draw outcomes
    y1=((g*alpha1[None,:]).sum(axis=1))+e1
    y2=((g*alpha2[None,:]).sum(axis=1))+e2
    # set intercept as baseline model regressor
    x=np.ones((n,1))
    xs=x.copy()
    k=1
    ks=1
    # store true effects
    paramtrue=np.ones((m+1,5))
    paramtrue[1:,0]=alpha1
    paramtrue[1:,1]=alpha2
    paramtrue[1:,2]=beta1
    paramtrue[1:,3]=beta2
    paramtrue[1:,4]=gamma

def Test():
    global n,m
    print('1. SIMULATING DATA')
    seed=1873798321
    n=int(5e4)
    m=20
    h2y1=0.5
    h2y2=0.4
    h2sig1=0.3
    h2sig2=0.2
    h2rho=0.1
    SimulateData(seed,h2y1,h2y2,h2sig1,h2sig2,h2rho)
    (param0,param0SE,snp,snpSE,snpRobustSE,snpWald,snpRobustWald,snpLRT)=GCAT()
    return param0,param0SE,snp,snpSE,snpRobustSE,snpWald,snpRobustWald,snpLRT

(param0,param0SE,snp,snpSE,snpRobustSE,snpWald,snpRobustWald,snpLRT)=Test()
