import numpy as np
import warnings
from tqdm import tqdm
from scipy import stats

# define globals
global MAXITER,TOL,TAUMAF,MINVAR,MINEVALH,thetainv
MAXITER=50
TOL=1E-12
TAUTRUEMAF=0.05
TAUDATAMAF=0.01
MINVAR=0.01
MINEVALMH=1E-3
thetainv=2/(1+5**0.5)

def CalcLogL(param,logLonly=False):
    # calculate log-likelihood constant
    cons=N*np.log(2*np.pi)
    # calculate st dev Y1, Y2, delta, rho
    sig1=np.exp((xs*param[None,:,2]).sum(axis=1))
    sig2=np.exp((xs*param[None,:,3]).sum(axis=1))
    delta=np.exp((xs*param[None,:,4]).sum(axis=1))
    rho=(delta-1)/(delta+1)
    # set halting variable to false
    halt=False
    # if at least one st.dev is zero: halt=True
    if (((sig1==0).sum())+((sig2==0).sum()))>0:
        halt=True
    # if at least one delta is 0 or np.inf: halt=True
    if (((delta==0).sum())+(np.isinf(delta).sum()))>0:
        halt=True
    # if at least one rsq is 1: halt=True
    if ((rho**2==1).sum())>0:
        halt=True
    # if halt=True: set -np.inf as log-likelihood
    if halt:
        logL=-np.inf
        if logLonly:
            return logL
        else:
            return logL,None,None,None
    # calculate errors 
    e1=y1-(xs*param[None,:,0]).sum(axis=1)
    e2=y2-(xs*param[None,:,1]).sum(axis=1)
    # set rho and errors to zero and delta to one at missing points
    rho[~ybothnotnan]=0
    e1[~y1notnan]=0
    e2[~y2notnan]=0
    delta[~ybothnotnan]=1
    # set sigma1 to np.inf when y1 is missing and vice versa
    sig1[~y1notnan]=np.inf
    sig2[~y2notnan]=np.inf
    # calculate rescaled errors (i.e. error/stdev)
    r1=(e1/sig1)
    r2=(e2/sig2)
    # calculate 1-rsq
    unexp=1-(rho**2)
    # calculate log|V| and quadratic term
    logdetV=2*nboth*np.log(2)+(xs[ybothnotnan,:]*(param[:,4])[None,:]).sum()\
        +(xs[y1notnan,:]*(2*param[:,2])).sum()\
            +(xs[y2notnan,:]*(2*param[:,3])).sum()\
            -2*((np.log(delta[ybothnotnan]+1)).sum())
    quadratic=(((r1**2)+(r2**2)-2*(rho*r1*r2))/unexp).sum()
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
        # set gradient to zero w.r.t. beta1 for observation with missing y1
        # and idem w.r.t. beta2 if y2 is missing
        G[:,2,~y1notnan]=0
        G[:,3,~y2notnan]=0
        # calculate gradient by summing gradient per obs along observations
        grad=G.sum(axis=2)
        # initialise hessian
        H=np.zeros((ks,5,ks,5))
        # get entries hessian
        H[:,0,:,0]=-(xs.T@(xs*((1/(unexp*(sig1**2)))[:,None])))/n
        H[:,1,:,1]=-(xs.T@(xs*((1/(unexp*(sig2**2)))[:,None])))/n
        H[:,0,:,1]=-(xs.T@(xs*((-rho/(unexp*sig1*sig2))[:,None])))/n
        H[:,0,:,2]=(xs.T@(xs*(((1/(sig1*unexp))*(rho*r2-2*r1))[:,None])))/n
        H[:,1,:,3]=(xs.T@(xs*(((1/(sig2*unexp))*(rho*r1-2*r2))[:,None])))/n
        H[:,0,:,3]=(xs.T@(xs*(((1/(sig1*unexp))*(rho*r2))[:,None])))/n
        H[:,1,:,2]=(xs.T@(xs*(((1/(sig2*unexp))*(rho*r1))[:,None])))/n
        H[:,0,:,4]=(xs.T@(xs*(((1/(sig1*unexp))*(rho*r1\
                                           -((1+(rho**2))*(r2/2))))[:,None])))/n
        H[:,1,:,4]=(xs.T@(xs*(((1/(sig2*unexp))*(rho*r2\
                                           -((1+(rho**2))*(r1/2))))[:,None])))/n
        H[:,2,:,2]=-(xs.T@(xs*(((1/unexp)*(2*(r1**2)-rho*r1*r2))[:,None])))/n
        H[:,3,:,3]=-(xs.T@(xs*(((1/unexp)*(2*(r2**2)-rho*r1*r2))[:,None])))/n
        H[:,2,:,3]=-(xs.T@(xs*(((1/unexp)*(-rho*r1*r2))[:,None])))/n
        H[:,2,:,4]=-(xs.T@(xs*(((0.5*r1*r2)-((rho/unexp)*(r1**2)))[:,None])))/n
        H[:,3,:,4]=-(xs.T@(xs*(((0.5*r1*r2)-((rho/unexp)*(r2**2)))[:,None])))/n
        H[:,4,:,4]=-(xs[ybothnotnan,:].T@(xs[ybothnotnan,:]*(((((((delta**2)+1)/(8*delta))*((r1**2)+(r2**2)))\
                          -((((delta**2)-1)/(4*delta))*(r1*r2)))\
                         -(unexp/4))[ybothnotnan,None])))/n
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
        # if log-likelihood is -np.inf: quit; on a dead track for this SNP
        if np.isinf(logL):
            return param,logL,grad,H,G,converged
        # unpack Hessian to matrix
        UH=H.reshape((ks*5,ks*5))
        # take average of UH and UH.T for numerical stability
        UH=(UH+UH.T)/2
        # get eigenvalue decomposition of minus unpackage Hessian
        (D,P)=np.linalg.eigh(-UH)
        # if lowest eigenvalue too low
        if (D.min()<MINEVALMH):
            # bend s.t. Newton becomes more like gradient descent
            a=(MINEVALMH-D.min())/(1-D.min())
            D=(1-a)*D+a
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
    x1=x[y1notnan,:]
    x2=x[y2notnan,:]
    invXTX1=np.linalg.inv((x1.T@x1)/n)
    invXTX2=np.linalg.inv((x2.T@x2)/n)
    a1=invXTX1@((x1.T@y1[y1notnan])/n)
    a2=invXTX2@((x2.T@y2[y2notnan])/n)
    e1=y1-x@a1
    e2=y2-x@a2
    z1=0.5*np.log(e1**2)
    z2=0.5*np.log(e2**2)
    b1=invXTX1@((x1.T@z1[y1notnan])/n)
    b2=invXTX2@((x2.T@z2[y2notnan])/n)
    r1=e1/np.exp(x@b1)
    r2=e2/np.exp(x@b2)
    rhomean=(np.corrcoef(r1[ybothnotnan],r2[ybothnotnan]))[0,1]
    gc=np.zeros(k)
    gc[0]=np.log((1+rhomean)/(1-rhomean))
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
        raise RuntimeError('Estimates baseline model (=no SNPs) not converged')
    invH0=np.linalg.inv(H0.reshape((k*5,k*5)))
    GGT0=(G0.reshape((k*5,n)))@((G0.reshape((k*5,n))).T)
    param0Var=invH0@GGT0@invH0
    param0SE=((np.diag(param0Var))**0.5).reshape((k,5))
    print('3. ESTIMATION MODELS WITH SNPS')
    snp=np.empty((m,5))*np.nan
    snpSE=np.empty((m,5))*np.nan
    snpLRT=np.empty((m))*np.nan
    snpAPEsig1=np.empty(m)*np.nan
    snpAPEsig1SE=np.empty(m)*np.nan
    snpAPEsig2=np.empty(m)*np.nan
    snpAPEsig2SE=np.empty(m)*np.nan
    snpAPErho=np.empty(m)*np.nan
    snpAPErhoSE=np.empty(m)*np.nan
    for j in tqdm(range(m)):
        param1=np.vstack((param0.copy(),np.zeros((1,5))))
        xs=np.hstack((x,g[:,j][:,None]))
        ks=k+1
        (param1,logL1,grad1,H1,G1,converged1)=Newton(param1,silent=True)
        if converged1:
            invH1=np.linalg.inv(H1.reshape((ks*5,ks*5)))
            GGT1=(G1.reshape((ks*5,n)))@((G1.reshape((ks*5,n))).T)
            param1Var=invH1@GGT1@invH1
            param1SE=((np.diag(param1Var))**0.5).reshape((ks,5))
            b1Var=(param1Var.reshape((ks,5,ks,5)))[:,2,:,2]
            b2Var=(param1Var.reshape((ks,5,ks,5)))[:,3,:,3]
            sig1=np.exp((xs*param1[None,:,2]).sum(axis=1))
            sig2=np.exp((xs*param1[None,:,3]).sum(axis=1))
            snpAPEsig1[j]=param1[-1,2]*sig1[y1notnan].mean()
            snpAPEsig2[j]=param1[-1,3]*sig2[y2notnan].mean()
            deltaAPEsig1=param1[-1,2]*(xs*sig1[:,None])[y1notnan,:].mean(axis=0)
            deltaAPEsig2=param1[-1,3]*(xs*sig2[:,None])[y2notnan,:].mean(axis=0)
            deltaAPEsig1[-1]=sig1[y1notnan].mean()+deltaAPEsig1[-1]
            deltaAPEsig2[-1]=sig2[y2notnan].mean()+deltaAPEsig2[-1]
            snpAPEsig1SE[j]=(deltaAPEsig1@b1Var@deltaAPEsig1)**0.5
            snpAPEsig2SE[j]=(deltaAPEsig2@b2Var@deltaAPEsig2)**0.5
            gcVar=(param1Var.reshape((ks,5,ks,5)))[:,4,:,4]
            delta=np.exp((xs*param1[None,:,4]).sum(axis=1))
            snpAPErho[j]=param1[-1,4]*(2*delta/((delta+1)**2))[ybothnotnan].mean()
            deltaAPErho=2*param1[-1,4]\
                *(xs*(((1-delta)/((1+delta)**3))[:,None]))[ybothnotnan,:].mean(axis=0)
            deltaAPErho[-1]=(2*delta/((delta+1)**2))[ybothnotnan].mean()+deltaAPErho[-1]
            snpAPErhoSE[j]=(deltaAPErho@gcVar@deltaAPErho)**0.5
            snp[j,:]=param1[-1,:]
            snpSE[j,:]=param1SE[-1,:]
            snpLRT[j]=2*n*(logL1-logL0)
        else:
            warnings.warn('Model for SNP '+str(j)+' did not converge')
    snpWald=(snp/snpSE)**2
    snpPWald=1-stats.chi2.cdf(snpWald,1)
    snpPLRT=1-stats.chi2.cdf(snpLRT,5)
    return param0,param0SE,snp,snpSE,snpWald,snpPWald,snpLRT,snpPLRT\
        ,snpAPEsig1,snpAPEsig1SE,snpAPEsig2,snpAPEsig2SE\
            ,snpAPErho,snpAPErhoSE

def SimulateData(seed,h2y1,h2y2,rG,h2sig1,h2sig2,h2rho,rholoc,rhoscale):
    # define globals for data and true parameters
    global n,N,k,ks,y1,y2,y1notnan,y2notnan,ybothnotnan,nboth,x,xs,g,paramtrue,eaf
    # set random-number generator
    rng=np.random.default_rng(seed)
    # draw allele frequencies between (TAUTRUEMAF,1-TAUTRUEMAF)
    f=TAUTRUEMAF+(1-2*TAUTRUEMAF)*rng.uniform(size=m)
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
        notdone=(eaf*(1-eaf))<(TAUDATAMAF*(1-TAUDATAMAF))
        mnotdone=notdone.sum()
    # standardise genotype matrix
    gs=(g-2*(eaf[None,:]))/(((2*eaf*(1-eaf))**0.5)[None,:])
    # draw factors for SNP effects on expectations
    gf1=rng.normal(size=m)
    gf2=rng.normal(size=m)
    # draw correlated SNP effects on expectations
    alpha1=gf1*((h2y1/m)**0.5)
    alpha2=((rG*gf1)+(((1-(rG**2))**0.5)*gf2))*((h2y2/m)**0.5)
    # draw SNP effects on variances and correlation
    beta1=rng.normal(size=m)*((h2sig1/m)**0.5)
    beta2=rng.normal(size=m)*((h2sig2/m)**0.5)
    gamma=rhoscale*(rng.normal(size=m)*((h2rho/m)**0.5))
    # draw error terms for sigma and rho
    esig1=rng.normal(size=n)*((1-h2sig1)**0.5)
    esig2=rng.normal(size=n)*((1-h2sig2)**0.5)
    erho=rhoscale*(rng.normal(size=n)*((1-h2rho)**0.5))
    # calculate standard deviations
    sig1=np.exp(-0.5+esig1+((gs*beta1[None,:]).sum(axis=1)))
    sig2=np.exp(-0.5+esig2+((gs*beta2[None,:]).sum(axis=1)))
    # find intercept for linear part of correlation, such that average
    # correlation equals rholoc
    gamma0=np.log((1+rholoc)/(1-rholoc))
    delta=np.exp(gamma0+erho+((gs*gamma[None,:]).sum(axis=1)))
    rho=(delta-1)/(delta+1)
    # draw noise factors
    eta1=rng.normal(size=n)
    eta2=rng.normal(size=n)
    # scale and mix noise to achieve desired standard deviations and correlations 
    e1=eta1*sig1*((1-h2y1)**0.5)
    e2=((rho*eta1)+(((1-(rho**2))**0.5)*eta2))*sig2*((1-h2y2)**0.5)
    # draw outcomes
    y1=((gs*alpha1[None,:]).sum(axis=1))+e1
    y2=((gs*alpha2[None,:]).sum(axis=1))+e2
    # convert true standardised effects to raw effects, and store
    paramtrue=np.empty((m+1,5))
    paramtrue[1:,0]=alpha1/((2*eaf*(1-eaf))**0.5)
    paramtrue[1:,1]=alpha2/((2*eaf*(1-eaf))**0.5)
    paramtrue[1:,2]=beta1/((2*eaf*(1-eaf))**0.5)
    paramtrue[1:,3]=beta2/((2*eaf*(1-eaf))**0.5)
    paramtrue[1:,4]=gamma/((2*eaf*(1-eaf))**0.5)
    paramtrue[0,:]=-2*((paramtrue[1:,:]*eaf[:,None]).sum(axis=0))
    paramtrue[0,2]=paramtrue[0,2]+0.5*np.log((1-h2y1))-0.5
    paramtrue[0,3]=paramtrue[0,3]+0.5*np.log((1-h2y2))-0.5
    paramtrue[0,4]=paramtrue[0,4]+gamma0
    # randomly set 0% of data as missing
    y1[rng.uniform(size=n)<0]=np.nan
    y2[rng.uniform(size=n)<0]=np.nan
    # keep only observations for whom at least one trait is observed
    atleast1=~(np.isnan(y1)*np.isnan(y2))
    y1=y1[atleast1]
    y2=y2[atleast1]
    g=g[atleast1,:]
    n=atleast1.sum()
    # find indices of non missing observations
    y1notnan=~np.isnan(y1)
    y2notnan=~np.isnan(y2)
    # find total number of observation in multivariate model
    N=(y1notnan.sum())+(y2notnan.sum())
    # find individuals with complete data
    ybothnotnan=y1notnan*y2notnan
    nboth=ybothnotnan.sum()
    # set intercept as baseline model regressor
    x=np.ones((n,1))
    xs=x.copy()
    k=1
    ks=1
    # recalculate empirical AFs
    eaf=g.mean(axis=0)/2

def Test():
    global n,m
    print('1. SIMULATING DATA')
    seed=1873798321
    n=int(5e4)
    m=100
    h2y1=0.4
    h2y2=0.5
    rG=0.9
    h2sig1=0.6
    h2sig2=0.7
    h2rho=0.8
    rholoc=0
    rhoscale=0.5
    SimulateData(seed,h2y1,h2y2,rG,h2sig1,h2sig2,h2rho,rholoc,rhoscale)
    (param0,param0SE,snp,snpSE,snpWald,snpPWald,snpLRT,snpPLRT\
        ,snpAPEsig1,snpAPEsig1SE,snpAPEsig2,snpAPEsig2SE\
            ,snpAPErho,snpAPErhoSE)=GCAT()
    return param0,param0SE,snp,snpSE,snpWald,snpPWald,snpLRT,snpPLRT\
        ,snpAPEsig1,snpAPEsig1SE,snpAPEsig2,snpAPEsig2SE\
            ,snpAPErho,snpAPErhoSE

(param0,param0SE,snp,snpSE,snpWald,snpPWald,snpLRT,snpPLRT\
    ,snpAPEsig1,snpAPEsig1SE,snpAPEsig2,snpAPEsig2SE\
        ,snpAPErho,snpAPErhoSE)=Test()
