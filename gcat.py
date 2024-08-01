import argparse
import logging
import time
import traceback
import os, psutil
import numpy as np
import mmap
import pandas as pd
import linecache
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from scipy import stats
from functools import reduce

# define constants
MAXITER=50
TOL=1E-12
TAUTRUEMAF=0.05
TAUDATAMAF=0.01
MINVAR=0.01
MINEVAL=1E-3
MINN=10
MINM=1
ARMIJO=1E-4
thetainv=2/(1+5**0.5)
oneminthetainv=1-thetainv
RESULTSMBLOCK=2000

# define PLINK binary data variables
extBED='.bed'
extBIM='.bim'
extFAM='.fam'
binBED1=bytes([0b01101100])
binBED2=bytes([0b00011011])
binBED3=bytes([0b00000001])
nperbyte=4
lAlleles=['A','C','G','T']

# define other extensions and prefixes
outDEFAULT='output'
extLOG='.log'
extEFF='.true_effects.txt'
extPHE='.phe'
extFRQ='.frq'
extBASE='.baseline_estimates.txt'
extASSOC='.assoc'

# set separator and end-of-line character
sep='\t'
eol='\n'

# define text for welcom screen
__version__ = 'v0.1'
HEADER = eol
HEADER += '------------------------------------------------------------'+eol
HEADER += '| GCAT (Genome-wide Cross-trait concordance Analysis Tool) |'+eol
HEADER += '------------------------------------------------------------'+eol
HEADER += '| BETA {V}, (C) 2024 Ronald de Vlaming                    |'.format(V=__version__)+eol
HEADER += '| Vrije Universiteit Amsterdam                             |'+eol
HEADER += '| GNU General Public License v3                            |'+eol
HEADER += '------------------------------------------------------------'+eol

def CalcLogL(param,y1,y2,y1notnan,y2notnan,ybothnotnan,n,nboth,N,X,K,mode=3):
    ''''
    Calulcation log-likelihood bivariate model
     Mode: tells function what to calculate and return
       1=logL
       2=logL,gradient
       3=logL,gradient,Hessian
       4=G (iid-specific contribution to grad); at end of Newton algo
       5=G, Hessian; at end of BFGS algo
    '''
    # calculate log-likelihood constant
    cons=N*np.log(2*np.pi)
    # calculate linear parts
    linpart=(X[:,:,None]*param[None,:,:]).sum(axis=1)
    # get sigma1, sigma2, delta, and rho
    sig1=np.exp(linpart[:,2])
    sig2=np.exp(linpart[:,3])
    delta=np.exp(linpart[:,4])
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
        if mode==1:
            return logL
        elif mode==2:
            return logL,None
        elif mode==3:
            return logL,None,None
        elif mode==4:
            return None
        elif mode==5:
            return None,None
    # calculate errors 
    e1=y1-linpart[:,0]
    e2=y2-linpart[:,1]
    # at missing points, set rho and errors to zero, and set delta to one
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
    logdetV=2*nboth*np.log(2)+linpart[ybothnotnan,4].sum()\
        +2*(linpart[y1notnan,2].sum()+linpart[y2notnan,3].sum())\
        -2*((np.log(delta[ybothnotnan]+1)).sum())
    quadratic=(((r1**2)+(r2**2)-2*(rho*r1*r2))/unexp).sum()
    # calculate logL/n
    logL=-0.5*(cons+logdetV+quadratic)/n
    # if just interested in logL
    if mode==1:
        return logL
    else:
        # initialise weights matrix for gradient
        Wg=np.empty((nPLINK,5))
        # calculate weights matrix for gradient
        Wg[:,0]=((r1/sig1)-(rho*r2/sig1))/unexp
        Wg[:,1]=((r2/sig2)-(rho*r1/sig2))/unexp
        Wg[:,2]=(((r1**2)-(rho*r1*r2))/unexp)-1
        Wg[:,3]=(((r2**2)-(rho*r1*r2))/unexp)-1
        Wg[:,4]=(0.5*rho-0.5*((((delta**2)-1)/(4*delta))*(r1**2+r2**2)-(((delta**2+1)/(2*delta))*r1*r2)))
        # set gradient=0 w.r.t. beta1 for missing y1 and idem w.r.t. beta2 for missing y2
        Wg[~y1notnan,2]=0
        Wg[~y2notnan,3]=0
        # if just interested in G or H
        if mode==4 or mode==5:
            # calculate individual-specific contribution to gradient/n as 3d array
            G=((X.T[:,None,:])*(Wg.T[None,:,:]))/n
            # if just interest in G, return that
            if mode==4:
                return G
        else:
            # calculate gradient
            grad=(X.T@Wg)/n
            # if just interest in logL and grad, return those
            if mode==2:
                return logL,grad
        # initialise weights array Hessian (nPLINK-by-5-by-5)
        wH=np.empty((nPLINK,5,5))
        # calculate weights array for Hessian
        wH[:,0,0]=-1/(unexp*(sig1**2))
        wH[:,1,1]=-1/(unexp*(sig2**2))
        wH[:,0,1]=rho/(unexp*sig1*sig2)
        wH[:,0,2]=(1/(sig1*unexp))*(rho*r2-2*r1)
        wH[:,1,3]=(1/(sig2*unexp))*(rho*r1-2*r2)
        wH[:,0,3]=(1/(sig1*unexp))*(rho*r2)
        wH[:,1,2]=(1/(sig2*unexp))*(rho*r1)
        wH[:,0,4]=(1/(sig1*unexp))*(rho*r1-((1+(rho**2))*(r2/2)))
        wH[:,1,4]=(1/(sig2*unexp))*(rho*r2-((1+(rho**2))*(r1/2)))
        wH[:,2,2]=-(1/unexp)*(2*(r1**2)-rho*r1*r2)
        wH[:,3,3]=-(1/unexp)*(2*(r2**2)-rho*r1*r2)
        wH[:,2,3]=-(1/unexp)*(-rho*r1*r2)
        wH[:,2,4]=-((0.5*r1*r2)-((rho/unexp)*(r1**2)))
        wH[:,3,4]=-((0.5*r1*r2)-((rho/unexp)*(r2**2)))
        wH[:,4,4]=-((((((delta**2)+1)/(8*delta))*((r1**2)+(r2**2)))-((((delta**2)-1)/(4*delta))*(r1*r2)))-(unexp/4))
        # set weight w.r.t. gamma twice to zero when either y1 and/or y2 is missing
        wH[~ybothnotnan,4,4]=0
        # initialise Hessian/n
        H=np.empty((K,5,K,5))
        # use symmetry to fill gaps in weights
        for i in range(5):
            for j in range(i,5):
                # calculate Hessian/n
                H[:,i,:,j]=(X.T@(X*wH[:,None,i,j]))/n
                if j>i:
                    # use symmetry to find out counterparts
                    H[:,j,:,i]=H[:,i,:,j]
        # if just interest in G and H, return that
        if mode==5:
            return G,H
        else:
            return logL,grad,H

def Newton(param,y1,y2,y1notnan,y2notnan,ybothnotnan,n,nboth,N,g=None,gisnan=None,silent=False,linesearch=False):
    # if genotype vector provided, calculate some key ingredients
    if g is not None:
        # combine into grand X matrix
        X=np.hstack((x,g[:,None]))
        # ensure rows of X are zero for observations where genotype is missing
        X[gisnan,:]=0
        # set number of regressors to no. of control variables + 1 (for the SNP)
        K=k+1
    else:
        # otherwise use version without SNP as grand X
        X=x
        K=k
    # set iteration counter to zero and convergence to false
    i=0
    converged=False
    # while not converged and MAXITER not reached
    while not(converged) and i<MAXITER:
        # calculate log-likelihood, its gradient, and Hessian
        (logL,grad,H)=CalcLogL(param,y1,y2,y1notnan,y2notnan,ybothnotnan,n,nboth,N,X,K)
        # if log-likelihood is -np.inf: quit; on a dead track for this SNP
        if np.isinf(logL):
            D=[0]
            G=[0]
            return param,logL,grad,H,G,D,converged
        # unpack Hessian to matrix
        UH=H.reshape((K*5,K*5))
        # take average of UH and UH.T for numerical stability
        UH=(UH+UH.T)/2
        # get eigenvalue decomposition of minus unpackage Hessian
        (D,P)=np.linalg.eigh(-UH)
        # if lowest eigenvalue too low
        if min(D)<MINEVAL:
            # bend s.t. Newton becomes more like gradient descent
            a=(MINEVAL-D.min())/(1-D.min())
            Dadj=(1-a)*D+a
        else:
            Dadj=D
        # get Newton-Raphson update vector
        update=P@((((grad.reshape((K*5,1))).T@P)/Dadj).T)
        # calculate convergence criterion
        msg=(update*grad.reshape((K*5,1))).sum()
        # if convergence criterion met
        if msg<TOL:
            # set convergence to true and calculate sampling variance
            converged=True
        else:
            if linesearch:
                # perform golden section to get new parameters estimates
                (param,j,step)=GoldenSection(param,logL,grad,update.reshape((K,5)),y1,y2,y1notnan,y2notnan,ybothnotnan,n,nboth,N,X,K)
            else:
                # just apply Newton step directly
                param+=update.reshape((K,5))
            # update iteration counter
            i+=1
            # print update if not silent
            if not(silent):
                if linesearch:
                    logger.info('Newton iteration '+str(i)+': logL='+str(logL)+'; '\
                        +str(j)+' line-search steps, yielding step size = '+str(step))
                else:
                    logger.info('Newton iteration '+str(i)+': logL='+str(logL))
    # calculate individual-specific contributions to gradient/n
    G=CalcLogL(param,y1,y2,y1notnan,y2notnan,ybothnotnan,n,nboth,N,X,K,mode=4)
    return param,logL,grad,H,G,D,converged

def GoldenSection(param,logL,grad,update,y1,y2,y1notnan,y2notnan,ybothnotnan,n,nboth,N,X,K):
    # calculate update'grad
    utg=(grad*update).sum()
    # initialise parameters at various points along interval
    param1=param
    param2=param+oneminthetainv*update
    param3=param+thetainv*update
    param4=param+update
    # set corresponding step sizes
    step1=0
    step2=oneminthetainv
    step3=thetainv
    step4=1
    # set iteration counter to one and convergence to false
    i=1
    converged=False
    # calculate log likelihood at right, mid-left and mid-right
    logL4=CalcLogL(param4,y1,y2,y1notnan,y2notnan,ybothnotnan,n,nboth,N,X,K,mode=1)
    # directly try Armijo's rule perform actually performing section search
    if logL4>=logL+ARMIJO*step4*utg:
        return param4,i,step4
    # calculate remaining log likelihoods
    logL1=logL
    logL2=CalcLogL(param2,y1,y2,y1notnan,y2notnan,ybothnotnan,n,nboth,N,X,K,mode=1)
    logL3=CalcLogL(param3,y1,y2,y1notnan,y2notnan,ybothnotnan,n,nboth,N,X,K,mode=1)
    # while not converged and MAXITER not reached
    while not(converged) and i<MAXITER:
        # update iteration counter
        i+=1
        #if mid-left val >= mid-right val: set mid-right as right
        if logL2>=logL3: 
            # set parameters accordingly
            param4=param3
            param3=param2
            param2=thetainv*param1+oneminthetainv*param4
            # set step sizes accordingly
            step4=step3
            step3=step2
            step2=thetainv*step1+oneminthetainv*step4
            # calculate log likelihood at new mid-left and mid-right
            logL4=logL3
            logL3=logL2
            logL2=CalcLogL(param2,y1,y2,y1notnan,y2notnan,ybothnotnan,n,nboth,N,X,K,mode=1)
        #if mid-right val > mid-left val: set mid-left as left
        else:
            # set parameters accordingly
            param1=param2
            param2=param3
            param3=thetainv*param4+oneminthetainv*param1
            # set step sizes accordingly
            step1=step2
            step2=step3
            step3=thetainv*step4+oneminthetainv*step1
            # calculate log likelihood at new mid-left and mid-right
            logL1=logL2
            logL2=logL3
            logL3=CalcLogL(param3,y1,y2,y1notnan,y2notnan,ybothnotnan,n,nboth,N,X,K,mode=1)
        # inexact convergence if Armijo's rule satisfied
        if logL4>=logL+ARMIJO*step4*utg:
            return param4,i,step4
    return param4,i,step4

def InitialiseParams(y1,y2,y1notnan,y2notnan,ybothnotnan,n1,n2):
    # get x for observations where y1 resp. y2 is not missing
    x1=x[y1notnan,:]
    x2=x[y2notnan,:]
    # calculate inv(X'X) for these two subsets of observations
    invXTX1=np.linalg.inv((x1.T@x1)/n1)
    invXTX2=np.linalg.inv((x2.T@x2)/n2)
    # calculate coefficients for baseline regressors w.r.t. E[y1] and E[y2]
    a1=invXTX1@((x1.T@y1[y1notnan])/n1)
    a2=invXTX2@((x2.T@y2[y2notnan])/n2)
    # get regression residuals e1 and e2
    e1=y1-x@a1
    e2=y2-x@a2
    # take transformation f() of residuals, such that under GCAT's DGP
    # E[f(e)] is linear function of regressors
    z1=0.5*np.log(e1**2)
    z2=0.5*np.log(e2**2)
    # get regression coefficients for baseline regressors w.r.t. Var(y1) and Var(y2)
    b1=invXTX1@((x1.T@z1[y1notnan])/n1)
    b2=invXTX2@((x2.T@z2[y2notnan])/n2)
    # rescale residuals based on implied standard deviation
    r1=e1/np.exp(x@b1)
    r2=e2/np.exp(x@b2)
    # calculate correlation between rescaled residuals
    rhomean=(np.corrcoef(r1[ybothnotnan],r2[ybothnotnan]))[0,1]
    # set intercept for gamma s.t. Corr(y1,y2)=rhomean for all individuals
    gc=np.zeros(k)
    gc[0]=np.log((1+rhomean)/(1-rhomean))
    # collect and return initialised values
    param0=np.vstack((a1,a2,b1,b2,gc)).T
    return param0

def GCAT():
    # initialise control variables (and some derived variables) and number of people in PLINK file as globals
    global x, k, nPLINK
    # print update
    logger.info('Reading data')
    # count number of SNPs from bim
    M=CountLines(args.bfile+extBIM)
    logger.info('Found '+str(M)+ ' SNPs in '+args.bfile+extBIM)
    # initialise starting point and end point to analyses
    Mstart=1
    Mend=M
    # if SNP range has been provided
    if args.snp is not None:
        Mstart=args.snp[0]
        Mend=args.snp[1]
        if Mstart>M:
            raise ValueError('Index for first SNP to analyse based on option --snp exceeds number of SNPs in data')
        if Mend>M:
            raise ValueError('Index for last SNP to analyse based on option --snp exceeds number of SNPs in data')
    # read fam
    famdata=pd.read_csv(args.bfile+extFAM,sep=sep,header=None,names=['FID','IID','PID','MID','SEX','PHE'])
    nPLINK=famdata.shape[0]
    logger.info('Found '+str(nPLINK)+' individuals in '+args.bfile+extFAM)
    # retain only FID and IID of fam data
    DATA=famdata.iloc[:,[0,1]]
    # read pheno file
    ydata=pd.read_csv(args.pheno,sep=sep)
    # check two phenotypes in phenotype data
    if ydata.shape[1]!=4:
        raise ValueError(args.pheno+' does not contain exactly two phenotypes')
    # get phenotype labels
    y1label=ydata.columns[2]
    y2label=ydata.columns[3]
    # print update
    logger.info('Found '+str(ydata.shape[0])+' individuals and two traits, labelled '\
                 +y1label+' and '+y2label+', in '+args.pheno)
    # left join FID,IID from fam with pheno data
    DATA=pd.merge(left=DATA,right=ydata,how='left',left_on=['FID','IID'],right_on=['FID','IID'])
    # retrieve matched phenotypes baseline model
    y1=DATA.values[:,2]
    y2=DATA.values[:,3]
    # if covars provideded
    if covars:
        # read covars
        xdata=pd.read_csv(args.covar,sep=sep)
        # if no covariates found, throw error
        if xdata.shape[1]<3:
            raise ValueError(args.covar+' does not contain data on any covariates')
        # print update
        logger.info('Found '+str(xdata.shape[0])+' individuals and '\
                     +str(xdata.shape[1]-2)+' control variables in '+args.covar)
        # left joint data with covariates
        DATA=pd.merge(left=DATA,right=xdata,how='left',left_on=['FID','IID'],right_on=['FID','IID'])
        # retrieve matched covariates baseline model
        x=np.hstack((np.ones((nPLINK,1)),DATA.values[:,4:]))
        xlabels=['intercept']+xdata.iloc[:,2:].columns.to_list()
        k=x.shape[1]
        # find observations where at least one covariate is missing
        xmissing=np.isnan(x.sum(axis=1))
        # count number of observations with nonmissing data on covariates; throw error if zero
        nxnotnan=(~xmissing).sum()
        if nxnotnan==0:
            raise ValueError('No observations in '+args.covar\
                            +' without any missingness that can be matched to '+args.bfile+extFAM)
        # set y1,y2 to missing for rows where any covariate is missing, and set x to zero
        x[xmissing,:]=0
        y1[xmissing]=np.nan
        y2[xmissing]=np.nan
        # calculate lowest eigenvalue of X'X for observations without any missingness
        (Dxtx,_)=np.linalg.eigh((x.T@x)/nxnotnan)
        # if too low, throw error:
        if min(Dxtx)<MINEVAL:
            raise ValueError('Regressors in '+args.covar+' have too much multicollinearity. '\
                            +'Did you add an intercept to your covariates file? Please remove it. '\
                            +'Or do you have set of dummies that is perfectly collinear with intercept? Please remove one category. '\
                            +'Recall: GCAT always adds intercept to model!')
    else: # else set intercept as only covariate
        k=1
        x=np.ones((nPLINK,1))
        xlabels=['intercept']
    # find indices of non missing observations
    y1notnan=~np.isnan(y1)
    y2notnan=~np.isnan(y2)
    # count number of complete observations per trait
    n1=y1notnan.sum()
    n2=y2notnan.sum()
    # report stats
    logger.info('Found '+str(n1)+' observations for '+ydata.columns[2]\
                +' with complete data (i.e. in PLINK files, and with all covariates, if any, nonmissing)')
    logger.info('Found '+str(n2)+' observations for '+ydata.columns[3]\
                +' with complete data') 
    # find total number of observation in multivariate model
    N=n1+n2
    # find individuals with complete data for x, y1, y2, and that can be matched to genotypes
    ybothnotnan=y1notnan&y2notnan
    nboth=ybothnotnan.sum()
    # find individuals with either y1 or y2 complete: count of that = no. of independent observations!
    y1ory2notnan=y1notnan|y2notnan
    n=y1ory2notnan.sum()
    # initialise parameters baseline model
    logger.info('Initialising baseline model (i.e. without any SNPs)')
    param0=InitialiseParams(y1,y2,y1notnan,y2notnan,ybothnotnan,n1,n2)
    # estimate baseline model
    logger.info('Estimating baseline model')
    (param0,logL0,_,_,_,_,converged0)=Newton(param0,y1,y2,y1notnan,y2notnan,ybothnotnan,n,nboth,N,linesearch=True)
    # write baseline model estimates to output file
    pd.DataFrame(param0,columns=['ALPHA1','ALPHA2','BETA1','BETA2','GAMMA'],\
                 index=xlabels).to_csv(args.out+extBASE,sep=sep)                     
    # if baseline model did not converge: throw error
    if not(converged0):
        raise RuntimeError('Estimates baseline model (=no SNPs) not converged')
    # count number of full bytes per SNP
    nb=int(nPLINK/nperbyte)
    # compute how many individuals in remainder byte per SNP
    nr=nPLINK%nperbyte
    # compute total bytes per SNP (full + remainder, if any)
    nbt=nb+(nr>0)
    # compute expected number of bytes in .bed file: 3 magic bytes + data
    expectedbytes=3+nbt*M
    # get observed number of bytes in .bed file
    observedbytes=(os.stat(args.bfile+extBED)).st_size
    # throw error if mismatch
    if observedbytes>expectedbytes:
        raise ValueError('More bytes in '+args.bfile+extBED+' than expected. File corrupted?')
    elif observedbytes<expectedbytes:
        raise ValueError('Fewer bytes in '+args.bfile+extBED+' than expected. File corrupted?')
    # print update
    logger.info('Estimating model for each SNP in '+args.bfile+extBED)
    # compute rounded n (i.e. empty including empty bits)
    roundedn=nbt*nperbyte
    # get rowid of first two bits per byte being read
    ids=nperbyte*np.arange(nbt)
    # connect to bed file
    connbed=open(args.bfile+extBED,'rb')
    # check if first three bytes bed file are correct
    if ord(connbed.read(1))!=(ord(binBED1)) or ord(connbed.read(1))!=(ord(binBED2)) or ord(connbed.read(1))!=(ord(binBED3)):
        raise ValueError(args.bfile+extBED+' not a valid PLINK .bed file')
    # close connection to bed file
    connbed.close()
    # connect to write association results file
    connassoc=open(args.out+extASSOC,'w')
    # write first line to results file
    connassoc.write('CHROMOSOME'+sep\
                    +'SNP_ID'+sep\
                    +'BASELINE_ALLELE'+sep\
                    +'BASELINE_ALLELE_FREQ'+sep\
                    +'EFFECT_ALLELE'+sep\
                    +'EFFECT_ALLELE_FREQ'+sep\
                    +'HWE_PVAL'+sep\
                    +'N_GENOTYPED'+sep\
                    +'N_'+y1label+'_COMPLETE'+sep\
                    +'N_'+y2label+'_COMPLETE'+sep\
                    +'NBOTHCOMPLETE'+sep\
                    +'LRT'+sep\
                    +'P_LRT'+sep\
                    +'ESTIMATE_ALPHA1'+sep\
                    +'SE_ALPHA1'+sep\
                    +'WALD_ALPHA1'+sep\
                    +'P_ALPHA1'+sep\
                    +'ESTIMATE_ALPHA2'+sep\
                    +'SE_ALPHA2'+sep\
                    +'WALD_ALPHA2'+sep\
                    +'P_ALPHA2'+sep\
                    +'ESTIMATE_BETA1'+sep\
                    +'SE_BETA1'+sep\
                    +'WALD_BETA1'+sep\
                    +'P_BETA1'+sep\
                    +'APE_STDEV_'+y1label+sep\
                    +'SE_APE_STDEV_'+y1label+sep\
                    +'ESTIMATE_BETA2'+sep\
                    +'SE_BETA2'+sep\
                    +'WALD_BETA2'+sep\
                    +'P_BETA2'+sep\
                    +'APE_STDEV_'+y2label+sep\
                    +'SE_APE_STDEV_'+y1label+sep\
                    +'ESTIMATE_GAMMA'+sep\
                    +'SE_GAMMA'+sep\
                    +'WALD_GAMMA'+sep\
                    +'P_GAMMA'+sep\
                    +'APE_CORR_'+y1label+'_'+y2label+sep\
                    +'SE_APE_CORR_'+y1label+'_'+y2label+eol)
    # calculate how many SNPs to analyse total and how many complete output blocks
    Mt=Mend+1-Mstart
    Bc=int(Mt/RESULTSMBLOCK)
    # calculate how many SNPs in remainder block and how many output block in total
    Mr=Mt%RESULTSMBLOCK
    Bt=Bc+(Mr>0)
    # set lambda function for analysis per SNP
    analysej=lambda j:AnalyseOneSNP(pbar,j,nbt,roundedn,ids,param0,logL0\
                                    ,y1,y2,y1notnan,y2notnan,ybothnotnan,y1ory2notnan,n,nboth,N)
    # initialise progress bar
    pbar=tqdm(total=Mt)
    # for each block
    for b in range(Bt):
        # get start and ending SNP
        m0=Mstart+(b*RESULTSMBLOCK)
        m1=min(Mend,Mstart+((b+1)*RESULTSMBLOCK)-1)
        # using parallel execution in block with writing at end of block
        with ThreadPoolExecutor() as executor:
            # analyse snp j
            outputlines=executor.map(analysej,[j for j in range(m0,m1+1)])
        for outputline in outputlines:
            connassoc.write(outputline)        
    # close progress bar
    pbar.close()
    # close connections to assoc file
    connassoc.close()

def AnalyseOneSNP(pbar,j,nbt,roundedn,ids,param0,logL0\
                  ,y1,y2,y1notnan,y2notnan,ybothnotnan,y1ory2notnan,n,nboth,N):
    # connect to bed file
    connbed=open(args.bfile+extBED,'rb')
    # go to starting point of jth SNP in BED file
    offset=3+(nbt*(j-1))
    connbed.seek(offset,0)
    # read bytes
    gbytes=np.frombuffer(connbed.read(nbt),dtype=np.uint8)
    # close connection to bed file
    connbed.close()
    # initialise genotypes for this read as empty
    g=np.empty(roundedn,dtype=np.uint8)
    # per individual in each byte
    for i in range(nperbyte):
        # take difference between what is left of byte after removing 2 bits
        gbytesleft=gbytes>>2
        g[ids[0:nbt]+i]=gbytes-(gbytesleft<<2)
        # keep part of byte that is left
        gbytes=gbytesleft
    # drop rows corresponding to empty bits of last byte for each SNP
    g=g[0:nPLINK]
    # find rows with missing genotype
    gisnan=(g==1)
    # set missing values to zero
    g[gisnan]=0
    # recode genotype, where 0=homozygote A1, 1=heterozygote, 2=homozygote A2
    g[g==2]=1
    g[g==3]=2
    # count number of nonmissing genotypes
    ngeno=(~gisnan).sum()
    # initialise empirical allele frequency and HWE pval as NaN
    eaf=np.nan
    hweP=np.nan
    # if at least 1 nonmissing genotype
    if ngeno>0:
        # calculate empirical frequency
        eaf=((g[~gisnan]).mean())/2
        # if empirical frequency is not precisely zero or one
        if (eaf*(1-eaf))!=0:
            # calculate counts of homozygotes and heterozygotes
            n0=(g[~gisnan]==0).sum()
            n1=(g[~gisnan]==1).sum()
            n2=(g[~gisnan]==2).sum()
            # calculate expected counts
            en0=((1-eaf)**2)*ngeno
            en1=(2*eaf*(1-eaf))*ngeno
            en2=(eaf**2)*ngeno
            # calculate HWE test stat
            hwe=(((n0-en0)**2)/en0)+(((n1-en1)**2)/en1)+(((n2-en2)**2)/en2)
            hweP=1-stats.chi2.cdf(hwe,1)
    # initialise SNP effects at zero
    param1=np.vstack((param0.copy(),np.zeros((1,5))))
    # make necessary copies of data for SNP-specific analysis
    y1s=y1.copy()
    y2s=y2.copy()
    y1notnans=y1notnan.copy()
    y2notnans=y2notnan.copy()
    ybothnotnans=ybothnotnan.copy()
    y1ory2notnans=y1ory2notnan.copy()
    # set y1 and y2 to missing for individuals with missing genotype
    y1s[gisnan]=np.nan
    y2s[gisnan]=np.nan
    y1notnans[gisnan]=False
    y2notnans[gisnan]=False
    ybothnotnans[gisnan]=False
    y1ory2notnans[gisnan]=False
    # calculate corresponding SNP-specific sample sizes
    n1s=y1notnans.sum()
    n2s=y2notnans.sum()
    nboths=ybothnotnans.sum()
    ns=y1ory2notnans.sum()
    Ns=n1s+n2s
    # apply Newton's method, provided nboth>=MINN
    if nboths>=MINN:
        (param1,logL1,grad1,H1,G1,D1,converged1)=Newton(param1,y1s,y2s,y1notnans,y2notnans,ybothnotnans,ns,nboths,Ns,g=g,gisnan=gisnan,silent=True)
    else: # else don't even try
        (param1,logL1,grad1,H1,G1,D1,converged1)=(None,None,None,None,None,[0],False)
    # calculate and store estimates, standard errors, etc.
    outputline=CalculateStats(ngeno,eaf,hweP,param1,logL1,logL0,H1,G1,D1,converged1,j\
                              ,y1notnans,y2notnans,ybothnotnans,n1s,n2s,ns,nboths,Ns,g,gisnan)
    # update progress bar
    pbar.update(1)
    # return output line with results
    return outputline

def CalculateStats(ngeno,eaf,hweP,param1,logL1,logL0,H1,G1,D1,converged1,j\
                   ,y1notnans,y2notnans,ybothnotnans,n1s,n2s,ns,nboths,Ns,g,gisnan):
    # combine SNP and control variables into grand X matrix
    X=np.hstack((x,g[:,None]))
    # ensure rows of X are zero for observations where genotype is missing
    X[gisnan,:]=0
    # set number of regressors to no. of control variables + 1 (for the SNP)
    K=k+1
    # read line from bim file, strip trailing newline, split by tabs
    snpline=linecache.getline(args.bfile+extBIM,j).rstrip(eol).split(sep)
    # get chromosome number, snp ID, baseline allele, and effect allele
    snpchr=snpline[0]
    snpid=snpline[1]
    snpbaseallele=snpline[4]
    snpeffallele=snpline[5]
    # build up line to write
    outputline=snpchr+sep+snpid+sep+snpbaseallele+sep+str(1-eaf)+sep\
               +snpeffallele+sep+str(eaf)+sep+str(hweP)+sep+str(ngeno)+sep\
               +str(n1s)+sep+str(n2s)+sep+str(nboths)
    # define sequence of NaNs for missing stuff, if any
    nanfield=sep+'nan'
    nanfields=28*nanfield
    # if converged and Hessian pd, calculate stats and write to assoc file
    if converged1 and min(D1)>MINEVAL:
        invH1=np.linalg.inv(H1.reshape((K*5,K*5)))
        GGT1=(G1.reshape((K*5,G1.shape[2])))@((G1.reshape((K*5,G1.shape[2]))).T)
        param1Var=invH1@GGT1@invH1
        param1SE=((np.diag(param1Var))**0.5).reshape((K,5))
        # calculate average partial effect on expectations, stdevs and correlation
        b1Var=(param1Var.reshape((K,5,K,5)))[:,2,:,2]
        b2Var=(param1Var.reshape((K,5,K,5)))[:,3,:,3]
        sig1=np.exp((X*param1[None,:,2]).sum(axis=1))
        sig2=np.exp((X*param1[None,:,3]).sum(axis=1))
        snpAPEsig1=param1[-1,2]*sig1[y1notnans].mean()
        snpAPEsig2=param1[-1,3]*sig2[y2notnans].mean()
        deltaAPEsig1=param1[-1,2]*(X*sig1[:,None])[y1notnans,:].mean(axis=0)
        deltaAPEsig2=param1[-1,3]*(X*sig2[:,None])[y2notnans,:].mean(axis=0)
        deltaAPEsig1[-1]=sig1[y1notnans].mean()+deltaAPEsig1[-1]
        deltaAPEsig2[-1]=sig2[y2notnans].mean()+deltaAPEsig2[-1]
        snpAPEsig1SE=(deltaAPEsig1@b1Var@deltaAPEsig1)**0.5
        snpAPEsig2SE=(deltaAPEsig2@b2Var@deltaAPEsig2)**0.5
        gcVar=(param1Var.reshape((K,5,K,5)))[:,4,:,4]
        delta=np.exp((X*param1[None,:,4]).sum(axis=1))
        snpAPErho=param1[-1,4]*(2*delta/((delta+1)**2))[ybothnotnans].mean()
        deltaAPErho=2*param1[-1,4]\
            *(X*(((1-delta)/((1+delta)**3))[:,None]))[ybothnotnans,:].mean(axis=0)
        deltaAPErho[-1]=(2*delta/((delta+1)**2))[ybothnotnans].mean()+deltaAPErho[-1]
        snpAPErhoSE=(deltaAPErho@gcVar@deltaAPErho)**0.5
        # get SNP effect, standard error, inferences
        snp=param1[-1,:]
        snpSE=param1SE[-1,:]
        snpLRT=2*ns*(logL1-logL0)
        snpWald=(snp/snpSE)**2
        snpPWald=1-stats.chi2.cdf(snpWald,1)
        snpPLRT=1-stats.chi2.cdf(snpLRT,5)
        # add LRT results to line
        outputline+=sep+str(snpLRT)+sep+str(snpPLRT)
        # add results for effect on E[Y1] to line
        outputline+=sep+str(snp[0])+sep+str(snpSE[0])+sep+str(snpWald[0])+sep+str(snpPWald[0])
        # add results for effect on E[Y2] to line
        outputline+=sep+str(snp[1])+sep+str(snpSE[1])+sep+str(snpWald[1])+sep+str(snpPWald[1])
        # add results for effect on Stdev(Y1) to line
        outputline+=sep+str(snp[2])+sep+str(snpSE[2])+sep+str(snpWald[2])+sep+str(snpPWald[2])\
                    +sep+str(snpAPEsig1)+sep+str(snpAPEsig1SE)
        # add results for effect on Stdev(Y2) to line
        outputline+=sep+str(snp[3])+sep+str(snpSE[3])+sep+str(snpWald[3])+sep+str(snpPWald[3])\
                    +sep+str(snpAPEsig2)+sep+str(snpAPEsig2SE)
        # add results for effect on Stdev(Y2) to line
        outputline+=sep+str(snp[4])+sep+str(snpSE[4])+sep+str(snpWald[4])+sep+str(snpPWald[4])\
                    +sep+str(snpAPErho)+sep+str(snpAPErhoSE)
    else:
        # if model not converged: set NaNs as SNP results
        outputline+=nanfields
    # add eol to line
    outputline+=eol
    # return output line
    return outputline

def SimulateG():
    # get n and M
    n=args.n
    M=args.m
    # initialise random-numer generator
    rng=np.random.default_rng(args.seed_geno)
    # give update
    logger.info('Simulating data on '+str(M)+' SNPs for '+str(n)+' individuals,')
    logger.info('exporting to PLINK binary files '+args.out+extBED+','+extBIM+','+extFAM)
    logger.info('and writing allele frequencies to '+args.out+extFRQ)
    # set FIDs/IIDs as numbers from 1 to n, set PID and MID to zeroes,
    # set sex (=1 or 2) as random draw, set phenotype as missing
    FAM=np.zeros((n,6))
    FAM[:,0]=1+np.arange(n)
    FAM[:,1]=FAM[:,0]
    FAM[:,4]=np.ones(n)+(rng.uniform(size=n)>0.5)
    FAM[:,5]=-9*np.ones(n)
    np.savetxt(args.out+extFAM,FAM,fmt='%i\t%i\t%i\t%i\t%i\t%i')
    # open connection for writing PLINK bed file
    connbed=open(args.out+extBED,'wb')
    connbed.write(binBED1)
    connbed.write(binBED2)
    connbed.write(binBED3)
    # open connection to bim file
    connbim=open(args.out+extBIM,'w')
    # open connection for writing frequency file
    connfrq=open(args.out+extFRQ,'w')
    connfrq.write('CHR'+sep+'SNP'+sep+'A1'+sep+'A2'+sep+'AF1'+sep+'AF2'+eol)
    # calculate how many SNPs can be simulated at once (at least 1)
    Mperb=max(int(dimg/n),1)
    logger.info('Simulating SNPs and writing .bim file in blocks of '+str(Mperb)+' SNPs')
    # count modulo(m,#SNPs per block)
    MR=M%Mperb
    # count number of blocks (i.e. complete + remainder block if any)
    B=int(M/Mperb)+(MR>0)
    # count number of full bytes per SNP
    nb=int(n/nperbyte)
    # compute how many individuals in remainder byte per SNP
    nr=n%nperbyte
    # compute total export bytes per SNP (full + remainder, if any)
    nbt=nb+(nr>0)
    # compute number of observations that would still fit in remainder byte
    nleft=nbt*nperbyte-n
    # set counter for total number of SNPs handled for export to .bim
    i=0
    # for each blok
    for b in tqdm(range(B)):
        # find index for first SNP and last SNP in block
        m0=b*Mperb
        m1=min(M,(b+1)*Mperb)
        # count number of SNP in this block
        m=m1-m0
        # draw allele frequencies between (TAUTRUEMAF,1-TAUTRUEMAF)
        f=TAUTRUEMAF+(1-2*TAUTRUEMAF)*rng.uniform(size=m)
        # initialise genotype matrix and empirical AF
        g=np.zeros((n,m),dtype=np.uint8)
        eaf=np.zeros(m)
        # set number of SNPs not done drawing yet to m in this block
        notdone=(np.ones(m)==1)
        mnotdone=notdone.sum()
        # while number of SNPs not done is at least 1
        while mnotdone>0: 
            # draw as many biallelic SNPs
            u=rng.uniform(size=(n,mnotdone))
            thisg=np.ones((n,mnotdone),dtype=np.uint8)
            thisg[u<(((1-f[notdone])**2)[None,:])]=0
            thisg[u>((1-(f[notdone]**2))[None,:])]=2
            g[:,notdone]=thisg
            # calculate empirical AF
            eaf[notdone]=thisg.mean(axis=0)/2
            # find SNPs with insufficient variation
            notdone=(eaf*(1-eaf))<(TAUDATAMAF*(1-TAUDATAMAF))
            mnotdone=notdone.sum()
            # reshape and recode genotype matrix for export
        # 0=0 alleles; 2=1 allele; 3=2 alleles; 1=missing
        g=np.vstack((2*g,np.zeros((nleft,m),dtype=np.uint8)))
        g[g==4]=3
        # within each byte: 2 bits per individual; 4 individuals per byte in total
        base=np.array([2**0,2**2,2**4,2**6]*nbt,dtype=np.uint8)
        # per SNP, per byte: aggregate across individuals in that byte
        exportbytes=(g*base[:,None]).reshape(nbt,nperbyte,m).sum(axis=1).astype(np.uint8)
        # write bytes
        connbed.write(bytes(exportbytes.T.ravel()))
        # for each SNP in this block
        for j in range(m):
            # update counter
            i+=1
            # draw two alleles without replacement from four possible alleles
            A1A2=rng.choice(lAlleles,size=2,replace=False)
            # write line of .bim file
            connbim.write('0'+sep+'rs'+str(i)+sep+'0'+sep+str(j)+sep+A1A2[0]+sep+A1A2[1]+eol)
            # write line of .frq file
            connfrq.write('0'+sep+'rs'+str(i)+sep+A1A2[0]+sep+A1A2[1]+sep+str(1-eaf[j])+sep+str(eaf[j])+eol)
    # close connections
    connbed.close()
    connbim.close()
    connfrq.close()
    # store prefix of just generated PLINK binary files
    args.bfile=args.out

def CountLines(filename):
    with open(filename, "r+") as f:
        buffer=mmap.mmap(f.fileno(), 0)
        lines=0
        readline=buffer.readline
        while readline():
            lines+=1
    return lines

def SimulateY():
    # initialise random-numer generators
    rng=np.random.default_rng(args.seed_pheno)
    rngeffects=np.random.default_rng(args.seed_effects)
    # print update
    logger.info('Reading PLINK binary files '+args.bfile+extBED+','+extBIM+','+extFAM)
    # get n and M
    n=CountLines(args.bfile+extFAM)
    M=CountLines(args.bfile+extBIM)
    # initialise linear parts of expectations, variances, and correlation
    xalpha1=np.zeros(n)
    xalpha2=np.zeros(n)
    xbeta1=np.zeros(n)
    xbeta2=np.zeros(n)
    xgamma=np.zeros(n)
    # print update
    logger.info('Found '+str(n)+' individuals in '+args.bfile+extFAM)
    logger.info('Found '+str(M)+' SNPs in '+args.bfile+extBIM)
    # calculate how many SNPs can be read at once (at least 1)
    Mperb=max(int(dimg/n),1)
    # count modulo(m,#SNPs per block)
    MR=M%Mperb
    # count number of blocks (i.e. complete + remainder block if any)
    B=int(M/Mperb)+(MR>0)
    # count number of full bytes per SNP
    nb=int(n/nperbyte)
    # compute how many individuals in remainder byte per SNP
    nr=n%nperbyte
    # compute total bytes per SNP (full + remainder, if any)
    nbt=nb+(nr>0)
    # compute expected number of bytes in .bed file: 3 magic bytes + data
    expectedbytes=3+nbt*M
    # get observed number of bytes in .bed file
    observedbytes=(os.stat(args.bfile+extBED)).st_size
    # throw error if mismatch
    if observedbytes>expectedbytes:
        raise ValueError('More bytes in '+args.bfile+extBED+' than expected. File corrupted?')
    elif observedbytes<expectedbytes:
        raise ValueError('Fewer bytes in '+args.bfile+extBED+' than expected. File corrupted?')
    # compute rounded n (i.e. empty including empty bits)
    roundedn=nbt*nperbyte
    # get rowid of first two bits per byte being read
    ids=nperbyte*np.arange(nbt*Mperb)
    # connect to read bed and bim files
    connbed=open(args.bfile+extBED,'rb')
    connbim=open(args.bfile+extBIM,'r')
    # check if first three bytes bed file are correct
    if ord(connbed.read(1))!=(ord(binBED1)) or ord(connbed.read(1))!=(ord(binBED2)) or ord(connbed.read(1))!=(ord(binBED3)):
        raise ValueError(args.bfile+extBED+' not a valid PLINK .bed file')
    # connect to write effect file
    conneff=open(args.out+extEFF,'w')
    # print header row to effect file
    conneff.write('CHROMOSOME'+sep\
                    +'SNP_ID'+sep\
                    +'BASELINE_ALLELE'+sep\
                    +'EFFECT_ALLELE'+sep\
                    +'ALPHA1'+sep\
                    +'ALPHA2'+sep\
                    +'BETA1'+sep\
                    +'BETA2'+sep\
                    +'GAMMA'+eol)
    logger.info('Reading in '+args.bfile+extBED+' in blocks of '+str(Mperb)+' SNPs')
    # for each blok
    for b in tqdm(range(B)):
        # count number of SNP in this block
        m=min(M,(b+1)*Mperb)-b*Mperb
        # calculate how many bytes per read
        bytesperread=m*nbt
        # calculate how many distinct genotypes per read
        gperread=int(nperbyte*bytesperread)
        # read bytes
        gbytes=np.frombuffer(connbed.read(bytesperread),dtype=np.uint8)
        # initialise genotypes for this read as empty
        g=np.empty(gperread,dtype=np.uint8)
        # per individual in each byte
        for i in range(nperbyte):
            # take difference between what is left of byte after removing 2 bits
            gbytesleft=gbytes>>2
            g[ids[0:nbt*m]+i]=gbytes-(gbytesleft<<2)
            # keep part of byte that is left
            gbytes=gbytesleft
        # throw error if a genotype is missing; users should address this before simulation
        if (g==1).sum()>0:
            raise ValueError('Missing genotypes in PLINK files, which is not permissible in simulation of phenotypes; use e.g. `plink --bfile '+str(args.bfile)+' --geno 0 --make-bed --out '+str(args.bfile)+'2` to obtain PLINK binary dataset without missing values')
        # recode genotype where 0=homozygote A1, 1=heterozygote, 2=homozygote A2
        g[g==2]=1
        g[g==3]=2
        # reshape to genotype matrix
        g=g.reshape((m,roundedn)).T
        # drop rows corresponding to empty bits of last byte for each SNP
        g=g[0:n,:]
        # calculate empirical AFs
        eaf=g.mean(axis=0)/2
        # calculate standardised SNPs
        gs=(g-2*(eaf[None,:]))/(((2*eaf*(1-eaf))**0.5)[None,:])
        # draw factors for SNP effects on expectations
        gf1=rngeffects.normal(size=m)
        gf2=rngeffects.normal(size=m)
        # draw correlated SNP effects on expectations
        alpha1=gf1*((args.h2y1/M)**0.5)
        alpha2=((args.rg*gf1)+(((1-(args.rg**2))**0.5)*gf2))*((args.h2y2/M)**0.5)
        # draw SNP effects on variances and correlation
        beta1=rngeffects.normal(size=m)*((args.h2sig1/M)**0.5)
        beta2=rngeffects.normal(size=m)*((args.h2sig2/M)**0.5)
        gamma=args.rhoband*(rngeffects.normal(size=m)*((args.h2rho/M)**0.5))
        # update linear parts
        xalpha1+=((gs*alpha1[None,:]).sum(axis=1))
        xalpha2+=((gs*alpha2[None,:]).sum(axis=1))
        xbeta1+=((gs*beta1[None,:]).sum(axis=1))
        xbeta2+=((gs*beta2[None,:]).sum(axis=1))
        xgamma+=((gs*gamma[None,:]).sum(axis=1))
        # rescaling standardised coefficient to raw genotype effects
        scale=1/((2*eaf*(1-eaf))**0.5)
        alpha1=alpha1*scale
        alpha2=alpha2*scale
        beta1=beta1*scale
        beta2=beta2*scale
        gamma=gamma*scale
        # for each SNP in this block
        for j in range(m):
            # read line from bim file, strip trailing newline, split by tabs
            snpline=connbim.readline().rstrip(eol).split(sep)
            # get chromosome number, snp ID, baseline allele, and effect allele
            snpchr=snpline[0]
            snpid=snpline[1]
            snpbaseallele=snpline[4]
            snpeffallele=snpline[5]
            # print to .eff file the SNP info (above) and corresponding effects
            conneff.write(snpchr+sep+snpid+sep+snpbaseallele+sep\
                          +snpeffallele+sep+str(alpha1[j])+sep+str(alpha2[j])\
                          +sep+str(beta1[j])+sep+str(beta2[j])+sep+str(gamma[j])+eol)
    # close connection bed, bim, eff file
    connbed.close()
    connbim.close()
    conneff.close()
    # draw error terms for sigma and rho
    esig1=rng.normal(size=n)*((1-args.h2sig1)**0.5)
    esig2=rng.normal(size=n)*((1-args.h2sig2)**0.5)
    erho=args.rhoband*(rng.normal(size=n)*((1-args.h2rho)**0.5))
    # calculate standard deviations
    sig1=np.exp(-1+esig1+xbeta1)
    sig2=np.exp(-1+esig2+xbeta2)
    # find intercept for linear part of correlation, such that average
    # correlation equals rhomean
    gamma0=np.log((1+args.rhomean)/(1-args.rhomean))
    delta=np.exp(gamma0+erho+xgamma)
    rho=(delta-1)/(delta+1)
    # draw noise factors
    eta1=rng.normal(size=n)
    eta2=rng.normal(size=n)
    # scale and mix noise to achieve desired standard deviations and correlations 
    e1=eta1*sig1*((1-args.h2y1)**0.5)
    e2=((rho*eta1)+(((1-(rho**2))**0.5)*eta2))*sig2*((1-args.h2y2)**0.5)
    # draw outcomes and store in dataframe
    y1=xalpha1+e1
    y2=xalpha2+e2
    ydata=pd.DataFrame(np.hstack((y1[:,None],y2[:,None])),columns=['Y1','Y2'])
    # read fam file to dataframe
    famdata=pd.read_csv(args.bfile+extFAM,sep=sep,header=None,names=['FID','IID','PID','MID','SEX','PHE'])
    # concatenate FID,IID,Y1,Y2, and write to csv
    pd.concat([famdata.iloc[:,[0,1]],ydata],axis=1).to_csv(args.out+extPHE,index=False,sep=sep)
    # store name of just generated phenotype file
    args.pheno=args.out+extPHE

def sec_to_str(t):
    [d,h,m,s,n]=reduce(lambda ll, b : divmod(ll[0], b) + ll[1:], [(t, 1), 60, 60, 24])
    f = ''
    if d > 0:
        f += '{D}d:'.format(D=d)
    if h > 0:
        f += '{H}h:'.format(H=h)
    if m > 0:
        f += '{M}m:'.format(M=m)
    f += '{S}s'.format(S=s)
    return f

def positive_int(string):
    try:
        val=int(string)
    except:
        raise argparse.ArgumentTypeError(string+" is not a positive integer")
    if val>0:
        return val
    else:
        raise argparse.ArgumentTypeError("%s is not a positive integer" % val)

def number_between_0_1(string):
    try:
        val=float(string)
    except:
        raise argparse.ArgumentTypeError(string+" is not a number in the interval (0,1)")
    if val>0 and val<1:
        return val
    else:
        raise argparse.ArgumentTypeError("%s is not a number in the interval (0,1)" % val)

def number_between_m1_p1(string):
    try:
        val=float(string)
    except:
        raise argparse.ArgumentTypeError(string+" is not a number in the interval (-1,1)")
    if (val**2)<1:
        return val
    else:
        raise argparse.ArgumentTypeError("%s is not a number in the interval (-1,1)" % val)

def ParseInputArguments():
    # get global parser and initialise args as global
    global args
    # define input arguments
    parser.add_argument('--n', metavar = 'INTEGER', default = None, type = positive_int, 
                    help = '(simulation) number of individuals; at least 1000')
    parser.add_argument('--m', metavar = 'INTEGER', default = None, type = positive_int, 
                    help = '(simulation) number of SNPs; at least 100')
    parser.add_argument('--seed-geno', metavar = 'INTEGER', default = None, type = positive_int,
                    help = '(simulation) seed for random-number generator for genotypes')
    parser.add_argument('--bfile', metavar = 'PREFIX', default = None, type = str,
                    help = 'prefix of PLINK binary files; cannot be combined with --n, --m, and/or --seed-geno')
    parser.add_argument('--h2y1', metavar = 'NUMBER', default = None, type = number_between_0_1,
                    help = '(simulation) heritability of Y1; between 0 and 1')
    parser.add_argument('--h2y2', metavar = 'NUMBER', default = None, type = number_between_0_1,
                    help = '(simulation) heritability of Y2; between 0 and 1')
    parser.add_argument('--rg', metavar = 'NUMBER', default = None, type = number_between_m1_p1,
                    help = '(simulation) genetic correlation between Y1 and Y2; between -1 and 1')
    parser.add_argument('--h2sig1', metavar = 'NUMBER', default = None, type = number_between_0_1,
                    help = '(simulation) heritability of linear part of Var(Error term Y1); between 0 and 1')
    parser.add_argument('--h2sig2', metavar = 'NUMBER', default = None, type = number_between_0_1,
                    help = '(simulation) heritability of linear part of Var(Error term Y2); between 0 and 1')
    parser.add_argument('--h2rho', metavar = 'NUMBER', default = None, type = number_between_0_1,
                    help = '(simulation) heritability of linear part of Corr(Error term Y1,Error term Y2); between 0 and 1')
    parser.add_argument('--rhomean', metavar = 'NUMBER', default = None, type = number_between_m1_p1,
                    help = '(simulation) average Corr(Error term Y1,Error term Y2); between -1 and 1')
    parser.add_argument('--rhoband', metavar = 'NUMBER', default = None, type = number_between_0_1,
                    help = '(simulation) probabilistic bandwidth of correlation around level specified using --rhomean; between 0 and 1')
    parser.add_argument('--seed-effects', metavar = 'INTEGER', default = None, type = positive_int,
                    help = '(simulation) seed for random-number generator for unscaled true effects of standardised SNPs')
    parser.add_argument('--seed-pheno', metavar = 'INTEGER', default = None, type = positive_int,
                    help = '(simulation) seed for random-number generator for phenotypes')
    parser.add_argument('--pheno', metavar = 'FILENAME', default = None, type = str,
                    help = 'name of phenotype file: should be comma-, space-, or tab-separated, with one row per individual, with FID and IID as first two fields, followed by two fields for phenotypes Y1 and Y2; first row must contain labels (e.g. FID IID HEIGHT log(BMI)); requires --bfile to be specified; cannot be combined with --h2y1, --h2y2, --rg, --h2sig1, --h2sig2, --h2rho, --rhomean, --rhoband, --seed-effects, and/or --seed-pheno')
    parser.add_argument('--covar', metavar = 'FILENAME', default = None, type = str,
                    help = 'name of covariate file: should be comma-, space-, or tab-separated, with one row per individual, with FID and IID as first two fields, followed by a field per covariate; first row must contain labels (e.g. FID IID AGE AGESQ PC1 PC2 PC3 PC4 PC5); requires --pheno to be specified; WARNING: do not include an intercept in your covariate file, because GCAT always adds an intercept itself')
    parser.add_argument('--simul-only', action = 'store_true',
                    help = 'option to simulate data only (i.e. no analysis of simulated data); cannot be combined with --pheno')
    parser.add_argument('--snp', metavar = '', default = None, type = positive_int, nargs= '+',
                    help = '\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\bINTEGER INTEGER option to analyse only SNP with index j=s,...,t, where s=1st integer and t=2nd integer; cannot be combined with --simul-only')
    parser.add_argument('--out', metavar = 'PREFIX', default = None, type = str,
                    help = 'prefix of output files')
    try:
        # parse input arguments
        args=parser.parse_args()
    except Exception:
        raise SyntaxError('you specified incorrect input options')
        
def InitialiseLogger():
    # customise the logger using the prefix for output-files
    c_handler = logging.StreamHandler()
    # if no --out option has not been specified, use generic output prefix
    if args.out is not None:
        # get directory name if present within prefix
        sDir = os.path.dirname(args.out)
        # check if output holds directory name at all, and if so whether it doesn't exist
        if not(sDir == '') and not(os.path.isdir(sDir)):
            # if so, raise an error
            raise ValueError('prefix specified using --out may start with a directory name; this directory must exist however. ' + sDir + ' is not a directory')
        # set up log file handler with using provided output prefix
        f_handler=logging.FileHandler(args.out+extLOG,'w+',encoding="utf-8")
    else:
        # set up log file handler with using default output prefix
        f_handler=logging.FileHandler(outDEFAULT+extLOG,'w+',encoding="utf-8")
    # finish configuration logger
    c_handler.setLevel(logging.DEBUG)
    f_handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    # for output to the console, just print warnings, info, errors, etc.
    c_format=logging.Formatter('%(message)s')
    # for output to the log file also add timestamps
    f_format = logging.Formatter('%(asctime)s: %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    # add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

def ShowWelcome():
    try:
        defaults=vars(parser.parse_args(''))
        opts=vars(args)
        non_defaults=[x for x in opts.keys() if opts[x] != defaults[x]]
        header = HEADER
        header += eol+'Your call:'+eol
        header += './gcat.py \\'+eol
        options = ['--'+x.replace('_','-')+' '+str(opts[x])+' \\' for x in non_defaults]
        header += eol.join(options).replace('True','').replace('False','').replace("', \'", ' ').replace("']", '').replace("['", '').replace('[', '').replace(']', '').replace(', ', ' ').replace('  ', ' ')
        header = header[0:-1]+eol
        logger.info(header)
        if args.out is None:
            args.out=outDEFAULT
    except Exception:
        raise SyntaxError('you specified incorrect input options')

def CheckInputArgs():
    global simulg, simuly, covars
    # set covars to False by default, change if needed based on input
    covars=False
    if args.bfile is not None and (args.n is not None or args.m is not None):
        raise SyntaxError('you cannot combine --bfile with --n and/or --m')
    if args.n is not None and args.m is None:
        raise SyntaxError('--n must be combined with --m')
    if args.m is not None and args.n is None:
        raise SyntaxError('--m must be combined with --n')
    if args.n is None and args.m is None and args.bfile is None:
        raise SyntaxError('you must specify either --bfile or both --n and --m')
    if args.bfile is None:
        simulg=True
        simuly=True
        if args.m<MINM:
            raise ValueError('you simulate at least '+str(MINM)+' SNPs when using --m')
        if args.n<MINN:
            raise ValueError('you simulate at least '+str(MINN)+' individuals when using --n')
        if args.pheno is not None:
            raise SyntaxError('--pheno cannot be combined with --n and --m')
        if args.covar is not None:
            raise SyntaxError('--covar cannot be combined with --n and --m')
    else:
        simulg=False
        if not(os.path.isfile(args.bfile+extBED)):
            raise OSError('PLINK binary data file '+args.bfile+extBED+' cannot be found')
        if not(os.path.isfile(args.bfile+extBIM)):
            raise OSError('PLINK binary data file '+args.bfile+extBIM+' cannot be found')
        if not(os.path.isfile(args.bfile+extFAM)):
            raise OSError('PLINK binary data file '+args.bfile+extFAM+' cannot be found')
        if args.pheno is None:
            simuly=True
            if args.covar is not None:
                raise SyntaxError('--covar must be combined with --pheno')
        else:
            simuly=False
            if not(os.path.isfile(args.pheno)):
                raise OSError('Phenotype file '+args.pheno+ ' cannot be found')
            if args.covar is not None:
                covars=True
                if not(os.path.isfile(args.covar)):
                    raise OSError('Covariate file '+args.covar+ ' cannot be found')
            if args.simul_only:
                raise SyntaxError('--simul-only cannot be combined with --pheno')
    if simuly:
        if args.h2y1 is None:
            raise SyntaxError('--h2y1 must be specified when simulating phenotypes')
        if args.h2y2 is None:
            raise SyntaxError('--h2y2 must be specified when simulating phenotypes')
        if args.rg is None:
            raise SyntaxError('--rg must be specified when simulating phenotypes')
        if args.h2sig1 is None:
            raise SyntaxError('--h2sig1 must be specified when simulating phenotypes')
        if args.h2sig2 is None:
            raise SyntaxError('--h2sig2 must be specified when simulating phenotypes')
        if args.h2rho is None:
            raise SyntaxError('--h2rho must be specified when simulating phenotypes')
        if args.rhomean is None:
            raise SyntaxError('--rhomean must be specified when simulating phenotypes')
        if args.rhoband is None:
            raise SyntaxError('--rhoband must be specified when simulating phenotypes')
    else:
        if args.h2y1 is not None:
            raise SyntaxError('--h2y1 cannot be combined with --pheno')
        if args.h2y2 is not None:
            raise SyntaxError('--h2y2 cannot be combined with --pheno')
        if args.rg is not None:
            raise SyntaxError('--rg cannot be combined with --pheno')
        if args.h2sig1 is not None:
            raise SyntaxError('--h2sig1 cannot be combined with --pheno')
        if args.h2sig2 is not None:
            raise SyntaxError('--h2sig2 cannot be combined with --pheno')
        if args.h2rho is not None:
            raise SyntaxError('--h2rho cannot be combined with --pheno')
        if args.rhomean is not None:
            raise SyntaxError('--rhomean cannot be combined with --pheno')
        if args.rhoband is not None:
            raise SyntaxError('--rhoband cannot be combined with --pheno')
    if not(simulg):
        if args.seed_geno is not None:
            raise SyntaxError('--seed-geno may not be combined with --bfile')
    if not(simuly):
        if args.seed_pheno is not None:
            raise SyntaxError('--seed-pheno may not be combined with --pheno')
        if args.seed_effects is not None:
            raise SyntaxError('--seed-effects may not be combined with --pheno')
    if simulg:
        if args.seed_geno is None:
            raise SyntaxError('--seed-geno must be specified when simulating genotypes')
    if simuly:
        if args.seed_pheno is None:
            raise SyntaxError('--seed-pheno must be specified when simulating phenotypes')
        if args.seed_effects is None:
            raise SyntaxError('--seed-effects must be specified when simulating phenotypes')
    if args.simul_only and args.snp is not None:
        raise SyntaxError('--simul-only cannot be combined with --snp')
    if args.snp is not None:
        if len(args.snp)!=2:
            raise SyntaxError('--snp needs to be followed by two integers')
        elif args.snp[1]<args.snp[0]:
            raise SyntaxError('--snp requires 1st integer <= 2nd integer')

def FindBlockSize():
    global dimg
    # assigning 1% of available RAM to storage of raw genotypes
    # calculate how many raw genotypes can be held in RAM
    availtotal=(psutil.virtual_memory()[1])
    dimg=int(0.002*availtotal)

def main():
    # set parser, logger, memory tracker as globals
    global parser,logger,process
    # get start time
    t0=time.time()
    # initialise parser
    parser=argparse.ArgumentParser()
    # initialise logger
    logger=logging.getLogger(__name__)
    # initialise memory tracker
    process=psutil.Process(os.getpid())
    try: # try gcat
        # Parse input arguments
        ParseInputArguments()
        # Initialise logger
        InitialiseLogger()
        # Print welcome screen
        ShowWelcome()
        # Perform basic checks on input arguments
        CheckInputArgs()
        # Determine block size with which we can process SNPs staying within RAM
        FindBlockSize()
        # Simulate genotypes if necessary
        if simulg:
            logger.info(eol+'SIMULATING GENOTYPES')
            SimulateG()
        # Simulate phenotypes if necessary
        if simuly:
            logger.info(eol+'SIMULATING PHENOTYPES')
            SimulateY()
        # Perform GCAT if args.simul_only is False
        if not(args.simul_only):
            logger.info(eol+'PERFORMING GCAT')
            GCAT()
    except Exception:
        # print the traceback
        logger.error(traceback.format_exc())
        # wrap up with final error message
        logger.error('Error: GCAT did not exit properly. Please inspect the log file.')
        logger.info('Run `python ./gcat.py -h` to show all options')
    finally:
        # print total time elapsed
        logger.info('Total time elapsed: {T}'.format(T=sec_to_str(round(time.time()-t0, 2))))
        logger.info('Current memory usage is ' + str(int((process.memory_info().rss)/(1024**2))) + 'MB')

if __name__ == '__main__':
    main()
