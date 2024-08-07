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
RESULTSMBLOCK=100
READMBLOCK=1000

# where different types of parameters are located parameters array
ALPHA1COL=0
ALPHA2COL=1
BETA1COL=2
BETA2COL=3
GAMMACOL=4
TOTALCOLS=5

# define PLINK binary data variables
extBED='.bed'
extBIM='.bim'
extFAM='.fam'
binBED1=bytes([0b01101100])
binBED2=bytes([0b00011011])
binBED3=bytes([0b00000001])
nperbyte=4
lAlleles=['A','C','G','T']
fieldsFAM=['FID','IID','PID','MID','SEX','PHE']
idFIELD=['FID','IID']

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
__version__ = 'v0.2'
HEADER = eol
HEADER += '------------------------------------------------------------'+eol
HEADER += '| GCAT (Genome-wide Cross-trait concordance Analysis Tool) |'+eol
HEADER += '------------------------------------------------------------'+eol
HEADER += '| BETA {V}, (C) 2024 Ronald de Vlaming                    |'.format(V=__version__)+eol
HEADER += '| Vrije Universiteit Amsterdam                             |'+eol
HEADER += '| GNU General Public License v3                            |'+eol
HEADER += '------------------------------------------------------------'+eol

# define class for holding only numerical data on X and Y (different instances can be
# used e.g. to estimate SNP-specific models, with random subsampling) and parameter estimates
class Data:

    def __init__(self,x,xlabels,y1,y1label,y2,y2label,copy=False):
        # store input data as attributes
        self.x=x
        self.xlabels=xlabels
        self.y1=y1
        self.y1label=y1label
        self.y2=y2
        self.y2label=y2label
        # if totally new instance
        if not(copy):
            # store PLINK entries each row corresponds to
            self.ind=np.arange(self.x.shape[0])
            # find observations where at least one covariate is missing
            xmissing=np.isnan(self.x.sum(axis=1))
            # count number of observations with nonmissing data on covariates; throw error if zero
            nxnotnan=(~xmissing).sum()
            if nxnotnan==0:
                raise ValueError('No observations in '+args.covar\
                                +' without any missingness that can be matched to '+args.bfile+extFAM)
            # set y1,y2 to missing for rows where any covariate is missing, and set x to zero
            self.x[xmissing,:]=0
            self.y1[xmissing]=np.nan
            self.y2[xmissing]=np.nan
            # calculate lowest eigenvalue of X'X for observations without any missingness
            (Dxtx,_)=np.linalg.eigh((self.x.T@self.x)/nxnotnan)
            # if too low, throw error:
            if min(Dxtx)<MINEVAL:
                raise ValueError('Regressors in '+args.covar+' have too much multicollinearity. '\
                                +'Did you add an intercept to your covariates file? Please remove it. '\
                                +'Or do you have set of dummies that is perfectly collinear with intercept? '\
                                +'Please remove one category. '\
                                +'Recall: GCAT always adds intercept to model!')
            # find missingness, recode, and get counts
            self.FindAndRecodeMissingness()
            self.GetCounts()
            # initialise parameters and clean
            self.InitialiseParams()
            self.Clean()
    
    def copy(self):
        # make copy of data
        data=Data(self.x.copy(),self.xlabels.copy(),self.y1.copy(),self.y1label,self.y2.copy(),self.y2label,copy=True)
        # make sure to also copy booleans and indices of remaining PLINK data
        data.y1notnan=self.y1notnan.copy()
        data.y2notnan=self.y2notnan.copy()
        data.ybothnotnan=self.ybothnotnan.copy()
        data.y1ory2notnan=self.y1ory2notnan.copy()
        data.ind=self.ind.copy()
        # also copy parameter estimates and corresponding individual-specific weights
        data.param=self.param.copy()
        data.wL0=self.wL0.copy()
        data.wG0=self.wG0.copy()
        data.wH0=self.wH0.copy()
        # also copy counts
        data.n1=self.n1
        data.n2=self.n2
        data.nboth=self.nboth
        data.n=self.n
        data.N=self.N
        data.nrow=self.nrow
        data.k=self.k
        # return new Data instance
        return data
    
    def InitialiseParams(self):
        # report stats
        logger.info('Found '+str(self.n1)+' observations for '+self.y1label+' with complete data')
        logger.info('Found '+str(self.n2)+' observations for '+self.y2label+' with complete data')
        # initialise parameters baseline model
        logger.info('Initialising baseline model (i.e. without any SNPs)')
        # get x for observations where y1 resp. y2 is not missing
        x1=self.x[self.y1notnan,:]
        x2=self.x[self.y2notnan,:]
        # calculate EVD of X'X for these two subsets of observations
        (D1,P1,_)=BendEVDSymPSD(((x1.T@x1)/self.n1))
        (D2,P2,_)=BendEVDSymPSD(((x2.T@x2)/self.n2))
        # calculate coefficients for baseline regressors w.r.t. E[y1] and E[y2]
        alpha1=P1@((((x1.T@self.y1[self.y1notnan])/self.n1)@P1)/D1)
        alpha2=P2@((((x2.T@self.y2[self.y2notnan])/self.n2)@P2)/D2)
        # get regression residuals e1 and e2
        e1=self.y1-self.x@alpha1
        e2=self.y2-self.x@alpha2
        # take transformation f() of residuals, such that under GCAT's DGP
        # E[f(e)] is linear function of regressors
        z1=0.5*np.log(e1**2)
        z2=0.5*np.log(e2**2)
        # get regression coefficients for baseline regressors w.r.t. Var(y1) and Var(y2)
        beta1=P1@((((x1.T@z1[self.y1notnan])/self.n1)@P1)/D1)
        beta2=P2@((((x2.T@z2[self.y2notnan])/self.n2)@P2)/D2)
        # rescale residuals based on implied standard deviation
        r1=e1/np.exp(self.x@beta1)
        r2=e2/np.exp(self.x@beta2)
        # calculate correlation between rescaled residuals
        rhomean=(np.corrcoef(r1[self.ybothnotnan],r2[self.ybothnotnan]))[0,1]
        # set intercept for gamma s.t. Corr(y1,y2)=rhomean for all individuals
        gamma=np.zeros(self.k)
        gamma[0]=np.log((1+rhomean)/(1-rhomean))
        # collect and return initialised values
        self.param=np.empty((self.k,TOTALCOLS))
        self.param[:,ALPHA1COL]=alpha1
        self.param[:,ALPHA2COL]=alpha2
        self.param[:,BETA1COL]=beta1
        self.param[:,BETA2COL]=beta2
        self.param[:,GAMMACOL]=gamma
        # initialise individual-specific weights baseline model at zero
        self.wL0=np.zeros(self.nrow)
        self.wG0=np.zeros((self.nrow,TOTALCOLS))
        self.wH0=np.zeros((self.nrow,TOTALCOLS,TOTALCOLS))
    
    def FindAndRecodeMissingness(self):
        # find indices of non missing observations
        self.y1notnan=~np.isnan(self.y1)
        self.y2notnan=~np.isnan(self.y2)
        # find individuals with complete data
        self.ybothnotnan=self.y1notnan&self.y2notnan
        # find individuals with x and either y1 or y2 complete
        self.y1ory2notnan=self.y1notnan|self.y2notnan
        # set y1 and y2 to zero at missing points
        self.y1[~self.y1notnan]=0
        self.y2[~self.y2notnan]=0
    
    def GetCounts(self):
        # count number of complete observations per trait
        self.n1=self.y1notnan.sum()
        self.n2=self.y2notnan.sum()
        # count number of individuals with both traits complete
        self.nboth=self.ybothnotnan.sum()
        # count number of individuals with either y1, y2, or both (=no. of indep obs)
        self.n=self.y1ory2notnan.sum()
        # count total number of observation in multivariate model
        self.N=self.n1+self.n2
        # count dimensionality
        self.nrow=self.x.shape[0]
        self.k=self.x.shape[1]

    def Subsample(self,keep):
        # keep x, y1, y2 for subset of observations
        self.x=self.x[keep,:]
        self.y1=self.y1[keep]
        self.y2=self.y2[keep]
        # idem for booleans indicating missingness
        self.y1notnan=self.y1notnan[keep]
        self.y2notnan=self.y2notnan[keep]
        self.ybothnotnan=self.ybothnotnan[keep]
        self.y1ory2notnan=self.y1ory2notnan[keep]
        # idem for individual-specific weights
        self.wL0=self.wL0[keep]
        self.wG0=self.wG0[keep,:]
        self.wH0=self.wH0[keep,:,:]
        # store only the PLINK entries this corresponds to
        self.ind=self.ind[keep]
        # update counts
        self.GetCounts()

    def AddSNP(self,g,gisnan):
        # consider genotypes and their missingness only for individuals still in data
        g=g[self.ind]
        gisnan=gisnan[self.ind]
        # add SNP to matrix of regressors (last column)
        self.x=np.hstack((self.x,g[:,None]))
        # add SNP effects to matrix of parameters (last row)
        self.param=np.vstack((self.param,np.zeros((1,TOTALCOLS))))
        # keep only subsample for which genotype is not missing
        self.Subsample(~gisnan)
        
    def Clean(self):
        # keep only observations for which we have either y1, y2, or both
        self.Subsample(self.y1ory2notnan)
    
# define class for reading phenotypes, covariates, and perfoming analyses
class Analyser:
    
    def __init__(self,snpreader):
        # print update
        logger.info('Reading phenotype data')
        # read fam file
        famdata=pd.read_csv(args.bfile+extFAM,sep=sep,header=None,names=fieldsFAM)
        self.nPLINK=famdata.shape[0]
        logger.info('Found '+str(self.nPLINK)+' individuals in '+args.bfile+extFAM)
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
        DATA=pd.merge(left=DATA,right=ydata,how='left',left_on=idFIELD,right_on=idFIELD)
        # retrieve matched phenotypes baseline model
        y1=DATA.values[:,2]
        y2=DATA.values[:,3]
        # if covars provideded
        if covars:
            # print update
            logger.info('Reading covariates data')
            # read covars
            xdata=pd.read_csv(args.covar,sep=sep)
            # if no covariates found, throw error
            if xdata.shape[1]<3:
                raise ValueError(args.covar+' does not contain data on any covariates')
            # print update
            logger.info('Found '+str(xdata.shape[0])+' individuals and '\
                         +str(xdata.shape[1]-2)+' control variables in '+args.covar)
            # left joint data with covariates
            DATA=pd.merge(left=DATA,right=xdata,how='left',left_on=idFIELD,right_on=idFIELD)
            # retrieve matched covariates baseline model, and add intercept
            x=np.hstack((np.ones((self.nPLINK,1)),DATA.values[:,4:]))
            xlabels=['intercept']+xdata.iloc[:,2:].columns.to_list()
        else: # else set intercept as only covariate
            x=np.ones((self.nPLINK,1))
            xlabels=['intercept']
        # initialise main data and parameters, keep only observations for which there is
        # no missingness in x, and at most one trait is missing
        self.data=Data(x,xlabels,y1,y1label,y2,y2label)
        # store genotype reader
        self.snpreader=snpreader
        # estimate baseline model
        logger.info('Estimating baseline model')
        converged0=self.Newton(self.data)
        # if baseline model did not converge: throw error
        if not(converged0):
            raise RuntimeError('Estimates baseline model (=no SNPs) not converged')
        # write baseline model estimates to output file
        pd.DataFrame(self.data.param,columns=['ALPHA1','ALPHA2','BETA1','BETA2','GAMMA'],\
                     index=self.data.xlabels).to_csv(args.out+extBASE,sep=sep)
        # if one-step efficient estimation
        if onestep:
            # make copy of the data
            self.datasmall=self.data.copy()
            # initialise random-number generator
            rng=np.random.default_rng(args.one[1])
            # randomly sample the desired proportion
            keep=np.sort(rng.permutation(self.data.nrow)[:min(int(self.nPLINK*args.one[0]),self.data.nrow)])
            logger.info('Keeping random subsample of '+str(len(keep))\
                        +' individuals for one-step efficient estimation')
            self.datasmall.Subsample(keep)
    
    def Newton(self,data,baseline=True,silent=False,onestepfinal=False,reestimation=False):
        # set iteration counter to 0 and convergence to false
        i=0
        converged=False
        # if baseline model or final part of one-step efficient estimation
        if baseline or onestepfinal: 
            # fully calculate log-likelihood, gradient, Hessian
            (logL,grad,H)=self.CalcLogL(data)
        else: # if SNP-specific model
            # calculate log-likelihood, gradient, Hessian utilising weights converged baseline model
            (logL,grad,H)=self.CalcLogL(data,useweightsbaseline=True)
        # while not converged and MAXITER not reached
        while not(converged) and i<MAXITER:
            # if log-likelihood is -np.inf: quit; on a dead track for this model
            if np.isinf(logL):
                if baseline:
                    data.wL0=None
                    data.wG0=None
                    data.wH0=None
                    data.logL0=logL
                    return converged
                if reestimation: return logL,converged
                if onestep and not(onestepfinal): return converged,i
                return logL,None,[0],None,converged,i
            # unpack Hessian to matrix
            UH=H.reshape((data.k*TOTALCOLS,data.k*TOTALCOLS))
            # get (bended) EVD of unpacked -Hessian
            (Dadj,P,D)=BendEVDSymPSD(-UH)
            # get Newton-Raphson update
            update=(P@((((grad.reshape((data.k*TOTALCOLS,1))).T@P)/Dadj).T)).reshape((data.k,TOTALCOLS))
            # calculate convergence criterion
            msg=(update*grad).mean()
            # if converged: convergenced=True
            if msg<TOL or (onestepfinal and i>0):
                converged=True
            else: # if not converged yet
                # do line search if baseline model or linesearch is activated
                if baseline or linesearch:
                    (j,step)=self.GoldenSection(data,logL,grad,update)
                else: data.param+=update # else just full Newton update
                # update iteration counter
                i+=1
                # provide info if not silent
                if not(silent):
                    report='Newton iteration '+str(i)+': logL='+str(logL)
                    if baseline or linesearch:
                        report+='; '+str(j)+' line-search steps, yielding step size = '+str(step)
                    logger.info(report)
                # calculate new log-likelihood, gradient, Hessian
                (logL,grad,H)=self.CalcLogL(data)
        # if baseline model
        if baseline:
            # get individual-specific weights: useful for 1st iteration SNP-specific model
            (wL,wG,wH)=self.CalcLogL(data,weightsonly=True)
            # store individual-specific weights and overall log-likelihood
            data.wL0=wL
            data.wG0=wG
            data.wH0=wH
            data.logL0=logL
            # and return whether converged
            return converged
        # if re-estimation for LRT, for subset of individuals for whom SNP is not missing
        if reestimation:
            return logL,converged
        # if one-step efficient estimation (doing full convergence in small sample)
        # and this is not the final step in the large sample
        if onestep and not(onestepfinal):
            # return number of iterations and whether converged
            return converged,i
        # calculate contribution each individual to gradient, needed for robust errors
        G=self.CalcLogL(data,Gonly=True)
        # return logL, grad contributions, EVD of unpacked -Hessian, and no. of iter and whether converged
        return logL,G,D,P,converged,i
    
    def BFGS(self,data,reestimation=False):
        # set iteration counter to 0 and convergence to false
        i=0
        converged=False
        # calculate log-likelihood, gradient, Hessian utilising weights converged baseline model
        (logL,grad,H)=self.CalcLogL(data,useweightsbaseline=True)
        # unpack Hessian to matrix
        UH=H.reshape((data.k*TOTALCOLS,data.k*TOTALCOLS))
        # get (bended) EVD of unpacked -Hessian
        (Dadj,P,D)=BendEVDSymPSD(-UH)
        # initialise approximated inverse Hessian
        AIH=-(P*(1/Dadj[None,:]))@P.T
        # while not converged and MAXITER not reached
        while not(converged) and i<MAXITER:
            # if log-likelihood is -np.inf: quit; on a dead track for this model
            if np.isinf(logL):
                if reestimation: return logL,converged
                if onestep: return converged,i
                return logL,None,[0],None,converged,i
            # get BFGS update
            update=(-AIH@grad.reshape((data.k*TOTALCOLS,1))).reshape((data.k,TOTALCOLS))
            # calculate convergence criterion
            msg=(update*grad).mean()
            # if converged: convergenced=True
            if msg<TOL:
                converged=True
            else: # if not converged yet: get new parameters, and corresponding logL and grad
                # either via line search
                if linesearch:
                    (j,step,paramnew,logLnew,gradnew)=self.GoldenSection(data,logL,grad,update,bfgs=True)
                else: # or full update
                    paramnew=data.param+update
                    (logLnew,gradnew)=self.CalcLogL(data,param=paramnew,logLgradonly=True)
                # calculate quantities needed for BFGS
                s=(paramnew-data.param).reshape((data.k*TOTALCOLS,1))
                y=(gradnew-grad).reshape((data.k*TOTALCOLS,1))
                sty=(s*y).sum()
                r=1/sty
                v=s*r
                w=AIH@y
                # store new parameters, gradient, logL, and update inverse Hessian, and stabilise
                data.param=paramnew
                grad=gradnew
                logL=logLnew
                AIH=AIH-np.outer(v,w)-np.outer(w,v)+np.outer(v,v)*((w*y).sum())+np.outer(v,s)
                AIH=(AIH+(AIH.T))/2
                # update iteration counter
                i+=1
        # if re-estimation for LRT, for subset of individuals for whom SNP is not missing
        if reestimation:
            return logL,converged
        # if one-step efficient estimation (doing inference based on final Newton step in full sample)
        if onestep:
            # return number of iterations and whether converged
            return converged,i
        # calculate log-likelihood, gradient, Hessian at BFGS solution
        (logL,grad,H)=self.CalcLogL(data)
        # unpack Hessian to matrix
        UH=H.reshape((data.k*TOTALCOLS,data.k*TOTALCOLS))
        # get (bended) EVD of unpacked -Hessian
        (Dadj,P,D)=BendEVDSymPSD(-UH)
        # calculate contribution each individual to gradient, needed for robust errors
        G=self.CalcLogL(data,Gonly=True)
        # return logL, grad contributions, EVD of unpacked -Hessian, whether converged, and no. of iter
        return logL,G,D,P,converged,i

    def GoldenSection(self,data,logL,grad,update,bfgs=False):
        # calculate update'grad
        utg=(grad*update).sum()
        # initialise parameters at various points along interval
        param1=data.param
        param2=data.param+oneminthetainv*update
        param3=data.param+thetainv*update
        param4=data.param+update
        # set corresponding step sizes
        step1=0
        step2=oneminthetainv
        step3=thetainv
        step4=1
        # set iteration counter to one and converged to false
        j=1
        converged=False
        # calculate log likelihood at right
        logL4=self.CalcLogL(data,param=param4,logLonly=True)
        # directly try Armijo's rule before performing actual section search
        if logL4>=logL+ARMIJO*step4*utg:
            converged=True
        else: # if not directly meeting criterion
            # calculate/set remaining log likelihoods (left, mid-left, mid-right)
            logL1=logL
            logL2=self.CalcLogL(data,param=param2,logLonly=True)
            logL3=self.CalcLogL(data,param=param3,logLonly=True)
        # while not converged and MAXITER not reached
        while not(converged) and j<MAXITER:
            # update iteration counter
            j+=1
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
                logL2=self.CalcLogL(data,param=param2,logLonly=True)
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
                logL3=self.CalcLogL(data,param=param3,logLonly=True)
            # test if Armijo's rule satisfied
            if logL4>=logL+ARMIJO*step4*utg:
                converged=True
        # if we're doing BFGS
        if bfgs:
            # calculate gradient, and return relevant output for new estimates
            grad4=self.CalcLogL(data,param=param4,gradonly=True)
            return j,step4,param4,logL4,grad4
        # otherwise, update parameters, and just return no. of steps, and step size
        data.param=param4
        return j,step4
    
    def CalcLogL(self,data,param=None,useweightsbaseline=False,weightsonly=False,\
                 logLonly=False,logLgradonly=False,Gonly=False,gradonly=False):
        # if we can use weights from estimated baseline model
        if useweightsbaseline:
            # calculate log-likelihood/n and gradient/n
            logL=(data.wL0.sum())/data.n
            grad=(data.x.T@data.wG0)/data.n
            # calculate Hessian/n
            H=CalculateHessian(data.x,data.wH0)/data.n
            # return those components
            return logL,grad,H
        # if parameters not provided, use what's in data
        if param is None:
            param=data.param
        # calculate linear parts
        linpart=(data.x[:,:,None]*param[None,:,:]).sum(axis=1)
        # set appropriate parts to zero for observations with missingness
        linpart[~data.y1notnan,ALPHA1COL]=0
        linpart[~data.y2notnan,ALPHA2COL]=0
        linpart[~data.y1notnan,BETA1COL]=0
        linpart[~data.y2notnan,BETA2COL]=0
        linpart[~data.ybothnotnan,GAMMACOL]=0
        # get sigma1, sigma2, delta, and rho
        sig1=np.exp(linpart[:,BETA1COL])
        sig2=np.exp(linpart[:,BETA2COL])
        delta=np.exp(linpart[:,GAMMACOL])
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
        # if halt=True: set -np.inf as log-likelihood, and stop
        if halt:
            logL=-np.inf
            if logLonly: return logL
            if logLgradonly: return logL,None
            if gradonly or Gonly: return None
            if weightsonly: return None,None,None
            return logL,None,None
        # set sig1 and sig2 to np.inf for missing observations
        sig1[~data.y1notnan]=np.inf
        sig2[~data.y2notnan]=np.inf
        # calculate errors
        e1=data.y1-linpart[:,ALPHA1COL]
        e2=data.y2-linpart[:,ALPHA2COL]
        # calculate rescaled errors (i.e. error/stdev)
        r1=(e1/sig1)
        r2=(e2/sig2)
        # calculate some other key ingredients for logL/grad/H
        rhosq=rho**2
        unexp=1-rhosq
        invunexp=1/unexp
        r1sq=r1**2
        r2sq=r2**2
        r1r2=r1*r2
        r1sqplusr2sq=r1sq+r2sq
        rhor1r2=rho*r1r2
        # calculate log-likelihood only if necessary
        if logLonly or logLgradonly or (not(gradonly) and not(Gonly) and not(weightsonly)):
            # calculate constant
            cons=data.N*np.log(2*np.pi)
            # calculate log|V| and quadratic term
            logdetV=2*data.nboth*np.log(2)+linpart[:,GAMMACOL].sum()\
                +2*(linpart[:,BETA1COL].sum()+linpart[:,BETA2COL].sum())\
                    -2*((np.log(delta[data.ybothnotnan]+1)).sum())
            quadratic=((r1sqplusr2sq-2*rhor1r2)*invunexp).sum()
            # calculate and return logL/n
            logL=-0.5*(cons+logdetV+quadratic)/data.n
        # if only logL desired, return that
        if logLonly:
            return logL
        # initialise weights matrix for gradient
        wG=np.empty((data.nrow,TOTALCOLS))
        # calculate key ingredients for grad/H
        rhor1=rho*r1
        rhor2=rho*r2
        invsig1unexp=invunexp/sig1
        invsig2unexp=invunexp/sig2
        deltasqm1div4delta=((delta**2)-1)/(4*delta)
        deltasqp1div2delta=((delta**2)+1)/(2*delta)
        # calculate weights matrix for gradient
        wG[:,ALPHA1COL]=(r1-rhor2)*invsig1unexp
        wG[:,ALPHA2COL]=(r2-rhor1)*invsig2unexp
        wG[:,BETA1COL]=((r1sq-rhor1r2)/unexp)-1
        wG[:,BETA2COL]=((r2sq-rhor1r2)/unexp)-1
        wG[:,GAMMACOL]=(rho-deltasqm1div4delta*r1sqplusr2sq+deltasqp1div2delta*r1r2)/2
        # set gradient=0 w.r.t. beta1 for missing y1 and idem w.r.t. beta2 for missing y2
        wG[~data.y1notnan,BETA1COL]=0
        wG[~data.y2notnan,BETA2COL]=0
        # calculate gradient only if necessary
        if logLgradonly or gradonly or (not(Gonly) and not(weightsonly)):
            grad=(data.x.T@wG)/data.n
        # if only logL and grad desired, return that
        if logLgradonly:
            return logL,grad
        # if only gradient desired, return that
        if gradonly:
            return grad
        # if only contribution per individual to gradient wanted
        if Gonly:
            # calculate individual-specific contribution to gradient/n as 3d array
            G=((data.x.T[:,None,:])*(wG.T[None,:,:]))/data.n
            # and return that
            return G
        # calculate key ingredients for Hessian
        rhodivunexp=rho*invunexp
        rhosqplus1=1+rhosq
        # initialise weights array Hessian (Nplink-by-TOTALCOLS-by-TOTALCOLS)
        wH=np.empty((data.nrow,TOTALCOLS,TOTALCOLS))
        # calculate weights array for Hessian
        wH[:,ALPHA1COL,ALPHA1COL]=-invsig1unexp/sig1
        wH[:,ALPHA2COL,ALPHA2COL]=-invsig2unexp/sig2
        wH[:,ALPHA1COL,ALPHA2COL]=rho*invsig1unexp/sig2
        wH[:,ALPHA1COL,BETA1COL]=invsig1unexp*(rhor2-2*r1)
        wH[:,ALPHA2COL,BETA2COL]=invsig2unexp*(rhor1-2*r2)
        wH[:,ALPHA1COL,BETA2COL]=invsig1unexp*(rhor2)
        wH[:,ALPHA2COL,BETA1COL]=invsig2unexp*(rhor1)
        wH[:,ALPHA1COL,GAMMACOL]=invsig1unexp*(rhor1-(rhosqplus1*(r2/2)))
        wH[:,ALPHA2COL,GAMMACOL]=invsig2unexp*(rhor2-(rhosqplus1*(r1/2)))
        wH[:,BETA1COL,BETA1COL]=invunexp*(rhor1r2-2*r1sq)
        wH[:,BETA2COL,BETA2COL]=invunexp*(rhor1r2-2*r2sq)
        wH[:,BETA1COL,BETA2COL]=invunexp*rhor1r2
        wH[:,BETA1COL,GAMMACOL]=rhodivunexp*r1sq-(r1r2/2)
        wH[:,BETA2COL,GAMMACOL]=rhodivunexp*r2sq-(r1r2/2)
        wH[:,GAMMACOL,GAMMACOL]=deltasqm1div4delta*r1r2+(unexp-deltasqp1div2delta*r1sqplusr2sq)/4
        # set weight w.r.t. gamma twice to zero when either y1 and/or y2 is missing
        wH[~data.ybothnotnan,GAMMACOL,GAMMACOL]=0
        # if only weights desired
        if weightsonly:
            # initialise log-likelihood per individual as zero
            wL=np.zeros(data.nrow)
            # add constant
            wL[data.y1notnan]+=np.log(2*np.pi)
            wL[data.y2notnan]+=np.log(2*np.pi)
            # add log|V|
            wL+=linpart[:,GAMMACOL]+2*(linpart[:,BETA1COL]+linpart[:,BETA2COL])
            wL[data.ybothnotnan]+=2*np.log(2)
            wL[data.ybothnotnan]-=2*(np.log(delta[data.ybothnotnan]+1))
            # add quadratic term
            wL+=(r1sqplusr2sq-2*rhor1r2)*invunexp
            # turn to log-likelihood per individual
            wL=-0.5*wL
            # return weights
            return wL,wG,wH
        # calculate Hessian/n
        H=CalculateHessian(data.x,wH)/data.n
        return logL,grad,H
    
    def EstimateSNPModels(self):
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
                        +'N_'+self.data.y1label+'_COMPLETE'+sep\
                        +'N_'+self.data.y2label+'_COMPLETE'+sep\
                        +'NBOTHCOMPLETE'+sep\
                        +'ITERATIONS'+sep\
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
                        +'APE_STDEV_'+self.data.y1label+sep\
                        +'SE_APE_STDEV_'+self.data.y1label+sep\
                        +'ESTIMATE_BETA2'+sep\
                        +'SE_BETA2'+sep\
                        +'WALD_BETA2'+sep\
                        +'P_BETA2'+sep\
                        +'APE_STDEV_'+self.data.y2label+sep\
                        +'SE_APE_STDEV_'+self.data.y1label+sep\
                        +'ESTIMATE_GAMMA'+sep\
                        +'SE_GAMMA'+sep\
                        +'WALD_GAMMA'+sep\
                        +'P_GAMMA'+sep\
                        +'APE_CORR_'+self.data.y1label+'_'\
                            +self.data.y2label+sep\
                        +'SE_APE_CORR_'+self.data.y1label+'_'\
                            +self.data.y2label+sep\
                        +'WALD_JOINT'+sep\
                        +'P_WALD_JOINT')
        # if doing LRT per SNP, print two additional fields
        if args.lrt: connassoc.write(sep+'LRT_JOINT'+sep+'P_LRT_JOINT')
        # write EOL character to association results file
        connassoc.write(eol)
        # initialise progress bar
        self.pbar=tqdm(total=self.snpreader.Mt)
        # for each block
        for b in range(self.snpreader.ResultBlocksT):
            # get start and ending SNP
            m0=self.snpreader.Mstart+(b*RESULTSMBLOCK)
            m1=min(self.snpreader.Mstart+(b+1)*RESULTSMBLOCK,self.snpreader.Mend)
            # using parallel execution in block with writing at end of block
            with ThreadPoolExecutor() as executor:
                # analyse snp j
                outputlines=executor.map(self.AnalyseOneSNP,[j for j in range(m0,m1)])
            for outputline in outputlines:
                connassoc.write(outputline)        
        # close progress bar
        self.pbar.close()
        # close connections to assoc file
        connassoc.close()
    
    def AnalyseOneSNP(self,j):
        # get SNP data
        (g,gisnan)=self.snpreader.ReadSNP(j)
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
        # if one-step efficient estimation, set small data as main data, and add SNP
        if onestep:
            data1=self.datasmall.copy()
            data1.AddSNP(g,gisnan)
        else: # else: set large data as main data, and add SNP
            data1=self.data.copy()
            data1.AddSNP(g,gisnan)
        # estimate, provided nboth>=MINN
        if data1.nboth>=MINN:
            # using BFGS or Newton's method, depending on input
            if args.bfgs:
                # if one-step efficient estimation, only catch whether converged, and number of iterations
                if onestep:
                    (converged1,i1)=self.BFGS(data1)
                else: # else, catch all relevant output
                    (logL1,G1,D1,P1,converged1,i1)=self.BFGS(data1)
            else:
                # if one-step efficient estimation, only catch whether converged, and number of iterations
                if onestep:
                    (converged1,i1)=self.Newton(data1,baseline=False,silent=True)
                else: # else, catch all relevant output
                    (logL1,G1,D1,P1,converged1,i1)=self.Newton(data1,baseline=False,silent=True)
            # if one-step efficient estimation
            if onestep:
                # get final estimates from small data
                param1=data1.param
                # set large data as main data, add SNP, and replace baseline effects by estimates from small data
                data1=self.data.copy()
                data1.AddSNP(g,gisnan)
                data1.param=param1
                # take one Newton step
                (logL1,G1,D1,P1,_,_)=self.Newton(data1,baseline=False,silent=True,onestepfinal=True)
            # retrieve final parameter estimates from full data
            param1=data1.param
            # if LRT required
            if args.lrt:
                # if any difference in observations considered in baseline model vs. SNP-specific model (due to SNP missingness)
                if self.data.N!=data1.N:
                    # set large data as main data, keeping only observations where the SNP is not missing
                    data0=self.data.copy()
                    data0.Subsample(~gisnan)
                    # use Newton or BFGS to get baseline estimates for the subset of observations in SNP-specific model               
                    if args.bfgs:
                        (logL0,converged0)=self.BFGS(data0,reestimation=True)
                    else:
                        (logL0,converged0)=self.Newton(data0,baseline=False,silent=True,reestimation=True)
                    # only keep logL0 if converged, otherwise set to NaN
                    if not(converged0):
                        logL0=np.nan
                else: # if no differences, set logL0 as what's stored in the main data
                    logL0=self.data.logL0
            else:
                # set logL0 to NaN
                logL0=np.nan
        else: # else don't even try
            (param1,logL1,logL0,G1,D1,P1,converged1,i1)=(None,None,None,None,[0],None,False,None)
        # calculate and store estimates, standard errors, etc.
        outputline=CalculateStats(j,ngeno,eaf,hweP,param1,logL1,logL0,G1,D1,P1,converged1,i1,data1)
        # update progress bar
        self.pbar.update(1)
        # return output line with results
        return outputline

def CalculateStats(j,ngeno,eaf,hweP,param1,logL1,logL0,G1,D1,P1,converged1,i1,data1):
    # read line from bim file, strip trailing newline, split by tabs
    snpline=linecache.getline(args.bfile+extBIM,j+1).rstrip(eol).split(sep)
    # get chromosome number, snp ID, baseline allele, and effect allele
    snpchr=snpline[0]
    snpid=snpline[1]
    snpbaseallele=snpline[4]
    snpeffallele=snpline[5]
    # build up line to write
    outputline=snpchr+sep+snpid+sep+snpbaseallele+sep+str(1-eaf)+sep\
               +snpeffallele+sep+str(eaf)+sep+str(hweP)+sep+str(ngeno)+sep\
               +str(data1.n1)+sep+str(data1.n2)+sep+str(data1.nboth)+sep+str(i1)
    # define sequence of NaNs for missing stuff, if any
    nanfield=sep+'nan'
    # when doing LRT per SNP, we have 30 SNP-specific fields that can be missing
    if args.lrt:
        nanfields=30*nanfield
    else: # otherwise, 28 SNP-specific fields
        nanfields=28*nanfield
    # if converged and Hessian pd, calculate stats and write to assoc file
    if converged1 and min(D1)>MINEVAL:
        # get OPG
        GGT1=(G1.reshape((data1.k*TOTALCOLS,G1.shape[2])))@((G1.reshape((data1.k*TOTALCOLS,G1.shape[2]))).T)
        # get inverse of -Hessian, variance matrix, and standard errors of parameter estimates
        invMH1=(P1/D1[None,:])@P1.T
        param1Var=invMH1@GGT1@invMH1
        param1SE=((np.diag(param1Var))**0.5).reshape((data1.k,TOTALCOLS))
        # get covariance matrix for estimates of beta1 and beta2
        b1Var=(param1Var.reshape((data1.k,TOTALCOLS,data1.k,TOTALCOLS)))[:,BETA1COL,:,BETA1COL]
        b2Var=(param1Var.reshape((data1.k,TOTALCOLS,data1.k,TOTALCOLS)))[:,BETA2COL,:,BETA2COL]
        # get individual-specific standard deviations for Y1 and Y2
        sig1=np.exp((data1.x*param1[None,:,BETA1COL]).sum(axis=1))
        sig2=np.exp((data1.x*param1[None,:,BETA2COL]).sum(axis=1))
        # caclulate the APE of regressors on standard deviations
        snpAPEsig1=param1[-1,BETA1COL]*sig1[data1.y1notnan].mean()
        snpAPEsig2=param1[-1,BETA2COL]*sig2[data1.y2notnan].mean()
        # calculate derivative of those APEs with respect to the SNPs
        deltaAPEsig1=param1[-1,BETA1COL]*(data1.x*sig1[:,None])[data1.y1notnan,:].mean(axis=0)
        deltaAPEsig2=param1[-1,BETA2COL]*(data1.x*sig2[:,None])[data1.y2notnan,:].mean(axis=0)
        deltaAPEsig1[-1]=sig1[data1.y1notnan].mean()+deltaAPEsig1[-1]
        deltaAPEsig2[-1]=sig2[data1.y2notnan].mean()+deltaAPEsig2[-1]
        # calculate the standard error of the APEs using the Delta method
        snpAPEsig1SE=(deltaAPEsig1@b1Var@deltaAPEsig1)**0.5
        snpAPEsig2SE=(deltaAPEsig2@b2Var@deltaAPEsig2)**0.5
        # get covariance matrix for estimates of gamma
        gcVar=(param1Var.reshape((data1.k,TOTALCOLS,data1.k,TOTALCOLS)))[:,GAMMACOL,:,GAMMACOL]
        # get individual-specific delta (i.e. precursor of rho)
        delta=np.exp((data1.x*param1[None,:,GAMMACOL]).sum(axis=1))
        # use delta, to calculate individual-specific effect of SNP on rho, and average to get APE
        snpAPErho=param1[-1,GAMMACOL]*(2*delta/((delta+1)**2))[data1.ybothnotnan].mean()
        # calculate derivative of those APEs with respect to the SNPs
        deltaAPErho=2*param1[-1,GAMMACOL]\
            *(data1.x*(((1-delta)/((1+delta)**3))[:,None]))[data1.ybothnotnan,:].mean(axis=0)
        deltaAPErho[-1]=(2*delta/((delta+1)**2))[data1.ybothnotnan].mean()+deltaAPErho[-1]
        # calculate the standard error of the APEs using the Delta method
        snpAPErhoSE=(deltaAPErho@gcVar@deltaAPErho)**0.5
        # get SNP effect, standard error, inferences
        snp=param1[-1,:]
        snpSE=param1SE[-1,:]
        snpWald=(snp/snpSE)**2
        snpPWald=1-stats.chi2.cdf(snpWald,1)
        # get estimated covariance matrix of all SNP-specific effects
        snpVar=(param1Var.reshape((data1.k,TOTALCOLS,data1.k,TOTALCOLS)))[-1,:,-1,:]
        jointWald=(snp[None,:]*np.linalg.inv(snpVar)*snp[:,None]).sum()
        jointPWald=1-stats.chi2.cdf(jointWald,TOTALCOLS)
        # add results for effect on E[Y1] to line
        outputline+=sep+str(snp[ALPHA1COL])+sep+str(snpSE[ALPHA1COL])+sep+str(snpWald[ALPHA1COL])+sep+str(snpPWald[ALPHA1COL])
        # add results for effect on E[Y2] to line
        outputline+=sep+str(snp[ALPHA2COL])+sep+str(snpSE[ALPHA2COL])+sep+str(snpWald[ALPHA2COL])+sep+str(snpPWald[ALPHA2COL])
        # add results for effect on Stdev(Y1) to line
        outputline+=sep+str(snp[BETA1COL])+sep+str(snpSE[BETA1COL])+sep+str(snpWald[BETA1COL])+sep+str(snpPWald[BETA1COL])\
                    +sep+str(snpAPEsig1)+sep+str(snpAPEsig1SE)
        # add results for effect on Stdev(Y2) to line
        outputline+=sep+str(snp[BETA2COL])+sep+str(snpSE[BETA2COL])+sep+str(snpWald[BETA2COL])+sep+str(snpPWald[BETA2COL])\
                    +sep+str(snpAPEsig2)+sep+str(snpAPEsig2SE)
        # add results for effect on Stdev(Y2) to line
        outputline+=sep+str(snp[GAMMACOL])+sep+str(snpSE[GAMMACOL])+sep+str(snpWald[GAMMACOL])+sep+str(snpPWald[GAMMACOL])\
                    +sep+str(snpAPErho)+sep+str(snpAPErhoSE)
        # add results for Wald test on joint significance to line
        outputline+=sep+str(jointWald)+sep+str(jointPWald)
        # if we do LRT
        if args.lrt:
            # calculate statistic
            snpLRT=2*data1.n*(logL1-logL0)
            snpPLRT=1-stats.chi2.cdf(snpLRT,TOTALCOLS)
            # add LRT results to line
            outputline+=sep+str(snpLRT)+sep+str(snpPLRT)
    else:
        # if model not converged: set NaNs as SNP results
        outputline+=nanfields
    # add eol to line
    outputline+=eol
    # return output line
    return outputline

# define class for reading genotypes
class GenoDataReader:
    
    def __init__(self):
        # print update
        logger.info('Reading genotype data')
        # connect to bed file
        connbed=open(args.bfile+extBED,'rb')
        # check if first three bytes bed file are correct
        if ord(connbed.read(1))!=(ord(binBED1)) or ord(connbed.read(1))!=(ord(binBED2))\
                                                or ord(connbed.read(1))!=(ord(binBED3)):
            raise ValueError(args.bfile+extBED+' not a valid PLINK .bed file')
        # close connection to bed file
        connbed.close()
        # count number of observations and SNPs in PLINK data
        self.nPLINK=CountLines(args.bfile+extFAM)
        self.M=CountLines(args.bfile+extBIM)
        # print update
        logger.info('Found '+str(self.nPLINK)+' individuals in '+args.bfile+extFAM)
        logger.info('Found '+str(self.M)+' SNPs in '+args.bfile+extBIM)
        # initialise starting point and end point to analyses
        self.Mstart=0
        self.Mend=self.M
        # count number of complete and total bytes per SNP,
        (self.BytesC,_,self.BytesT)=GetNumberOfGroups(self.nPLINK,nperbyte)
        # compute expected number of bytes in .bed file: 3 magic bytes + data
        BytesExp=3+self.BytesT*self.M
        # get observed number of bytes in .bed file
        BytesObs=(os.stat(args.bfile+extBED)).st_size
        # throw error if mismatch
        if BytesExp!=BytesObs:
            raise ValueError('Unexpected number of bytes in '+args.bfile+extBED+'. File corrupted?')
        # compute rounded n
        self.nPLINKT=self.BytesT*nperbyte
        # if SNP range has been provided
        if args.snp is not None:
            self.Mstart=args.snp[0]-1 # start indexing at zero
            self.Mend=args.snp[1] # and use until instead of up until
            if self.Mstart>self.M:
                raise ValueError('Index for first SNP to analyse based on option '\
                                 +'--snp exceeds number of SNPs in data')
            if self.Mend>self.M:
                raise ValueError('Index for last SNP to analyse based on option '\
                                 +'--snp exceeds number of SNPs in data')
            if self.Mstart>=self.Mend:
                raise ValueError('Index for first SNP exceeds index for last SNP '\
                                 +'to analyse based on option --snp')
        # calculate how many SNPs to analyse in total and how many complete output blocks
        self.Mt=self.Mend-self.Mstart
        (self.ResultBlocksC,_,self.ResultBlocksT)=GetNumberOfGroups(self.Mt,RESULTSMBLOCK)
        # count how many complete SNP blocks, remainder, and total no. of blocks
        # when simulating phenotypes
        (self.BlocksC,_,self.BlocksT)=GetNumberOfGroups(self.M,READMBLOCK)
    
    def ReadSNP(self,j):
        # calculate offset
        offset=3+(self.BytesT*j)
        # return genotype bytes on SNP j
        g=self.ReadSNPBytes(offset)
        # drop rows corresponding to empty bits of last byte for each SNP
        g=g[0:self.nPLINK]
        # find rows with missing genotype
        gisnan=(g==1)
        # set missing values to zero
        g[gisnan]=0
        # recode genotype, where 0=homozygote A1, 1=heterozygote, 2=homozygote A2
        g[g==2]=1
        g[g==3]=2
        # return genotype and missingness vector
        return g, gisnan
    
    def ReadBlock(self,b):
        # find index of starting and ending SNP in this block
        m0=b*READMBLOCK
        m1=min(self.M,(b+1)*READMBLOCK)
        # count number of SNP in this block
        m=m1-m0
        # calculate offset
        offset=3+(self.BytesT*m0)
        # return genotype bytes on m SNPs
        g=self.ReadSNPBytes(offset,m)
        # throw error if a genotype is missing; users should address this before simulation
        if (g==1).sum()>0:
            raise ValueError('Missing genotypes in PLINK files, which is not permissible in simulation of phenotypes; use e.g. `plink --bfile '+str(args.bfile)+' --geno 0 --make-bed --out '+str(args.bfile)+'2` to obtain PLINK binary dataset without missing values')
        # recode genotype where 0=homozygote A1, 1=heterozygote, 2=homozygote A2
        g[g==2]=1
        g[g==3]=2
        # reshape to genotype matrix
        g=g.reshape((m,self.nPLINKT)).T
        # drop rows corresponding to empty bits of last byte for each SNP
        g=g[0:self.nPLINK,:]
        # return genotypes for SNP block b
        return g
    
    def ReadSNPBytes(self,offset,m=1):
        # calculate how many bytes in this read
        bytesread=m*self.BytesT
        # connect to bed file
        connbed=open(args.bfile+extBED,'rb')
        # go to starting point of SNP m0 in BED file
        connbed.seek(offset,0)
        # read bytes
        gbytes=np.frombuffer(connbed.read(bytesread),dtype=np.uint8)
        # close connection to bed file
        connbed.close()
        # calculate how many distinct genotypes in this read
        nread=int(nperbyte*bytesread)
        # initialise genotypes for this read as empty
        g=np.empty(nread,dtype=np.uint8)
        # per individual in each byte
        for i in range(nperbyte):
            # take difference between what is left of byte after removing 2 bits
            gbytesleft=gbytes>>2
            g[np.arange(i,nread+i,nperbyte)]=gbytes-(gbytesleft<<2)
            # keep part of byte that is left
            gbytes=gbytesleft
        # return read genotype bytes
        return g

    def SimulateY(self):
        # initialise random-numer generators
        rng=np.random.default_rng(args.seed_pheno)
        rngeffects=np.random.default_rng(args.seed_effects)
        # initialise linear parts of expectations, variances, and correlation
        xalpha1=np.zeros(self.nPLINK)
        xalpha2=np.zeros(self.nPLINK)
        xbeta1=np.zeros(self.nPLINK)
        xbeta2=np.zeros(self.nPLINK)
        xgamma=np.zeros(self.nPLINK)
        # connect to read bim file
        connbim=open(args.bfile+extBIM,'r')
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
        # print update
        logger.info('Reading in '+args.bfile+extBED+' in blocks of '+str(READMBLOCK)+' SNPs')
        # for each blok
        for b in tqdm(range(self.BlocksT)):
            # read genotypes
            g=self.ReadBlock(b)
            m=g.shape[1]
            # calculate empirical AFs
            eaf=g.mean(axis=0)/2
            # calculate standardised SNPs
            gs=(g-2*(eaf[None,:]))/(((2*eaf*(1-eaf))**0.5)[None,:])
            # draw factors for SNP effects on expectations
            gf1=rngeffects.normal(size=m)
            gf2=rngeffects.normal(size=m)
            # draw correlated SNP effects on expectations
            alpha1=gf1*((args.h2y1/self.M)**0.5)
            alpha2=((args.rg*gf1)+(((1-(args.rg**2))**0.5)*gf2))*((args.h2y2/self.M)**0.5)
            # draw SNP effects on variances and correlation
            beta1=rngeffects.normal(size=m)*((args.h2sig1/self.M)**0.5)
            beta2=rngeffects.normal(size=m)*((args.h2sig2/self.M)**0.5)
            gamma=args.rhoband*(rngeffects.normal(size=m)*((args.h2rho/self.M)**0.5))
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
        # close connection effects file and bim file
        conneff.close()
        connbim.close()
        # draw error terms for sigma and rho
        esig1=rng.normal(size=self.nPLINK)*((1-args.h2sig1)**0.5)
        esig2=rng.normal(size=self.nPLINK)*((1-args.h2sig2)**0.5)
        erho=args.rhoband*(rng.normal(size=self.nPLINK)*((1-args.h2rho)**0.5))
        # calculate standard deviations
        sig1=np.exp(-1+esig1+xbeta1)
        sig2=np.exp(-1+esig2+xbeta2)
        # find intercept for linear part of correlation, such that average
        # correlation equals rhomean
        gamma0=np.log((1+args.rhomean)/(1-args.rhomean))
        delta=np.exp(gamma0+erho+xgamma)
        rho=(delta-1)/(delta+1)
        # draw noise factors
        eta1=rng.normal(size=self.nPLINK)
        eta2=rng.normal(size=self.nPLINK)
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
    # report how many SNPs can be simulated/read/written at once
    logger.info('Simulating SNPs and writing .bim file in blocks of '+str(READMBLOCK)+' SNPs')
    # count how many complete blocks, remainder, and no. of blocks including remainder
    (BlocksC,_,BlocksT)=GetNumberOfGroups(M,READMBLOCK)
    # count number of complete bytes per SNP, and total bytes per SNP (including remainder bytes)
    (BytesC,_,BytesT)=GetNumberOfGroups(n,nperbyte)
    # compute rounded n
    nT=BytesT*nperbyte
    # set counter for total number of SNPs handled for export to .bim
    i=0
    # for each blok
    for b in tqdm(range(BlocksT)):
        # find index for first SNP and last SNP in block
        m0=b*READMBLOCK
        m1=min(M,(b+1)*READMBLOCK)
        # count number of SNP in this block
        m=m1-m0
        # draw allele frequencies between (TAUTRUEMAF,1-TAUTRUEMAF)
        f=TAUTRUEMAF+(1-2*TAUTRUEMAF)*rng.uniform(size=m)
        # initialise genotype matrix and empirical AF
        g=np.zeros((nT,m),dtype=np.uint8)
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
            g[0:n,notdone]=thisg
            # calculate empirical AF
            eaf[notdone]=thisg.mean(axis=0)/2
            # find SNPs with insufficient variation
            notdone=(eaf*(1-eaf))<(TAUDATAMAF*(1-TAUDATAMAF))
            mnotdone=notdone.sum()
        # recode s.t. 0=zero alleles; 2=one allele; 3=two alleles; 1=missing
        g=2*g
        g[g==4]=3
        # within each byte: 2 bits per individual; 4 individuals per byte in total
        base=np.array([2**0,2**2,2**4,2**6]*BytesT,dtype=np.uint8)
        # per SNP, per byte: aggregate across individuals in that byte
        exportbytes=(g*base[:,None]).reshape(BytesT,nperbyte,m).sum(axis=1).astype(np.uint8)
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

def GetNumberOfGroups(elements,size):
    '''
    Given number of discrete elements divided into groups of equal size,
    calculates how many complete groups there are, what the remainder is,
    and the total number of groups (i.e. including a potential remainder group)

    Args:
        elements (int): number of discrete elements
        size (int): number of elements per group

    Returns:
        completegroups (int): number of full groups
        remainder (int): number of elements in remainder group
        totalgroups (int): number group groups, including potential remainder group
    '''
    completegroups=int(elements/size)
    remainder=elements%size
    totalgroups=completegroups+(remainder>0)
    return completegroups,remainder,totalgroups

def CountLines(filename):
    '''
    Efficiently calculate the number of lines in a file
    
    Args:
        filename (str): name of the file
    
    Returns:
        lines (int): number of lines in the file
    '''
    with open(filename, "r+") as f:
        buffer=mmap.mmap(f.fileno(), 0)
        lines=0
        readline=buffer.readline
        while readline():
            lines+=1
    return lines

def ConvertSecToStr(t):
    '''
    Convert number of seconds to string of the form '1d:3h:46m:40s'
    
    Args:
        t (float): runtime in seconds
    
    Returns:
        f (str): formatted time string
    '''
    # round to whole miliseconds
    r=int(1000*t)+(((1000*t)-int(1000*t))>=0.5)
    # get number of days, hours, minutes, seconds
    (d,r)=divmod(r,1000*60*60*24)
    (h,r)=divmod(r,1000*60*60)
    (m,r)=divmod(r,1000*60)
    (s,ms)=divmod(r,1000)
    # add rounded decimal part back to remaining seconds
    s+=(ms/1000)
    # begin with empty string, and append days, hours, minutes
    f = ''
    if d > 0:
        f += '{D}d:'.format(D=d)
    if h > 0:
        f += '{H}h:'.format(H=h)
    if m > 0:
        f += '{M}m:'.format(M=m)
    # wrap up string with number of seconds
    f += '{S}s'.format(S=s)
    return f

def positive_int(string):
    '''
    Try to parse input argument as positive integer
    '''
    try:
        val=int(string)
    except:
        raise argparse.ArgumentTypeError(string+" is not a positive integer")
    if val>0:
        return val
    else:
        raise argparse.ArgumentTypeError("%s is not a positive integer" % val)

def number_between_0_1(string):
    '''
    Try to parse input argument as number in (0,1)
    '''
    try:
        val=float(string)
    except:
        raise argparse.ArgumentTypeError(string+" is not a number in the interval (0,1)")
    if val>0 and val<1:
        return val
    else:
        raise argparse.ArgumentTypeError("%s is not a number in the interval (0,1)" % val)

def number_between_m1_p1(string):
    '''
    Try to parse input argument as number in (-1,1)
    '''
    try:
        val=float(string)
    except:
        raise argparse.ArgumentTypeError(string+" is not a number in the interval (-1,1)")
    if (val**2)<1:
        return val
    else:
        raise argparse.ArgumentTypeError("%s is not a number in the interval (-1,1)" % val)

def CalculateHessian(X,wH):
    '''
    Calculate Hessian of log-likelihood function, given matrix of regressors
    and individual-specific weights for the various submatrices of the Hessian

    Args:
        X (ndarray): N-by-K matrix of regressors
        wH (ndarray): N-by-(TOTALCOLS-by-TOTALCOLS) array of individual-specific weights
    
    Returns:
        H (ndarray): (K-by-TOTALCOLS)-by-(K-by-TOTALCOLS) Hessian array
    '''
    K=X.shape[1]
    H=np.empty((K,TOTALCOLS,K,TOTALCOLS))
    for i in range(TOTALCOLS):
        for j in range(i,TOTALCOLS):
            # calculate Hessian for component i vs. j and j vs. i
            H[:,i,:,j]=(X.T@(X*wH[:,None,i,j]))
            if j>i:
                # use symmetry to find out counterparts
                H[:,j,:,i]=H[:,i,:,j]
    return H

def BendEVDSymPSD(A):
    '''
    Gets bended eigenvalue decomposition of symmetric psd matrix A
    
    Args:
        A (ndarray): symmetric n-by-n psd matrix
    
    Returns:
        D (ndarray): vector of n (bended eigenvalues)
        P (ndarray): n-by-n matrix of eigenvectors
    '''
    # stabilise A
    A=(A+A.T)/2
    # get eigenvalue decomposition of A
    (D,P)=np.linalg.eigh(A)
    # if lowest eigenvalue too low
    if min(D)<MINEVAL:
        # bend s.t. Newton becomes more like gradient descent
        a=(MINEVAL-D.min())/(1-D.min())
        Dadj=(1-a)*D+a
        # return bended eigenvalues, eigenvectors, and original eigenvalues
        return Dadj,P,D
    # else return original values and vectors
    return D,P,D

# main function
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
    # try gcat
    try:
        # Parse input arguments
        ParseInputArguments()
        # Initialise logger
        InitialiseLogger()
        # Print welcome screen
        ShowWelcome()
        # Perform basic checks on input arguments
        CheckInputArgs()
        # Simulate genotypes if necessary
        if simulg:
            logger.info(eol+'SIMULATING GENOTYPES')
            SimulateG()
        # Initialise genotype reader
        snpreader=GenoDataReader()
        # Simulate phenotypes if necessary
        if simuly:
            logger.info(eol+'SIMULATING PHENOTYPES')
            snpreader.SimulateY()
        # Analyse data if necessary
        if not(args.simul_only):
            logger.info(eol+'PERFORMING GCAT')
            # Initialise analyser and estimate baseline model using Newton's method with line search
            analyser=Analyser(snpreader)
            # Estimate SNPs-specific model
            analyser.EstimateSNPModels()
    except Exception:
        # print the traceback
        logger.error(traceback.format_exc())
        # wrap up with final error message
        logger.error('Error: GCAT did not exit properly. Please inspect the log file.')
        logger.info('Run `python ./gcat.py -h` to show all options')
    finally:
        # print total time elapsed
        logger.info('Total time elapsed: '+ConvertSecToStr(time.time()-t0))
        logger.info('Current memory usage is ' + str(int((process.memory_info().rss)/(1024**2))) + 'MB')

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
    parser.add_argument('--bfgs', action = 'store_true',
                    help = 'option to estimate SNP-specific models using BFGS algorithm; cannot be combined with --simul-only')
    parser.add_argument('--section', action = 'store_true',
                    help = 'option to turn on golden section when estimating SNP-specific models using either a BFGS or Newton procedure; cannot be combined with --simul-only')
    parser.add_argument('--one', metavar = '', default = None, nargs= '+',
                    help = '\b\b\b\b\b\b\b\b\b\b\b\b\b\b\bNUMBER INTEGER option to perform 1-step efficient estimation, by randomly sampling a fraction (1st input argument; between 0 and 1) of the observations, to obtain estimates that serve as starting point for a single Newton step based on full data; 2nd input argument is the seed for the random-number generator for the random sampling; cannot be combined with --simul-only')
    parser.add_argument('--lrt', action = 'store_true',
                    help = 'option to perform a likelihood-ratio test for the joint significance per SNP; this test can be more reliable than the Wald test for joint significance; WARNING: this test can double the CPU time for SNPs with any missingness; WARNING: this test can be overly conservative when combined with --one; cannot be combined with --simul-only')
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
    # set global Booleans
    global simulg, simuly, covars, onestep, linesearch
    # set all those to False by default, change if needed based on input
    simulg=False
    simuly=False
    covars=False
    onestep=False
    linesearch=False
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
    if args.bfgs and args.simul_only:
        raise SyntaxError('--simul-only cannot be combined with --bfgs')
    if args.section and args.simul_only:
        raise SyntaxError('--simul-only cannot be combined with --section')
    if args.section:
        linesearch=True
    if args.snp is not None:
        if len(args.snp)!=2:
            raise SyntaxError('--snp needs to be followed by two integers')
        elif args.snp[1]<args.snp[0]:
            raise SyntaxError('--snp requires 1st integer <= 2nd integer')
    if args.one is not None:
        if args.simul_only:
            raise SyntaxError('--one cannot be combined with --simul-only')
        if len(args.one)!=2:
            raise SyntaxError('--one needs to be followed by a number between zero and one (fraction of individuals that will be randomly sample) and a positive integer (to set the random-number generator for random sampling')
        args.one[0]=number_between_0_1(args.one[0])
        args.one[1]=positive_int(args.one[1])
        onestep=True
    if args.lrt and args.simul_only:
        raise SyntaxError('--simul-only cannot be combined with --lrt')

# invoke main function when called on as a script
if __name__ == '__main__':
    main()
