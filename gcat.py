# IMPORTS
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

## CONSTANTS ##
# numerical method constants
MAXITER=50                  # max no. of iterations Newton, BFGS, line search
TOL=1E-12                   # threshold for Newton and BFGS
ARMIJO=1E-4                 # threshold for golden section (Armijo's rule)
THETAINV=2/(1+5**0.5)       # constant 1 in golden section
ONEMINTHETAINV=1-THETAINV   # constant 2 in golden section
MINEVAL=1E-3                # lower bound for eigenvalues of minus Hessian/n

# bounds on dimensionality
MINN=10                     # lower bound on sample size
MINM=1                      # lower bound on number of SNPs
RESULTSMBLOCK=100           # number of SNPs per block (writing GCAT results)
SIMULMBLOCK=1000            # number of SNPs per block (simulation)

# alleles and minor allele frequency (MAF) in simulation of genotypes
ALLELES=['A','C','G','T']   # set of possible alleles to draw from
TAUTRUEMAF=0.05             # lower bound on true MAF (simulation)
TAUDATAMAF=0.01             # lower bound on empirical MAF (simulation)

# columns assigned to different types of parameters
ALPHA1COL=0                 # column for effects of regressors on E[Y1]
ALPHA2COL=1                 # column for effects of regressors on E[Y2]
BETA1COL=2                  # column for effects of regressors on Var(Y1)
BETA2COL=3                  # column for effects of regressors on Var(Y2)
GAMMACOL=4                  # column for effects of regressors on Corr(Y1,Y2)
TOTALCOLS=5                 # total number of columns

# identifier individuals across files (e.g. --bfile,--pheno,--covar)
IDENTIFIER=['FID','IID']

# constants for PLINK binary files
extBED='.bed'               # extension BED file
extBIM='.bim'               # extension BIM file
extFAM='.fam'               # extension FAM file
binBED1=bytes([0b01101100]) # 1st magic byte BED file
binBED2=bytes([0b00011011]) # 2nd magic byte BED file
binBED3=bytes([0b00000001]) # 3rd magic byte BED file
nperbyte=4                  # number of individuals per full byte in BED file
fieldsFAM=['FID','IID','PID','MID','SEX','PHE'] # fields of FAM file

# other extensions, prefixes, headers
outDEFAULT='output'         # prefix output files when --out not used
extLOG='.log'               # extension log file
extEFF='.true_effects.txt'  # postfix file with true effects (simulation)
colEFF=['ALPHA1','ALPHA2','BETA1','BETA2','GAMMA'] # col names effects (simul)
extPHE='.phe'               # extension phenotype file (simulation)
extFRQ='.frq'               # extension frequency file (simulation)
extBASE='.baseline_estimates.txt' # postfix file with baseline model estimates
extASSOC='.assoc'           # extension association results file

# separator and end-of-line character
sep='\t'
eol='\n'

# help string per input arguments of tool
help_n='(simulation) number of individuals; at least '+str(MINN)
help_m='(simulation) number of SNPs; at least '+str(MINM)
help_seed_geno='(simulation) seed for random-number generator for genotypes'
help_bfile='prefix of PLINK binary files; cannot be combined with --n, --m,'+\
           ' and/or --seed-geno'
help_h2y1='(simulation) heritability of Y1; between 0 and 1'
help_h2y2='(simulation) heritability of Y2; between 0 and 1'
help_rg='(simulation) genetic correlation between Y1 and Y2; between -1 and 1'
help_h2sig1='(simulation) heritability of linear part of Var(Error term Y1);'+\
            ' between 0 and 1'
help_h2sig2='(simulation) heritability of linear part of Var(Error term Y2);'+\
            ' between 0 and 1'
help_h2rho='(simulation) heritability of linear part of Corr(Error term Y1,'+\
           'Error term Y2); between 0 and 1'
help_rhomean='(simulation) average Corr(Error term Y1,Error term Y2);'+\
             ' between -1 and 1'
help_rhoband='(simulation) probabilistic bandwidth of correlation around'+\
             ' level specified using --rhomean; between 0 and 1'
help_seed_effects='(simulation) seed for random-number generator for'+\
                  ' unscaled true effects of standardised SNPs'
help_seed_pheno='(simulation) seed for random-number generator for phenotypes'
help_pheno='name of phenotype file: should be comma-, space-, or'+\
           ' tab-separated, with one row per individual, with FID and IID as'+\
           ' first two fields, followed by two fields for phenotypes Y1 and'+\
           ' Y2; first row must contain labels (e.g. FID IID HEIGHT log(BMI)'+\
           '); requires --bfile to be specified; cannot be combined with'+\
           ' --h2y1, --h2y2, --rg, --h2sig1, --h2sig2, --h2rho, --rhomean,'+\
           ' --rhoband, --seed-effects, and/or --seed-pheno'
help_covar='name of covariate file: should be comma-, space-, or'+\
           ' tab-separated, with one row per individual, with FID and IID as'+\
           ' first two fields, followed by a field per covariate; first row'+\
           ' must contain labels (e.g. FID IID AGE AGESQ PC1 PC2 PC3 PC4 PC5'+\
           '); requires --pheno to be specified; WARNING: do not include an'+\
           ' intercept in your covariate file, because GCAT always adds an'+\
           ' intercept itself'
help_simul_only='option to simulate data only (i.e. no analysis of simulated'+\
                ' data); cannot be combined with --pheno'
help_bfgs='option to estimate SNP-specific models using BFGS algorithm;'+\
          ' cannot be combined with --simul-only'
help_section='option to turn on golden section when estimating SNP-specific'+\
             ' models using either a BFGS or Newton procedure; cannot be'+\
             ' combined with --simul-only'
help_one='\b\b\b\b\b\b\b\b\b\b\b\b\b\b\bNUMBER INTEGER option to perform'+\
         ' 1-step efficient estimation, by randomly sampling a fraction (1st'+\
         ' input argument; between 0 and 1) of the observations, to obtain'+\
         ' estimates that serve as starting point for a single Newton step'+\
         ' based on full data; 2nd input argument is the seed for the random'+\
         '-number generator for the random sampling; cannot be combined with'+\
         ' --simul-only'
help_lrt='option to perform a likelihood-ratio test for the joint'+\
         ' significance per SNP; this test can be more reliable than the'+\
         ' Wald test for joint significance; WARNING: this test can double'+\
         ' the CPU time for SNPs with any missingness; WARNING: this test'+\
         ' can be overly conservative when combined with --one; cannot be'+\
         ' combined with --simul-only'
help_snp='\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\bINTEGER INTEGER option to analyse'+\
         ' only SNP with index j=s,...,t, where s=1st integer and t=2nd'+\
         ' integer; cannot be combined with --simul-only'
help_out='prefix of output files'

# text for welcome screen
__version__ = 'v0.2'
HEADER=eol
HEADER+='------------------------------------------------------------'+eol
HEADER+='| GCAT (Genome-wide Cross-trait concordance Analysis Tool) |'+eol
HEADER+='------------------------------------------------------------'+eol
HEADER+='| BETA {V}, (C) 2024 Ronald de Vlaming                    |'.\
    format(V=__version__)+eol
HEADER+='| Vrije Universiteit Amsterdam                             |'+eol
HEADER+='| GNU General Public License v3                            |'+eol
HEADER+='------------------------------------------------------------'+eol

class Data:
    '''
    Class for holding numerical data on X and Y, parameter estimates, and some
    secondary data for efficient estimation (e.g. when adding new regressor
    or when considering subset of individuals)

    Different instances can be used to estimate SNP-specific model with random
    subsampling, without parallel tasks interfering
    '''

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
            # count number of observations with nonmissing data on covariates
            nxnotnan=(~xmissing).sum()
            # throw error if zero
            if nxnotnan==0:
                raise ValueError('No observations in '+args.covar\
                                 +' without any missingness that can be'\
                                    +' matched to '+args.bfile+extFAM)
            # for row with any missing covariate: set y1,y2=NaN and x=0
            self.x[xmissing,:]=0
            self.y1[xmissing]=np.nan
            self.y2[xmissing]=np.nan
            # calculate min eigenvalue of X'X/n (for samples w/o missingness)
            (Dxtx,_)=np.linalg.eigh((self.x.T@self.x)/nxnotnan)
            # if too low, throw error:
            if min(Dxtx)<MINEVAL:
                raise ValueError('Regressors in '+args.covar+' have too much'\
                                 +' multicollinearity. Intercept in your '\
                                 +' covariates file? Please remove it. Or set'\
                                 +' of dummies that is perfectly collinear'\
                                 +' with intercept? Please remove a category.'\
                                 +' Recall: GCAT adds intercept itself.')
            # find missingness, recode, and get counts
            self.FindAndRecodeMissingness()
            self.GetCounts()
            # initialise parameters and clean
            self.InitialiseParams()
            self.Clean()
    
    def InitialiseParams(self):
        # report update stats
        logger.info('Filtered out observations with missingness')
        logger.info('Found '+str(self.n1)+' individuals with complete data'+\
                    ' for '+self.y1label+' (and covariates, if applicable)')
        logger.info('Found '+str(self.n2)+' individuals with complete data'+\
                    ' for '+self.y2label+' (and covariates, if applicable)')
        logger.info('Of these '+str(self.nboth)+\
                    ' individuals have data on both traits')
        # terminate if no. of observations too low
        if self.nboth<MINN:
            raise ValueError('You need at least '+str(MINN)+' individuals '+\
                             'with data on both trait')
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
        # get regression coefficients for baseline regressors
        # w.r.t. Var(y1) and Var(y2)
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
        self.wL=np.zeros(self.nrow)
        self.wG=np.zeros((self.nrow,TOTALCOLS))
        self.wH=np.zeros((self.nrow,TOTALCOLS,TOTALCOLS))
        # initialise log-likelihood baseline model at -infinity
        self.logL=-np.inf
    
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
        # count number of individuals with either y1, y2, or both
        # (which is the number of supposedly independent observations)
        self.n=self.y1ory2notnan.sum()
        # count total number of observation in multivariate model
        self.N=self.n1+self.n2
        # count dimensionality
        self.nrow=self.x.shape[0]
        self.k=self.x.shape[1]
    
    def copysubset(self,keep=None):
        # if keep is not defined, keep all observations when copying
        if keep is None:
            keep=np.ones(self.nrow,dtype=np.bool_)
        # make copy of main data, in form of Data instance
        data=Data(self.x[keep,:].copy(),self.xlabels.copy(),\
                  self.y1[keep].copy(),self.y1label,
                  self.y2[keep].copy(),self.y2label,copy=True)
        # also copy booleans and indices of remaining PLINK data
        data.y1notnan=self.y1notnan[keep].copy()
        data.y2notnan=self.y2notnan[keep].copy()
        data.ybothnotnan=self.ybothnotnan[keep].copy()
        data.y1ory2notnan=self.y1ory2notnan[keep].copy()
        data.ind=self.ind[keep].copy()
        # also copy parameter estimates, individual-specific weights, logL
        data.param=self.param.copy()
        data.wL=self.wL[keep].copy()
        data.wG=self.wG[keep,:].copy()
        data.wH=self.wH[keep,:,:].copy()
        data.logL=self.logL
        # update counts
        data.GetCounts()
        # return new Data instance
        return data
    
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
        self.wL=self.wL[keep]
        self.wG=self.wG[keep,:]
        self.wH=self.wH[keep,:,:]
        # store only the PLINK entries this corresponds to
        self.ind=self.ind[keep]
        # update counts
        self.GetCounts()

    def AddSNP(self,g,gisnan):
        # consider genotypes + missingness only for individuals still in data
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

class Analyser:
    '''
    Class for reading phenotypes and covariates, and for perfoming analyses
    '''
    def __init__(self,snpreader):
        # print update
        logger.info('Reading phenotype data')
        # read fam file
        famfilename=args.bfile+extFAM
        famdata=pd.read_csv(famfilename,sep=sep,header=None,names=fieldsFAM)
        self.nPLINK=famdata.shape[0]
        logger.info('Found '+str(self.nPLINK)+' individuals in '+famfilename)
        # retain only FID and IID of fam data
        DATA=famdata.iloc[:,[0,1]]
        # read pheno file
        ydata=pd.read_csv(args.pheno,sep=sep)
        # check two phenotypes in phenotype data
        if ydata.shape[1]!=4:
            raise ValueError(args.pheno+' does not contain exactly 2 traits')
        # get phenotype labels
        y1label=ydata.columns[2]
        y2label=ydata.columns[3]
        # print update
        logger.info('Found '+str(ydata.shape[0])+' individuals and 2 traits,'+\
                    ' labelled '+y1label+' and '+y2label+', in '+args.pheno)
        # left join FID,IID from fam with pheno data
        DATA=pd.merge(left=DATA,right=ydata,how='left',\
                      left_on=IDENTIFIER,right_on=IDENTIFIER)
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
                raise ValueError('No covariates found in '+args.covar)
            # print update
            logger.info('Found '+str(xdata.shape[0])+' individuals and '\
                         +str(xdata.shape[1]-2)+' covariates in '+args.covar)
            # left joint data with covariates
            DATA=pd.merge(left=DATA,right=xdata,how='left',\
                          left_on=IDENTIFIER,right_on=IDENTIFIER)
            # retrieve matched covariates baseline model, and add intercept
            x=np.hstack((np.ones((self.nPLINK,1)),DATA.values[:,4:]))
            xlabels=['intercept']+xdata.iloc[:,2:].columns.to_list()
        else: # else set intercept as only covariate
            x=np.ones((self.nPLINK,1))
            xlabels=['intercept']
        # initialise main data and parameters, keep only observations with
        # no missingness in x, and at most one trait is missing
        self.data=Data(x,xlabels,y1,y1label,y2,y2label)
        # store genotype reader
        self.snpreader=snpreader
        # estimate baseline model
        logger.info('Estimating baseline model')
        converged0=Newton(self.data)
        # if baseline model did not converge: throw error
        if not(converged0):
            raise RuntimeError('Estimates baseline model not converged')
        # write baseline model estimates to output file
        pd.DataFrame(self.data.param,columns=colEFF,\
                     index=self.data.xlabels).to_csv(args.out+extBASE,sep=sep)
        # if one-step efficient estimation
        if onestep:
            # initialise random-number generator
            rng=np.random.default_rng(args.one[1])
            # randomly sample the desired proportion
            keep=np.sort(rng.permutation(self.data.nrow)\
                         [:min(int(self.nPLINK*args.one[0]),self.data.nrow)])
            logger.info('Keeping random subsample of '+str(len(keep))\
                        +' individuals for one-step efficient estimation')
            # make copy of the data, keeping the desired subset
            self.datasmall=self.data.copysubset(keep)
        # report whether BFGS or Newton and whether line search is used
        if args.bfgs and not(linesearch):
            logger.info('Estimation SNP-specific models using BFGS:')
        elif args.bfgs and linesearch:
            logger.info('Estimation SNP-specific models using BFGS'+\
                        ' with golden-section line search:')
        elif not(args.bfgs) and not(linesearch): 
            logger.info('Estimation SNP-specific models using Newton method:')
        else:
            logger.info('Estimation SNP-specific models using Newton method'+\
                        ' with golden-section line search:')
    
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
            m1=min(self.snpreader.Mstart+(b+1)*RESULTSMBLOCK,\
                   self.snpreader.Mend)
            # using parallel execution in block with writing at end of block
            with ThreadPoolExecutor() as executor:
                # analyse snp j
                outputlines=executor.map(self.AnalyseOneSNP,\
                                         [j for j in range(m0,m1)])
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
        # if 1-step efficient estimation
        if onestep:
            # set small data as main data
            data1=self.datasmall.copysubset()
            # and add the SNP of interest
            data1.AddSNP(g,gisnan)
        else: # else: set large data as main data, and add SNP
            data1=self.data.copysubset()
            data1.AddSNP(g,gisnan)
        # estimate, provided nboth>=MINN
        if data1.nboth>=MINN:
            # using BFGS or Newton's method, depending on input
            if args.bfgs:
                # if 1-step efficient estimation, only catch whether converged
                # and number of iterations
                if onestep:
                    (converged1,i1)=BFGS(data1)
                else: # else, catch all relevant output
                    (logL1,GGT1,D1,P1,converged1,i1)=BFGS(data1)
            else:
                # if 1-step efficient estimation, only catch whether converged
                # and number of iterations
                if onestep:
                    (converged1,i1)=Newton(data1,baseline=False,silent=True)
                else: # else, catch all relevant output
                    (logL1,GGT1,D1,P1,converged1,i1)=Newton(data1,\
                                                    baseline=False,silent=True)
            # if 1-step efficient estimation
            if onestep:
                # get final estimates from small data
                param1=data1.param
                # set large data as main data
                data1=self.data.copysubset()
                # add SNP of interest
                data1.AddSNP(g,gisnan)
                # replace baseline estimates by estimates small data with snp
                data1.param=param1
                # take one Newton step
                (logL1,GGT1,D1,P1,_,_)=Newton(data1,baseline=False,\
                                                 silent=True,onestepfinal=True)
            # retrieve final parameter estimates from full data
            param1=data1.param
        else: # else don't even try: just return nan/none values
            (param1,logL1,GGT1,D1,P1,converged1,i1)=\
                (None,None,None,[0],None,False,None)
        # calculate and store estimates, standard errors, etc.
        outputline=self.CalculateStats(j,gisnan,ngeno,eaf,hweP,param1,logL1,\
                                  GGT1,D1,P1,converged1,i1,data1)
        # update progress bar
        self.pbar.update(1)
        # return output line with results
        return outputline

    def CalculateStats(self,j,gisnan,ngeno,eaf,hweP,param1,logL1,\
                    GGT1,D1,P1,converged1,i1,data1):
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
                +str(data1.n1)+sep+str(data1.n2)+sep+str(data1.nboth)+sep\
                +str(i1)
        # define sequence of NaNs for missing stuff, if any
        nanfield=sep+'nan'
        # when doing LRT per SNP: 30 SNP-specific fields that can be missing
        if args.lrt:
            nanfields=30*nanfield
        else: # else: 28 SNP-specific fields
            nanfields=28*nanfield
        # if converged and Hessian pd, calculate stats and write to assoc file
        if converged1 and min(D1)>MINEVAL:
            # get inverse of -Hessian and variance matrix
            invMH1=(P1/D1[None,:])@P1.T
            param1Var=invMH1@GGT1@invMH1
            # get standard errors of parameter estimates
            param1SE=((np.diag(param1Var))**0.5).reshape((data1.k,TOTALCOLS))
            # get covariance matrix for estimates of beta1 and beta2
            b1Var=(param1Var.reshape((data1.k,TOTALCOLS,\
                                    data1.k,TOTALCOLS)))[:,BETA1COL,:,BETA1COL]
            b2Var=(param1Var.reshape((data1.k,TOTALCOLS,\
                                    data1.k,TOTALCOLS)))[:,BETA2COL,:,BETA2COL]
            # get individual-specific standard deviations for Y1 and Y2
            sig1=np.exp((data1.x*param1[None,:,BETA1COL]).sum(axis=1))
            sig2=np.exp((data1.x*param1[None,:,BETA2COL]).sum(axis=1))
            # caclulate the APE of regressors on standard deviations
            snpAPEsig1=param1[-1,BETA1COL]*sig1[data1.y1notnan].mean()
            snpAPEsig2=param1[-1,BETA2COL]*sig2[data1.y2notnan].mean()
            # calculate derivative of those APEs with respect to the SNPs
            deltaAPEsig1=param1[-1,BETA1COL]*\
                        (data1.x*sig1[:,None])[data1.y1notnan,:].mean(axis=0)
            deltaAPEsig2=param1[-1,BETA2COL]*\
                        (data1.x*sig2[:,None])[data1.y2notnan,:].mean(axis=0)
            deltaAPEsig1[-1]=sig1[data1.y1notnan].mean()+deltaAPEsig1[-1]
            deltaAPEsig2[-1]=sig2[data1.y2notnan].mean()+deltaAPEsig2[-1]
            # calculate the standard error of the APEs using the Delta method
            snpAPEsig1SE=(deltaAPEsig1@b1Var@deltaAPEsig1)**0.5
            snpAPEsig2SE=(deltaAPEsig2@b2Var@deltaAPEsig2)**0.5
            # get covariance matrix for estimates of gamma
            gcVar=(param1Var.reshape((data1.k,TOTALCOLS,\
                                    data1.k,TOTALCOLS)))[:,GAMMACOL,:,GAMMACOL]
            # get individual-specific delta (i.e. precursor of rho)
            delta=np.exp((data1.x*param1[None,:,GAMMACOL]).sum(axis=1))
            # calculate APE of SNP on rho
            snpAPErho=param1[-1,GAMMACOL]*\
                    (2*delta/((delta+1)**2))[data1.ybothnotnan].mean()
            # calculate derivative of those APEs with respect to the SNPs
            deltaAPErho=2*param1[-1,GAMMACOL]*(data1.x*(((1-delta)\
                   /((1+delta)**3))[:,None]))[data1.ybothnotnan,:].mean(axis=0)
            deltaAPErho[-1]=(2*delta\
                    /((delta+1)**2))[data1.ybothnotnan].mean()+deltaAPErho[-1]
            # calculate the standard error of the APEs using the Delta method
            snpAPErhoSE=(deltaAPErho@gcVar@deltaAPErho)**0.5
            # get SNP effect, standard error, inferences
            snp=param1[-1,:]
            snpSE=param1SE[-1,:]
            snpWald=(snp/snpSE)**2
            snpPWald=1-stats.chi2.cdf(snpWald,1)
            # get estimated covariance matrix of all SNP-specific effects
            snpVar=(param1Var.reshape((data1.k,TOTALCOLS,\
                                    data1.k,TOTALCOLS)))[-1,:,-1,:]
            jointWald=(snp[None,:]*np.linalg.inv(snpVar)*snp[:,None]).sum()
            jointPWald=1-stats.chi2.cdf(jointWald,TOTALCOLS)
            # add results for effect on E[Y1] to line
            outputline+=sep+str(snp[ALPHA1COL])+sep+str(snpSE[ALPHA1COL])+sep+\
                str(snpWald[ALPHA1COL])+sep+str(snpPWald[ALPHA1COL])
            # add results for effect on E[Y2] to line
            outputline+=sep+str(snp[ALPHA2COL])+sep+str(snpSE[ALPHA2COL])+sep+\
                str(snpWald[ALPHA2COL])+sep+str(snpPWald[ALPHA2COL])
            # add results for effect on Stdev(Y1) to line
            outputline+=sep+str(snp[BETA1COL])+sep+str(snpSE[BETA1COL])+sep+\
                str(snpWald[BETA1COL])+sep+str(snpPWald[BETA1COL])\
                        +sep+str(snpAPEsig1)+sep+str(snpAPEsig1SE)
            # add results for effect on Stdev(Y2) to line
            outputline+=sep+str(snp[BETA2COL])+sep+str(snpSE[BETA2COL])+sep+\
                str(snpWald[BETA2COL])+sep+str(snpPWald[BETA2COL])\
                        +sep+str(snpAPEsig2)+sep+str(snpAPEsig2SE)
            # add results for effect on Stdev(Y2) to line
            outputline+=sep+str(snp[GAMMACOL])+sep+str(snpSE[GAMMACOL])+sep+\
                str(snpWald[GAMMACOL])+sep+str(snpPWald[GAMMACOL])\
                        +sep+str(snpAPErho)+sep+str(snpAPErhoSE)
            # add results for Wald test on joint significance to line
            outputline+=sep+str(jointWald)+sep+str(jointPWald)
            # if we do LRT
            if args.lrt:
                # if any difference in observations considered in baseline vs.
                # SNP-specific model (due to SNP missingness)
                if self.data.N!=data1.N:
                    # get copy full data for individuals with nonmissing SNP
                    data1=self.data.copysubset(~gisnan)
                    # use Newton or BFGS to get null model estimates for
                    # subset of observations in SNP-specific model
                    if args.bfgs:
                        (logL0,converged0)=BFGS(data1,reestimation=True)
                    else:
                        (logL0,converged0)=Newton(data1,baseline=False,\
                                                silent=True,reestimation=True)
                    # only keep logL0 if converged, otherwise set to NaN
                    if not(converged0):
                        logL0=np.nan
                else: # else: get logL0 from estimates baseline model
                    logL0=self.data.logL
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

def Newton(data,baseline=True,silent=False,\
            onestepfinal=False,reestimation=False):
    # set iteration counter to 0 and convergence to false
    i=0
    converged=False
    # if baseline model
    if baseline:
        # fully calculate log-likelihood, gradient, Hessian
        (logL,grad,H)=CalcLogL(data)
    elif onestepfinal: # if final step (i.e. full data) of 1-step estimation
        # fully calculate log-likelihood, gradient, Hessian, wG
        (logL,grad,H,wG)=CalcLogL(data,alsogradweights=True)
        # store individual-specific weights for gradient
        data.wG=wG
    else: # all other cases
        # calculate logL, grad, Hessian using weights baseline model
        (logL,grad,H)=CalcLogL(data,useexistingweights=True)
    # while not converged and MAXITER not reached
    while not(converged) and i<MAXITER:
        # if logL is -infinity: quit; on a dead track for this model
        if np.isinf(logL):
            if baseline:
                data.wL=None
                data.wG=None
                data.wH=None
                data.logL=logL
                return converged
            if reestimation: return logL,converged
            if onestep and not(onestepfinal): return converged,i
            return logL,None,[0],None,converged,i
        # get (bended) EVD of -Hessian
        (Dadj,P,D)=BendEVDSymPSD(-H)
        # get Newton-Raphson update
        update=(P@((((grad.reshape((data.k*TOTALCOLS,1))).T@P)/Dadj).T))\
            .reshape((data.k,TOTALCOLS))
        # calculate convergence criterion
        msg=(update*grad).mean()
        # if convergence criterion met or last step 1-step estimation is done
        if msg<TOL or (onestepfinal and i>0):
            converged=True
        else: # otherwise
            # do line search if baseline model or linesearch is activated
            if baseline or linesearch:
                (j,step)=GoldenSection(data,logL,grad,update)
            else: data.param+=update # else just full Newton update
            # update iteration counter
            i+=1
            # provide info if not silent
            if not(silent):
                report='Newton iteration '+str(i)+': logL='+str(logL)
                if baseline or linesearch:
                    report+='; '+str(j)+' line-search steps,'+\
                            ' yielding step size = '+str(step)
                logger.info(report)
            # if baseline model or, 1-step estimation but not final step
            if baseline or (onestep and not(onestepfinal)):
                # calculate new log-likelihood, gradient, Hessian
                (logL,grad,H)=CalcLogL(data)
            else:
                # calculate new log-likelihood, gradient, Hessian, wG
                (logL,grad,H,wG)=CalcLogL(data,alsogradweights=True)
                # store individual-specific weights for gradient
                data.wG=wG
    # if baseline model
    if baseline:
        # get individual-specific weights (for iter 1 SNP-specific model)
        (wL,wG,wH)=CalcLogL(data,weightsonly=True)
        # store individual-specific weights and overall log-likelihood
        data.wL=wL
        data.wG=wG
        data.wH=wH
        data.logL=logL
        # and return whether converged
        return converged
    # if re-estimation for LRT (for subsample with nonmissing genotype)
    if reestimation:
        # return logL and whether converged
        return logL,converged
    # if 1-step estimation but not final step
    if onestep and not(onestepfinal):
        # return number of iterations and whether converged
        return converged,i
    # calculate OPG (for robust SEs)
    GGT=CalcLogL(data,Gonly=True,useexistingweights=True)
    # return logL, OPG, EVD -H, converged, iterations
    return logL,GGT,D,P,converged,i

def BFGS(data,reestimation=False):
    # set iteration counter to 0 and convergence to false
    i=0
    converged=False
    # calculate logL, grad, Hessian using weights baseline model
    (logL,grad,H)=CalcLogL(data,useexistingweights=True)
    # get (bended) EVD of -Hessian
    (Dadj,P,D)=BendEVDSymPSD(-H)
    # initialise approximated inverse Hessian
    AIH=-(P*(1/Dadj[None,:]))@P.T
    # while not converged and MAXITER not reached
    while not(converged) and i<MAXITER:
        # if logL is -infinity: quit; on a dead track for this model
        if np.isinf(logL):
            if reestimation: return logL,converged
            if onestep: return converged,i
            return logL,None,[0],None,converged,i
        # get BFGS update
        update=(-AIH@grad.reshape((data.k*TOTALCOLS,1)))\
            .reshape((data.k,TOTALCOLS))
        # calculate convergence criterion
        msg=(update*grad).mean()
        # if converged: convergenced=True
        if msg<TOL:
            converged=True
        else: # if not converged yet: get new parameters, logL, grad
            # either via line search
            if linesearch:
                (j,step,paramnew,logLnew,gradnew)=\
                    GoldenSection(data,logL,grad,update,bfgs=True)
            else: # or full update
                paramnew=data.param+update
                (logLnew,gradnew)=\
                    CalcLogL(data,param=paramnew,logLgradonly=True)
            # complete update if new log-likelihood is not -infinity
            if not(np.isinf(logLnew)):
                # calculate quantities needed for BFGS
                s=(paramnew-data.param).reshape((data.k*TOTALCOLS,1))
                y=(gradnew-grad).reshape((data.k*TOTALCOLS,1))
                sty=(s*y).sum()
                r=1/sty
                v=s*r
                w=AIH@y
                # store new parameters, grad, logL
                data.param=paramnew
                grad=gradnew
                logL=logLnew
                # update approximated inverse Hessian and stabilise
                AIH=AIH-np.outer(v,w)-np.outer(w,v)+\
                    np.outer(v,v)*((w*y).sum())+np.outer(v,s)
                AIH=(AIH+(AIH.T))/2
                # update iteration counter
                i+=1
            else:
                # otherwise only set logL to -np.inf, causing BFGS to stop
                # because of condition at start of while loop
                logL=-np.inf
    # if re-estimation for LRT (for subsample with nonmissing genotype)
    if reestimation:
        # return logL and whether converged
        return logL,converged
    # if 1-step efficient estimation (full convergence in small sample;
    # doing inference based on final Newton step in full sample)
    if onestep:
        # return number of iterations and whether converged
        return converged,i
    # calculate logL, OPG, H at BFGS solution
    (logL,GGT,H)=CalcLogL(data,Ginsteadofgrad=True)
    # get (bended) EVD of unpacked -Hessian
    (_,P,D)=BendEVDSymPSD(-H)
    # return logL, OPG, EVD -H, converged, iterations
    return logL,GGT,D,P,converged,i

def GoldenSection(data,logL,grad,update,bfgs=False):
    # calculate update'grad
    utg=(grad*update).sum()
    # initialise parameters at various points along interval
    param1=data.param
    param2=data.param+ONEMINTHETAINV*update
    param3=data.param+THETAINV*update
    param4=data.param+update
    # set corresponding step sizes
    step1=0
    step2=ONEMINTHETAINV
    step3=THETAINV
    step4=1
    # set iteration counter to one and converged to false
    j=1
    converged=False
    # calculate log likelihood at right
    logL4=CalcLogL(data,param=param4,logLonly=True)
    # directly try Armijo's rule before performing actual section search
    if logL4>=logL+ARMIJO*step4*utg:
        converged=True
    else: # if not directly meeting criterion
        # calculate log likelihoods mid-left and mid-right
        logL2=CalcLogL(data,param=param2,logLonly=True)
        logL3=CalcLogL(data,param=param3,logLonly=True)
    # while not converged and MAXITER not reached
    while not(converged) and j<MAXITER:
        # update iteration counter
        j+=1
        #if mid-left val >= mid-right val: set mid-right as right
        if logL2>=logL3: 
            # set parameters accordingly
            param4=param3
            param3=param2
            param2=THETAINV*param1+ONEMINTHETAINV*param4
            # set step sizes accordingly
            step4=step3
            step3=step2
            step2=THETAINV*step1+ONEMINTHETAINV*step4
            # calculate log likelihood at new mid-left and mid-right
            logL4=logL3
            logL3=logL2
            logL2=CalcLogL(data,param=param2,logLonly=True)
            # test if Armijo's rule satisfied
            if logL4>=logL+ARMIJO*step4*utg:
                converged=True
        #if mid-right val > mid-left val: set mid-left as left
        else:
            # set parameters accordingly
            param1=param2
            param2=param3
            param3=THETAINV*param4+ONEMINTHETAINV*param1
            # set step sizes accordingly
            step1=step2
            step2=step3
            step3=THETAINV*step4+ONEMINTHETAINV*step1
            # calculate log likelihood at new mid-left and mid-right
            logL2=logL3
            logL3=CalcLogL(data,param=param3,logLonly=True)
    # if we're doing BFGS
    if bfgs:
        # calculate gradient, and return relevant output for new estimates
        grad4=CalcLogL(data,param=param4,gradonly=True)
        return j,step4,param4,logL4,grad4
    # otherwise: update params, and just return no. of steps and step size
    data.param=param4
    return j,step4

def CalcLogL(data,param=None,useexistingweights=False,\
                weightsonly=False,logLonly=False,logLgradonly=False,\
                Gonly=False,gradonly=False,alsogradweights=False,\
                Ginsteadofgrad=False):
    # if only contribution per individual to gradient wanted
    # and we can use weights from previous calculation
    if useexistingweights and Gonly:
        # calculate and return OPG
        GGT=CalculateOPG(data.x,data.wG,data.n)
        return GGT
    # if we can use weights from previous calculation to get logL,grad,H
    if useexistingweights:
        # calculate log-likelihood/n and gradient/n using those weights
        logL=(data.wL.sum())/data.n
        grad=(data.x.T@data.wG)/data.n
        # calculate Hessian/n using those weights
        H=CalculateHessian(data.x,data.wH,data.n)
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
    if logLonly or logLgradonly or \
        (not(gradonly) and not(Gonly) and not(weightsonly)):
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
    wG[:,GAMMACOL]=(rho-deltasqm1div4delta*r1sqplusr2sq+\
                    deltasqp1div2delta*r1r2)/2
    # set gradient=0 w.r.t. beta1 for missing y1
    # and idem w.r.t. beta2 for missing y2
    wG[~data.y1notnan,BETA1COL]=0
    wG[~data.y2notnan,BETA2COL]=0
    # calculate gradient only if necessary
    if logLgradonly or gradonly or not(Gonly or Ginsteadofgrad or weightsonly):
        grad=(data.x.T@wG)/data.n
    # if only logL and grad desired, return that
    if logLgradonly:
        return logL,grad
    # if only gradient desired, return that
    if gradonly:
        return grad
    # calculate G (individual-specific contributions grad/n) only if necessary
    if Gonly or Ginsteadofgrad:
        # calculate OPG
        GGT=CalculateOPG(data.x,wG,data.n)
    # if only OPG wanted, return that
    if Gonly:
        return GGT
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
    wH[:,GAMMACOL,GAMMACOL]=deltasqm1div4delta*r1r2+\
        (unexp-deltasqp1div2delta*r1sqplusr2sq)/4
    # set weight=0 w.r.t. gamma twice when either y1 and/or y2 is missing
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
    H=CalculateHessian(data.x,wH,data.n)
    # if we also need the individual-specific weights for gradient
    if alsogradweights:
        # return logL, gradient, Hessian, and those weights
        return logL,grad,H,wG
    # if we need OPG instead of grad
    if Ginsteadofgrad:
        # return logL, OPG, Hessian
        return logL,GGT,H
    # otherwise, return logL, gradient, Hessian
    return logL,grad,H

# define class for reading genotypes
class GenoDataReader:
    
    def __init__(self):
        # print update
        logger.info('Reading genotype data')
        # connect to bed file
        connbed=open(args.bfile+extBED,'rb')
        # check if first three bytes bed file are correct
        if ord(connbed.read(1))!=(ord(binBED1)) or \
            ord(connbed.read(1))!=(ord(binBED2)) or \
                ord(connbed.read(1))!=(ord(binBED3)):
            raise ValueError(args.bfile+extBED+' not a valid PLINK .bed file')
        # close connection to bed file
        connbed.close()
        # set filenames
        famfilename=args.bfile+extFAM
        bimfilename=args.bfile+extBIM
        bedfilename=args.bfile+extBED
        # count number of observations and SNPs in PLINK data
        self.nPLINK=CountLines(famfilename)
        self.M=CountLines(bimfilename)
        # print update N and M, throwing error if either too low
        logger.info('Found '+str(self.nPLINK)+' individuals in '+famfilename)
        if self.nPLINK<MINN:
            raise ValueError('You need at least '+str(MINN)+' individuals')
        logger.info('Found '+str(self.M)+' SNPs in '+bimfilename)
        if self.M<MINM:
            raise ValueError('Need at least '+str(MINM)+' SNP')
        # initialise starting point and end point to analyses
        self.Mstart=0
        self.Mend=self.M
        # get total bytes per SNP
        self.BytesT=GetNumberOfGroups(self.nPLINK,nperbyte)
        # compute expected number of bytes in .bed file: 3 magic bytes + data
        BytesExp=3+self.BytesT*self.M
        # get observed number of bytes in .bed file
        BytesObs=(os.stat(bedfilename)).st_size
        # throw error if mismatch
        if BytesExp!=BytesObs:
            raise ValueError('Unexpected number of bytes in '+bedfilename+\
                             '. File corrupted?')
        # compute rounded n
        self.nPLINKT=self.BytesT*nperbyte
        # if SNP range has been provided
        if args.snp is not None:
            # start indexing at zero
            self.Mstart=args.snp[0]-1 
            # and use until instead of up until
            self.Mend=args.snp[1]
            if self.Mstart>self.M:
                raise ValueError('Index first SNP to analyse according to '\
                                 +'--snp exceeds number of SNPs in data')
            if self.Mend>self.M:
                raise ValueError('Index last SNP to analyse according to '\
                                 +'--snp exceeds number of SNPs in data')
            if self.Mstart>=self.Mend:
                raise ValueError('Index first SNP exceeds index last SNP '\
                                 +'to analyse according to --snp')
        # count how many SNPs and how many output blocks
        self.Mt=self.Mend-self.Mstart
        self.ResultBlocksT=GetNumberOfGroups(self.Mt,RESULTSMBLOCK)
        # count how many SNP blocks when simulating data
        self.BlocksT=GetNumberOfGroups(self.M,SIMULMBLOCK)
    
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
        # recode genotype: 0=homozygote A1, 1=heterozygote, 2=homozygote A2
        g[g==2]=1
        g[g==3]=2
        # return genotype and missingness vector
        return g, gisnan
    
    def ReadBlock(self,b):
        # find index of starting and ending SNP in this block
        m0=b*SIMULMBLOCK
        m1=min(self.M,(b+1)*SIMULMBLOCK)
        # count number of SNP in this block
        m=m1-m0
        # calculate offset
        offset=3+(self.BytesT*m0)
        # return genotype bytes on m SNPs
        g=self.ReadSNPBytes(offset,m)
        # throw error if missing genotypes
        if (g==1).sum()>0:
            raise ValueError('Missing genotypes in PLINK files, which is not'+\
                ' permitted in simulation; use e.g. `plink --bfile '+\
                args.bfile+' --geno 0 --make-bed --out '+args.bfile+\
                    '2` to obtain dataset without missing genotypes')
        # recode genotype: 0=homozygote A1, 1=heterozygote, 2=homozygote A2
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
            # get two bits at a time from each byte, going from RIGHT to LEFT
            # where 00=0, 01=1, 10=2, 11=3
            # example: byte [11100100], becomes PLINK genotypes [0, 1, 2, 3]
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
        logger.info('Reading in '+args.bfile+extBED+' in blocks of '+\
                    str(SIMULMBLOCK)+' SNPs')
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
            alpha2=((args.rg*gf1)+(((1-(args.rg**2))**0.5)*gf2))*\
                ((args.h2y2/self.M)**0.5)
            # draw SNP effects on variances and correlation
            beta1=rngeffects.normal(size=m)*((args.h2sig1/self.M)**0.5)
            beta2=rngeffects.normal(size=m)*((args.h2sig2/self.M)**0.5)
            gamma=args.rhoband*(rngeffects.normal(size=m)*\
                                ((args.h2rho/self.M)**0.5))
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
                # read line bim file, strip trailing newline, split by tabs
                snpline=connbim.readline().rstrip(eol).split(sep)
                # get chromosome no., snp ID, baseline allele, effect allele
                snpchr=snpline[0]
                snpid=snpline[1]
                snpbaseallele=snpline[4]
                snpeffallele=snpline[5]
                # write SNP info and true SNP effects to .eff file 
                conneff.write(snpchr+sep+snpid+sep+snpbaseallele+sep\
                              +snpeffallele+sep+str(alpha1[j])+sep\
                              +str(alpha2[j])+sep+str(beta1[j])+sep\
                              +str(beta2[j])+sep+str(gamma[j])+eol)
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
        # scale and mix noise to achieve desired
        # standard deviations and correlations 
        e1=eta1*sig1*((1-args.h2y1)**0.5)
        e2=((rho*eta1)+(((1-(rho**2))**0.5)*eta2))*sig2*((1-args.h2y2)**0.5)
        # draw outcomes and store in dataframe
        y1=xalpha1+e1
        y2=xalpha2+e2
        ydata=pd.DataFrame(np.hstack((y1[:,None],y2[:,None])),\
                           columns=['Y1','Y2'])
        # set fam filename
        famfilename=args.bfile+extFAM
        # store name of to be generated phenotype file
        args.pheno=args.out+extPHE
        # read fam file to dataframe
        famdata=pd.read_csv(famfilename,sep=sep,header=None,names=fieldsFAM)
        # concatenate FID,IID,Y1,Y2, and write to csv
        pd.concat([famdata.iloc[:,[0,1]],ydata],axis=1).to_csv(args.pheno,\
                                                        index=False,sep=sep)

def SimulateG():
    # get n and M
    n=args.n
    M=args.m
    # initialise random-numer generator
    rng=np.random.default_rng(args.seed_geno)
    # give update
    logger.info('Simulating data on '+str(M)+' SNPs for '+str(n)+\
                ' individuals,')
    logger.info('exporting to PLINK binary files '+args.out+extBED+','+extBIM+\
                ','+extFAM)
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
    logger.info('Simulating SNPs and writing .bim file in blocks of '+\
                str(SIMULMBLOCK)+' SNPs')
    # count total number of blocks of SNPs
    BlocksT=GetNumberOfGroups(M,SIMULMBLOCK)
    # count total bytes per SNP
    BytesT=GetNumberOfGroups(n,nperbyte)
    # compute rounded n
    nT=BytesT*nperbyte
    # set counter for total number of SNPs handled for export to .bim
    i=0
    # for each blok
    for b in tqdm(range(BlocksT)):
        # find index for first SNP and last SNP in block
        m0=b*SIMULMBLOCK
        m1=min(M,(b+1)*SIMULMBLOCK)
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
        # recode: 0=homozygote A1; 2=heterozygote; 3=homozygote A2 (1=missing)
        g=2*g
        g[g==4]=3
        # within each byte: 2 bits per individual
        base=np.array([2**0,2**2,2**4,2**6]*BytesT,dtype=np.uint8)
        # per SNP, per byte: aggregate across individuals in that byte
        exportbytes=(g*base[:,None]).reshape(BytesT,nperbyte\
                                             ,m).sum(axis=1).astype(np.uint8)
        # write bytes
        connbed.write(bytes(exportbytes.T.ravel()))
        # for each SNP in this block
        for j in range(m):
            # update counter
            i+=1
            # draw two alleles without replacement from four possible alleles
            A1A2=rng.choice(ALLELES,size=2,replace=False)
            # write line of .bim file
            connbim.write('0'+sep+'rs'+str(i)+sep+'0'+sep+str(j)+sep+\
                          A1A2[0]+sep+A1A2[1]+eol)
            # write line of .frq file
            connfrq.write('0'+sep+'rs'+str(i)+sep+A1A2[0]+sep+A1A2[1]+sep+\
                          str(1-eaf[j])+sep+str(eaf[j])+eol)
    # close connections
    connbed.close()
    connbim.close()
    connfrq.close()
    # store prefix of just generated PLINK binary files
    args.bfile=args.out

def GetNumberOfGroups(elements,size):
    '''
    Given number of discrete elements divided into groups of equal size,
    calculates how many groups there are in total
    (i.e. including a potential remainder group)

    Args:
        elements (int): number of discrete elements
        size (int): number of elements per group

    Returns:
        groups (int): number of groups, including potential remainder group
    '''
    fullgroups=int(elements/size)
    remainder=elements%size
    groups=fullgroups+(remainder>0)
    return groups

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
        raise argparse.ArgumentTypeError(string+\
                                    " is not a number in the interval (0,1)")
    if val>0 and val<1:
        return val
    else:
        raise argparse.ArgumentTypeError("%s is not a number in the"+\
                                         " interval (0,1)" % val)

def number_between_m1_p1(string):
    '''
    Try to parse input argument as number in (-1,1)
    '''
    try:
        val=float(string)
    except:
        raise argparse.ArgumentTypeError(string+\
                                    " is not a number in the interval (-1,1)")
    if (val**2)<1:
        return val
    else:
        raise argparse.ArgumentTypeError("%s is not a number in the"+\
                                         " interval (-1,1)" % val)

def CalculateOPG(X,W,n):
    '''
    Calculate OPG of log-likelihood function, given matrix of regressors
    and individual-specific weights for the various submatrices of the Hessian

    Args:
        X (ndarray): N-by-K matrix of regressors
        W (ndarray): N-by-TOTALCOLS array of weights
        n (int): sample size of analysis, to divide G for numerical stability
    
    Returns:
        GGT (ndarray): (K*TOTALCOLS)-by-(K*TOTALCOLS) OPG matrix
    '''
    K=X.shape[1]
    GGT=np.empty((K,TOTALCOLS,K,TOTALCOLS))
    for i in range(TOTALCOLS):
        for j in range(i,TOTALCOLS):
            # calculate OPG for component i vs. j and j vs. i
            GGT[:,i,:,j]=(X*(W[:,None,i]/n)).T@(X*(W[:,None,j]/n))
            if j>i:
                # use symmetry to find out counterparts
                GGT[:,j,:,i]=GGT[:,i,:,j]
    # return unpacked OPG
    return GGT.reshape((K*TOTALCOLS,K*TOTALCOLS))

def CalculateHessian(X,W,n):
    '''
    Calculate Hessian of log-likelihood function, given matrix of regressors
    and individual-specific weights for the various submatrices of the Hessian

    Args:
        X (ndarray): N-by-K matrix of regressors
        W (ndarray): N-by-(TOTALCOLS-by-TOTALCOLS) array of weights
        n (int): sample size of analysis, to divide H for numerical stability
    
    Returns:
        H (ndarray): (K*TOTALCOLS)-by-(K*TOTALCOLS) Hessian matrix
    '''
    K=X.shape[1]
    H=np.empty((K,TOTALCOLS,K,TOTALCOLS))
    for i in range(TOTALCOLS):
        for j in range(i,TOTALCOLS):
            # calculate Hessian for component i vs. j and j vs. i
            H[:,i,:,j]=(X.T@(X*W[:,None,i,j]))/n
            if j>i:
                # use symmetry to find out counterparts
                H[:,j,:,i]=H[:,i,:,j]
    # return unpacked Hessian
    return H.reshape((K*TOTALCOLS,K*TOTALCOLS))

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
            # Initialise analyser and estimate baseline model
            # using Newton's method with line search
            analyser=Analyser(snpreader)
            # Estimate SNPs-specific model
            analyser.EstimateSNPModels()
    except Exception:
        # print the traceback
        logger.error(traceback.format_exc())
        # wrap up with final error message
        logger.error('Error: GCAT did not exit properly.'+\
                     ' Please inspect the log file.')
        logger.info('Run `python ./gcat.py -h` to show all options')
    finally:
        # print total time elapsed
        logger.info('Total time elapsed: '+ConvertSecToStr(time.time()-t0))
        logger.info('Current memory usage is '+\
                    str(int((process.memory_info().rss)/(1024**2)))+'MB')

def ParseInputArguments():
    # get global parser and initialise args as global
    global args
    # define input arguments
    parser.add_argument('--n', metavar = 'INTEGER', default = None,
                    type = positive_int, 
                    help = help_n)
    parser.add_argument('--m', metavar = 'INTEGER', default = None,
                    type = positive_int, 
                    help = help_m)
    parser.add_argument('--seed-geno', metavar = 'INTEGER', default = None,
                    type = positive_int,
                    help = help_seed_geno)
    parser.add_argument('--bfile', metavar = 'PREFIX', default = None,
                    type = str,
                    help = help_bfile)
    parser.add_argument('--h2y1', metavar = 'NUMBER', default = None,
                    type = number_between_0_1,
                    help = help_h2y1)
    parser.add_argument('--h2y2', metavar = 'NUMBER', default = None,
                    type = number_between_0_1,
                    help = help_h2y2)
    parser.add_argument('--rg', metavar = 'NUMBER', default = None,
                    type = number_between_m1_p1,
                    help = help_rg)
    parser.add_argument('--h2sig1', metavar = 'NUMBER', default = None,
                    type = number_between_0_1,
                    help = help_h2sig1)
    parser.add_argument('--h2sig2', metavar = 'NUMBER', default = None,
                    type = number_between_0_1,
                    help = help_h2sig2)
    parser.add_argument('--h2rho', metavar = 'NUMBER', default = None,
                    type = number_between_0_1,
                    help = help_h2rho)
    parser.add_argument('--rhomean', metavar = 'NUMBER', default = None,
                    type = number_between_m1_p1,
                    help = help_rhomean)
    parser.add_argument('--rhoband', metavar = 'NUMBER', default = None,
                    type = number_between_0_1,
                    help = help_rhoband)
    parser.add_argument('--seed-effects', metavar = 'INTEGER', default = None,
                    type = positive_int,
                    help = help_seed_effects)
    parser.add_argument('--seed-pheno', metavar = 'INTEGER', default = None,
                    type = positive_int,
                    help = help_seed_pheno)
    parser.add_argument('--pheno', metavar = 'FILENAME', default = None,
                    type = str,
                    help = help_pheno)
    parser.add_argument('--covar', metavar = 'FILENAME', default = None,
                    type = str,
                    help = help_covar)
    parser.add_argument('--simul-only', action = 'store_true',
                    help = help_simul_only)
    parser.add_argument('--bfgs', action = 'store_true',
                    help = help_bfgs)
    parser.add_argument('--section', action = 'store_true',
                    help = help_section)
    parser.add_argument('--one', metavar = '', default = None, nargs= '+',
                    help = help_one)
    parser.add_argument('--lrt', action = 'store_true',
                    help = help_lrt)
    parser.add_argument('--snp', metavar = '', default = None,
                    type = positive_int, nargs= '+',
                    help = help_snp)
    parser.add_argument('--out', metavar = 'PREFIX', default = None,
                    type = str,
                    help = help_out)
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
        # check if output holds directory name and, if so, if it exists
        if not(sDir == '') and not(os.path.isdir(sDir)):
            # if so, raise an error
            raise ValueError('prefix specified using --out may start with a '+\
                             'directory name, provided it exists; ' +\
                                sDir + ' does not exist')
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
        options = ['--'+x.replace('_','-')+' '+str(opts[x])+\
                   ' \\' for x in non_defaults]
        header += eol.join(options).replace('True','').replace('False','').\
            replace("', \'", ' ').replace("']", '').replace("['", '').\
                replace('[', '').replace(']', '').replace(', ', ' ').\
                    replace('  ', ' ')
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
        raise SyntaxError('you must specify either --bfile or'+\
                          ' both --n and --m')
    if args.bfile is None:
        simulg=True
        simuly=True
        if args.m<MINM:
            raise ValueError('you must simulate at least '+str(MINM)+\
                             ' SNPs when using --m')
        if args.n<MINN:
            raise ValueError('you must simulate at least '+str(MINN)+\
                             ' individuals when using --n')
        if args.pheno is not None:
            raise SyntaxError('--pheno cannot be combined with --n and --m')
        if args.covar is not None:
            raise SyntaxError('--covar cannot be combined with --n and --m')
    else:
        if not(os.path.isfile(args.bfile+extBED)):
            raise OSError('PLINK binary data file '+args.bfile+extBED+\
                          ' cannot be found')
        if not(os.path.isfile(args.bfile+extBIM)):
            raise OSError('PLINK binary data file '+args.bfile+extBIM+\
                          ' cannot be found')
        if not(os.path.isfile(args.bfile+extFAM)):
            raise OSError('PLINK binary data file '+args.bfile+extFAM+\
                          ' cannot be found')
        if args.pheno is None:
            simuly=True
            if args.covar is not None:
                raise SyntaxError('--covar must be combined with --pheno')
        else:
            if not(os.path.isfile(args.pheno)):
                raise OSError('phenotype file '+args.pheno+' cannot be found')
            if args.covar is not None:
                covars=True
                if not(os.path.isfile(args.covar)):
                    raise OSError('covariate file '+args.covar+\
                                  ' cannot be found')
            if args.simul_only:
                raise SyntaxError('do not combine --simul-only and --pheno')
    if simuly:
        if args.h2y1 is None:
            raise SyntaxError('--h2y1 needed to simulate phenotypes')
        if args.h2y2 is None:
            raise SyntaxError('--h2y2 needed to simulate phenotypes')
        if args.rg is None:
            raise SyntaxError('--rg needed to simulate phenotypes')
        if args.h2sig1 is None:
            raise SyntaxError('--h2sig1 needed to simulate phenotypes')
        if args.h2sig2 is None:
            raise SyntaxError('--h2sig2 needed to simulate phenotypes')
        if args.h2rho is None:
            raise SyntaxError('--h2rho needed to simulate phenotypes')
        if args.rhomean is None:
            raise SyntaxError('--rhomean needed to simulate phenotypes')
        if args.rhoband is None:
            raise SyntaxError('--rhoband needed to simulate phenotypes')
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
            raise SyntaxError('--seed-geno cannot be combined with --bfile')
    if not(simuly):
        if args.seed_pheno is not None:
            raise SyntaxError('--seed-pheno cannot be combined with --pheno')
        if args.seed_effects is not None:
            raise SyntaxError('--seed-effects cannot be combined with --pheno')
    if simulg:
        if args.seed_geno is None:
            raise SyntaxError('--seed-geno needed to simulate genotypes')
    if simuly:
        if args.seed_pheno is None:
            raise SyntaxError('--seed-pheno needed to simulate phenotypes')
        if args.seed_effects is None:
            raise SyntaxError('--seed-effects needed to simulate phenotypes')
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
            raise SyntaxError('--one needs to be followed by a number'+\
                              ' between 0 and 1e (fraction of individuals'+\
                              ' that is randomly sampled) and a positive'+\
                              ' integer (to set random-number generator'+\
                              ' for random sampling)')
        args.one[0]=number_between_0_1(args.one[0])
        args.one[1]=positive_int(args.one[1])
        onestep=True
    if args.lrt and args.simul_only:
        raise SyntaxError('--simul-only cannot be combined with --lrt')

# invoke main function when called on as a script
if __name__ == '__main__':
    main()
