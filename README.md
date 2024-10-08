# GCAT (Genome-wide Cross-trait concordance Analysis Tool) `beta v0.2`

`gcat` is a Python 3.x package for estimating the effects of SNPs on trait levels, variance, and covariance.

## Installation

:warning: Before downloading `gcat`, please make sure [Git](https://git-scm.com/downloads) and [Anaconda](https://www.anaconda.com/) with **Python 3.x** are installed.

In order to download `gcat`, open a command-line interface by starting [Anaconda Prompt](https://docs.anaconda.com/anaconda/user-guide/getting-started/), navigate to your working directory, and clone the `gcat` repository using the following command:

```  
git clone https://github.com/devlaming/gcat.git
```

Now, enter the newly created `gcat` directory using:

```
cd gcat
```

Then run the following commands to create a custom Python environment which has all of `gcat`'s dependencies (i.e. all packages it needs):

```
conda env create --file gcat.yml
conda activate gcat
```

(or `activate gcat` instead of `conda activate gcat` on some machines).

In case you cannot create a customised conda environment (e.g. because of insufficient user rights) or simply prefer to use Anaconda Navigator or `pip` to install packages e.g. in your base environment rather than a custom environment, please notice that `gcat` only requires Python 3.x with the packages `numpy`, `pandas`, `psutil`, `scipy`, and `tqdm` installed.

## Tutorial

To simulate both genotype and phenotype data, for 100 SNPs and 5000 individuals, and applying `gcat` to that data, try the following line of code:

```
python ./gcat.py --n 5000 --m 100 --seed-geno 123 --h2y1 0.3 --h2y2 0.4 --rg 0.5 --h2sig1 0.25 --h2sig2 0.5 --h2rho 0.75 --rhomean 0 --rhoband 0.5 --seed-effects 456 --seed-pheno 789 --out simulation
```

Feel free to play around with different levels of heritability, correlation, and the seed of the random-number generator.

## Options

*tba: describe main options*

## Basic parallelisation using PLINK

*tba: describe workflow: when simulating, use --simul-only, then split into chunks using PLINK; otherwise directly split into chunks; run GCAT per chunk on separate (virtual) machine; aggregate using UNIX's cat command, perhaps with tail -n -1 to get rid of headers*

## Updating `gcat`

You can update to the latest version of `gcat` using `git`. First, navigate to your `gcat` directory (e.g. `cd gcat`), then run
```
git pull
```
If `gcat` is up to date, you will see 
```
Already up to date.
```
otherwise, you will see `git` output similar to 
```
remote: Enumerating objects: 8, done.
remote: Counting objects: 100% (8/8), done.
remote: Compressing objects: 100% (4/4), done.
remote: Total 6 (delta 2), reused 6 (delta 2), pack-reused 0
Unpacking objects: 100% (6/6), 2.82 KiB | 240.00 KiB/s, done.
From https://github.com/devlaming/gcat
   481a4bf..fddd8cc  main       -> origin/main
Updating 481a4bf..fddd8cc
Fast-forward
 README.md | 128 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 gcat.py |  26 ++++++++++++-
 2 files changed, 153 insertions(+), 1 deletion(-)
 create mode 100644 README.md
```
which tells you which files were changed.

If you have modified the `gcat` source code yourself, `git gcat` may fail with an error such as `error: Your local changes [...] would be overwritten by merge`. 

## Support

Before contacting me, please try the following:

1. Go over the tutorial in this `README.md` file
2. Go over the method, described in *tba* (citation below)

### Contact

In case you have a question that is not resolved by going over the preceding two steps, or in case you have encountered a bug, please send an e-mail to r\[dot\]devlaming\[at\]vu\[dot\]nl.

## Citation

If you use the software, please cite the following papers:

*tba*

## License

This project is licensed under GNU GPL v3.

## Authors

Ronald de Vlaming (Vrije Universiteit Amsterdam)
