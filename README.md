# info_theory
Kraskov-style (KNN) estimators for mutual information and transfer entropy

the GitHub homepage (this page) is [kate_git](https://github.com/kmdaftari/info_theory)

# how to install
run this command in terminal to install the `info_theory` package

```
# clone the repo info_theory and puts the copy in the active directory
git clone https://github.com/kmdaftari/info_theory 

# navigate to the directory info_theory in the cloned repo
cd info_theory

# install everything (.) in the info_theory directory in developer mode (-e)
pip install -e .
```

to update the `info-theory` package,:

```
# navigate to the info_theory directory
cd info_theory

# pull the changes
git pull

# update the package 
pip install -e . --upgrade
```

# running (automatic) tests of code
first, install test packages
```
# installs all packages listed in requirements_test.txt. these packages are used to test code.
pip install -r requirements_test.txt
```
to run tests
```
# runs pytest package, which looks at all files named test_*.py with functions named test_*
# and executes them
pytest
```

# background

## KNN mutual information estimator method
All estimators are computed using Algorithm 1 from [Kraskov](https://arxiv.org/abs/cond-mat/0305641), et al. In this method, local subspace densities are estimated by counting the number of nearest neighbors around each point. From this, the entropy is estimated and then the mutual information. In this package, we can compute the mutual information between the following pairs of random variables:

* $Z = (X,Y)$ where $X$ and $Y$ are one-dimensional scalar random variables
* $Z = (\theta_1,\theta_2)$ where $\theta_1$ and $\theta_2$ are one-dimensional angular random variables
* $Z = (X,\theta)$ where $X$ is a one-dimensional scalar random variable and $\theta$ is a one-dimensional angular random variable
* $Z = ((X_1,Y_1),(X_2,Y_2))$ where $(X_1, Y_1)$ and $(X_2,Y_2)$ are two-dimensional scalar random variables

## time-lagged mutual information
Time-lagged mutual information is an intermediate statistic between pure mutual information (time-independent) and transfer entropy (time-dependent).
To compute time-lagged mutual information between two random time-dependent variables (or random process) or a time-dependent random variable and itself, we order both random variable or a random variable and a copy of itself. We sample the first variable and record the timestamps, then sample the second variable at the same timesteps plus the timelag. Compute mutual information using these two random variables and vary the timelag to determine significant timescales. 
* $Z(X,X + \tau)$ or $Z(X,Y + \tau)$ for a timelag of length $\tau$



# helpful links
Here is a link to markdown [syntax](https://www.markdownguide.org/basic-syntax/)
* a bullet point
* another bullet
    * yet another

