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

# additional information
For more background information about KNN estimator methods and some code examples, navigate to [Tutorial](notebooks/Tutorial.ipynb)





