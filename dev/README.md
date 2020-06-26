# Running code on slurm
Please see the script `dev/launch_slurm.sh` for running the code on slurm cluster for training purposes.

# Practices for coding quality

Run "./dev/linter.sh" at the project root before you commit. An explanation of what happens under the hood:

Read the doc https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/ for how all the components operate and the pipeline/steps involved.

## Pre-commit-config

For every PR, we run/mandate a few checks before code is ready for review. The checks are as defined by:
1. pre-commit-hooks
2. flake8
3. black
4. isort

See https://pre-commit.com/ for more information.


## isort

We check that imports are properly sorted https://pypi.org/project/isort/

## black

We format the code using black8

## flake8

We enforce coding style per file with defined flake8 rules. flake8 is a checker only and doesn't format code.
