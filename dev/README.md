# Running code on slurm
Please see the script `dev/launch_slurm.sh` for running the code on slurm cluster for training purposes.

# Practices for coding quality

For every PR, we run/mandate a few checks before code is ready for review. The checks are as defined by:
1. flake8
2. black
3. isort

See https://pre-commit.com/ for more information.

## isort

We check that imports are properly sorted https://pypi.org/project/isort/

## black

We format the code using black8

## flake8

We enforce coding style per file with defined flake8 rules. flake8 is a checker only and doesn't format code.

### Option1: use `linter.sh`

Run "./dev/linter.sh" at the project root before you commit. This will run isort, black and flake8 formatting.

### Option2: use `.pre-commit-config`

We provide pre-commit hooks so as you build and commit (locally or github), the code formatting will be automatically run.
You need to run `pre-commit install` once to enable this.

Read the doc https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/ for how all the components operate and the pipeline/steps involved.
