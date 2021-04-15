## Run everything on 1 gpu `low_resource_1gpu_train_wrapper.sh`

If you have a configuration file (any vissl compatible file) that you want to run on 1-gpu only (for example: train SimCLR on 1 gpu, etc), you don't need to modify the config file. VISSL also takes care of auto-scaling the Learning rate for various schedules (cosine, multistep, step etc.) if you have enabled the auto_scaling (see `config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling`). You can simply achieve this by using the `low_resource_1gpu_train_wrapper.sh` script. An example usage:

```bash
cd $HOME/vissl
./dev/low_resource_1gpu_train_wrapper.sh config=test/integration_test/quick_swav
```

## Running quick tests `run_quick_tests.sh`

To run trainings (SimCLR, SwAV, PIRL etc.) for a few iterations, we provide `run_quick_tests.sh`. This requires 1 or 2 gpus.

## Running code on slurm with `dev/launch_slurm.sh`
Please see the script `dev/launch_slurm.sh` for running the code on slurm cluster for training purposes.

## Practices for coding quality

For every PR, we run/mandate a few checks before code is ready for review. The checks are:
1. **flake8**: We enforce coding style per file with defined flake8 rules. flake8 is a checker only and doesn't format code.
2. **black**: We format the code using black8
3. **isort**: We check that imports are properly sorted https://pypi.org/project/isort/. See `setup.cfg` for the settings.

In order to format code before code review, there are several options:

### Option 1: use `dev/linter.sh`

Install flake8, isort, black by running `cd $HOME/vissl && pip install -e ".[dev]"`.
Run "./dev/linter.sh" at the project root before you commit. This will run isort, black and flake8 formatting.

### Option 2: use `.pre-commit-config`

We provide pre-commit hooks so as you build and commit (locally or github), the code formatting will be automatically run.

Install by running `pre-commit install` once to enable this.

Read the doc https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/ for how all the components operate and the pipeline/steps involved.

### Option 3: use `python dev/lint_commit.py -a`

This script is an improved version of `dev/linter.sh` which requires an additional dependency (`pip install gitpython`) but allows you to run the linting in a more selective manner:
- on the full repository or just the files you modified
- with or without auto-correction of the files
- with selective checks only (running black only)

Here are a few examples of commands showing the different options:

- `python dev/lint_commit.py --all` will run all the checks on the modified files
- `python dev/lint_commit.py --all --check` will do the same but without auto-correction
- `python dev/lint_commit.py --all --repo` will run all the checks on the full repository
- `python dev/lint_commit.py --black` will run black on the modified files
- `python dev/lint_commit.py --sort` will run isort on the modified files
- `python dev/lint_commit.py --flake` will run flake8 on the modified files
