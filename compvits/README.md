# Usage
## Code structure
All code that integrates with `vissl` framework directly is placed into directories inside the framework code itself, as advised by the [tutorial](`https://vissl.readthedocs.io/en/v0.1.5/extend_modules/data_transforms.html`). The code that runs scripts, gathers and processes data is placed into separate `compvits` directory.

As for now, extra code is placed in the following files/directories:
 - `compvits`
 - `configs/config/compvits`
 - `vissl/data//ssl_transforms/compvits.py`

Direct `vissl` code modifications were applied to the following files:
 - `vissl/models/base_ssl_model.py`
 - `vissl/models/trunks/feature_extractor.py`
 - `vissl/models/trunks/vision_transformer.py`
 - `vissl/trainer/trainer_main.py`
 - `vissl/utils/hydra_config.py`
 - `vissl/utils/knn_utils.py`


## Running scripts
All scripts have to be run from root directory.
```bash
./compvits/scripts/run_sweep.sh compvits/scripts/extract_features.sh 
```

## Gather data and plot results
All scripts write results to `/vissl/log` directory by default. To gather the data from experiments into `.csv` files, run `extract_features.py` script.
The script saves the data to `plots/data` directory.

## Creating new configs
New configs should be placed into `configs/config/compvits` directory.

## Creating new transforms
New transforms should be placed into `vissl/data/ssl_transforms/compvits.py` file.