# Summary: Feature Eval Config Settings

This doc describes summary of how to set :code:`MODEL.FEATURE_EVAL_SETTINGS` parameter for different evaluations.

In order to evaluate the model, you need to set :code:`MODEL.FEATURE_EVAL_SETTINGS` in yaml config file. Various options determine how the model is evaluated and also what part of the model is initialized from weights or what part of the model is frozen.

Below we provide instructions for setting the :code:`MODEL.FEATURE_EVAL_SETTINGS` for evaluating a pre-trained model on several benchmark tasks. Below are only some example scenarios but hopefully provide an idea for any different use case one might have in mind.


## Linear Image Classification with MLP heads

### Attach MLP heads to several layers of the trunk

  - If you want Zhang et. al style which has `BN -> FC` as the head, use `eval_mlp` head. Example:

```yaml
MODEL:
  FEATURE_EVAL_SETTINGS:
    EVAL_MODE_ON: True
    FREEZE_TRUNK_ONLY: True
    SHOULD_FLATTEN_FEATS: False
    LINEAR_EVAL_FEAT_POOL_OPS_MAP: [
      ["conv1", ["AvgPool2d", [[10, 10], 10, 4]]],
      ["res2", ["AvgPool2d", [[16, 16], 8, 0]]],
      ["res3", ["AvgPool2d", [[13, 13], 5, 0]]],
      ["res4", ["AvgPool2d", [[8, 8], 3, 0]]],
      ["res5", ["AvgPool2d", [[6, 6], 1, 0]]],
      ["res5avg", ["Identity", []]],
    ]
  TRUNK:
    NAME: resnet
    RESNETS:
      DEPTH: 50
  HEAD:
    PARAMS: [
      ["eval_mlp", {"in_channels": 64, "dims": [9216, 1000]}],
      ["eval_mlp", {"in_channels": 256, "dims": [9216, 1000]}],
      ["eval_mlp", {"in_channels": 512, "dims": [8192, 1000]}],
      ["eval_mlp", {"in_channels": 1024, "dims": [9216, 1000]}],
      ["eval_mlp", {"in_channels": 2048, "dims": [8192, 1000]}],
      ["eval_mlp", {"in_channels": 2048, "dims": [2048, 1000]}],
    ]
  WEIGHTS_INIT:
    PARAMS_FILE: ""
    STATE_DICT_KEY_NAME: classy_state_dict
```

- If you want `FC` layer only in the head, use `mlp` head. Example:

```yaml
MODEL:
  FEATURE_EVAL_SETTINGS:
    EVAL_MODE_ON: True
    FREEZE_TRUNK_ONLY: True
    SHOULD_FLATTEN_FEATS: False
    LINEAR_EVAL_FEAT_POOL_OPS_MAP: [
        ["res5avg", ["Identity", []]],
    ]
  TRUNK:
    NAME: resnet
    RESNETS:
      DEPTH: 50
  HEAD:
    PARAMS: [
        ["mlp", {"dims": [2048, 1000]}],
    ]
  WEIGHTS_INIT:
    PARAMS_FILE: ""
    STATE_DICT_KEY_NAME: classy_state_dict
```

### Attach MLP head to the trunk output

```yaml
MODEL:
  FEATURE_EVAL_SETTINGS:
    EVAL_MODE_ON: True
    FREEZE_TRUNK_ONLY: True
    SHOULD_FLATTEN_FEATS: False
  TRUNK:
    NAME: resnet
    RESNETS:
      DEPTH: 50
  HEAD:
    PARAMS: [
      ["eval_mlp", {"in_channels": 2048, "dims": [2048, 1000]}],
    ]
  WEIGHTS_INIT:
    PARAMS_FILE: ""
    STATE_DICT_KEY_NAME: classy_state_dict
```

## Linear Image Classification with SVM trainings

### Train SVM on several layers of the trunk

```yaml
MODEL:
  FEATURE_EVAL_SETTINGS:
    EVAL_MODE_ON: True
    FREEZE_TRUNK_ONLY: True
    EXTRACT_TRUNK_FEATURES_ONLY: True   # only extract the features and we will train SVM on these
    LINEAR_EVAL_FEAT_POOL_OPS_MAP: [
      ["res4", ["AvgPool2d", [[8, 8], 3, 0]]],
      ["res5", ["AvgPool2d", [[6, 6], 1, 0]]],
      ["res5avg", ["Identity", []]],
    ]
  TRUNK:
    NAME: resnet
    RESNETS:
      DEPTH: 50
```

### Train SVM on the trunk output

```yaml
MODEL:
  FEATURE_EVAL_SETTINGS:
    EVAL_MODE_ON: True
    FREEZE_TRUNK_ONLY: True
    EXTRACT_TRUNK_FEATURES_ONLY: True
  TRUNK:
    NAME: resnet
    RESNETS:
      DEPTH: 50
```

## Nearest Neighbor

### knn test on trunk output

```yaml
MODEL:
  FEATURE_EVAL_SETTINGS:
    EVAL_MODE_ON: True
    FREEZE_TRUNK_ONLY: True  # only freeze the trunk
    EXTRACT_TRUNK_FEATURES_ONLY: True   # we extract features from the trunk only
    SHOULD_FLATTEN_FEATS: False   # don't flatten the features and return as is
  TRUNK:
    NAME: resnet
    RESNETS:
      DEPTH: 50
  WEIGHTS_INIT:
    PARAMS_FILE: ""
    STATE_DICT_KEY_NAME: classy_state_dict
```

### knn test on model head output (self-supervised head)

```yaml
MODEL:
  FEATURE_EVAL_SETTINGS:
    EVAL_MODE_ON: True
    FREEZE_TRUNK_AND_HEAD: True   # both head and trunk will be frozen (including BN in eval mode)
    EVAL_TRUNK_AND_HEAD: True  # initialized the model head as well from weights
  TRUNK:
    NAME: resnet
    RESNETS:
      DEPTH: 50
  HEAD:
    # SimCLR model head structure
    PARAMS: [
      ["mlp", {"dims": [2048, 2048], "use_relu": True}],
      ["mlp", {"dims": [2048, 128]}],
    ]
  WEIGHTS_INIT:
    PARAMS_FILE: ""
    STATE_DICT_KEY_NAME: classy_state_dict
```

### knn test on several layers of the trunk

```yaml
MODEL:
  FEATURE_EVAL_SETTINGS:
    EVAL_MODE_ON: True
    FREEZE_TRUNK_ONLY: True  # only freeze the trunk
    EXTRACT_TRUNK_FEATURES_ONLY: True   # we extract features from the trunk only
    SHOULD_FLATTEN_FEATS: False   # don't flatten the features and return as is
    LINEAR_EVAL_FEAT_POOL_OPS_MAP: [
      ["res4", ["AvgPool2d", [[8, 8], 3, 0]]],
      ["res5", ["AvgPool2d", [[6, 6], 1, 0]]],
    ]
  TRUNK:
    NAME: resnet
    RESNETS:
      DEPTH: 50
  WEIGHTS_INIT:
    PARAMS_FILE: ""
    STATE_DICT_KEY_NAME: classy_state_dict
```

## Feature Extraction

You need to set `engine_name: extract_features` in the config file or pass the `engine_name=extract_features` as an additional input from the command line.

### Extract features from several layers of the trunk

```yaml
MODEL:
  FEATURE_EVAL_SETTINGS:
    EVAL_MODE_ON: True
    FREEZE_TRUNK_ONLY: True
    EXTRACT_TRUNK_FEATURES_ONLY: True
    SHOULD_FLATTEN_FEATS: False
    LINEAR_EVAL_FEAT_POOL_OPS_MAP: [
      ["conv1", ["AvgPool2d", [[10, 10], 10, 4]]],
      ["res2", ["AvgPool2d", [[16, 16], 8, 0]]],
      ["res3", ["AvgPool2d", [[13, 13], 5, 0]]],
      ["res4", ["AvgPool2d", [[8, 8], 3, 0]]],
      ["res5", ["AvgPool2d", [[6, 6], 1, 0]]],
      ["res5avg", ["Identity", []]],
    ]
  TRUNK:
    NAME: resnet
    RESNETS:
      DEPTH: 50
EXTRACT_FEATURES:
  OUTPUT_DIR: ""
  CHUNK_THRESHOLD: 0
```

### Extract features of the trunk output

```yaml
MODEL:
  FEATURE_EVAL_SETTINGS:
    EVAL_MODE_ON: True
    FREEZE_TRUNK_ONLY: True
    EXTRACT_TRUNK_FEATURES_ONLY: True
    SHOULD_FLATTEN_FEATS: False
  TRUNK:
    NAME: resnet
    RESNETS:
      DEPTH: 50
EXTRACT_FEATURES:
  OUTPUT_DIR: ""
  CHUNK_THRESHOLD: 0
```

### Extract features of the model head output (self-supervised head)

```yaml
MODEL:
  FEATURE_EVAL_SETTINGS:
    EVAL_MODE_ON: True
    FREEZE_TRUNK_AND_HEAD: True
    EVAL_TRUNK_AND_HEAD: True
  TRUNK:
    NAME: resnet
    RESNETS:
      DEPTH: 50
  HEAD:
    PARAMS: [
      ["mlp", {"dims": [2048, 2048], "use_relu": True}],
      ["mlp", {"dims": [2048, 128]}],
    ]
EXTRACT_FEATURES:
  OUTPUT_DIR: ""
  CHUNK_THRESHOLD: 0
```

## Full finetuning

Since this only requires to initialize the model from the pre-trained model weights, there's no need for the `FEATURE_EVAL_SETTINGS` params. Simply set the `MODEL.WEIGHTS_INIT` params.
