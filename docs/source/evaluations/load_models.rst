How to Load Pretrained Models
================================

VISSL supports loading Torchvision models trunks out of the box. Generally, for loading any non-VISSL model, one needs to correctly set the following configuration options:

.. code-block:: yaml

    WEIGHTS_INIT:
      # path to the .torch weights files
      PARAMS_FILE: ""
      # name of the state dict. checkpoint = {"classy_state_dict": {layername:value}}. Options:
      #   1. classy_state_dict - if model is trained and checkpointed with VISSL.
      #      checkpoint = {"classy_state_dict": {layername:value}}
      #   2. "" - if the model_file is not a nested dictionary for model weights i.e.
      #      checkpoint = {layername:value}
      #   3. key name that your model checkpoint uses for state_dict key name.
      #      checkpoint = {"your_key_name": {layername:value}}
      STATE_DICT_KEY_NAME: "classy_state_dict"
      # specify what layer should not be loaded. Layer names with this key are not copied
      # By default, set to BatchNorm stats "num_batches_tracked" to be skipped.
      SKIP_LAYERS: ["num_batches_tracked"]
      ####### If loading a non-VISSL trained model, set the following two args carefully #########
      # to make the checkpoint compatible with VISSL, if you need to remove some names
      # from the checkpoint keys, specify the name
      REMOVE_PREFIX: ""
      # In order to load the model (if not trained with VISSL) with VISSL, there are 2 scenarios:
      #    1. If you are interested in evaluating the model features and freeze the trunk.
      #       Set APPEND_PREFIX="trunk.base_model." This assumes that your model is compatible
      #       with the VISSL trunks. The VISSL trunks start with "_feature_blocks." prefix. If
      #       your model doesn't have these prefix you can append them. For example:
      #       For TorchVision ResNet trunk, set APPEND_PREFIX="trunk.base_model._feature_blocks."
      #    2. where you want to load the model simply and finetune the full model.
      #       Set APPEND_PREFIX="trunk."
      #       This assumes that your model is compatible with the VISSL trunks. The VISSL
      #       trunks start with "_feature_blocks." prefix. If your model doesn't have these
      #       prefix you can append them.
      #       For TorchVision ResNet trunk, set APPEND_PREFIX="trunk._feature_blocks."
      # NOTE: the prefix is appended to all the layers in the model
      APPEND_PREFIX: ""
