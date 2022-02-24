## Indicator2: Geographical Diversity

- **Step1: Setup Data**

  This indicator performs _inference_ on the [DollarStreet](https://www.gapminder.org/dollar-street) dataset which contains images from different households all over the world. The goal of this indicator it to test if models can recognize objects correctly all over the world. For this reason, model should have the capability to predict labels. Optionally, if your model doesn't predict labels, then the model needs to be finetuned to predict labels. The finetuning can be done on a subset of ImageNet-22K classes. 

   -  __[Optional][Finetuning] Setting up ImageNet-22k subset__: ImageNet-22K dataset has 21,842 classes which map to the labels in wordnet. The full ImageNet-22K taxonomy is available [here](https://github.com/atong01/Imagenet-Tensorflow/blob/master/model/imagenet_synset_to_human_label_map.txt). Subsample the dataset corresponding to the [subset 108 classes in taxonomy](https://dl.fbaipublicfiles.com/vissl/fairness/geographical_diversity/subset_in22k_class_names.json). Note that the classes with prefix `1k_` indicate these classes are part of ImageNet-1K dataset as well.

   - __[Inference] Setup Dollar Street dataset__: We provide the [full DollarStreet metadata](https://dl.fbaipublicfiles.com/vissl/fairness/geographical_diversity/metadata_full_dollar_street.json) information including the URLs for the images and can be used to setup the full dataset including the data loader for the images. We also provide the [mapping of ImageNet-22k classees to DollarStreet labels](https://dl.fbaipublicfiles.com/vissl/fairness/geographical_diversity/in22k_cls_idx_to_dollar_street_labels_map.json).


- **Step2 [Optional]: Finetune the model** 

   This step is only needed if you need to finetune your model such that it can generate label predictions (if it doesn't already). 
   
   - You can use VISSL for this purpose. [Install VISSL](https://github.com/facebookresearch/vissl/blob/main/INSTALL.md).

   - Make your model compatible with VISSL [following these guidelines](https://vissl.readthedocs.io/en/latest/evaluations/load_models.html).

   - Make the input data format compatible by [following these guidelines](https://vissl.readthedocs.io/en/latest/vissl_modules/data.html#using-data).

   - Finetune your model. Example command (for [ResNet-50 supervised torchvision model](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L28)):

      ```bash
      python tools/run_distributed_engines.py \
          config=fairness/finetune/dollar_street/eval_resnet_8gpu_transfer_dollarstreet_in22k_fulltune \
          +config/fairness/finetune/dollar_street/models=resnext50 \
          +config/fairness/finetune/dollar_street/dataset=dollar_street \
          config.OPTIMIZER.num_epochs=105 \
          config.OPTIMIZER.weight_decay=0.0001 \
          config.OPTIMIZER.regularize_bn=True \
          config.OPTIMIZER.nesterov=False \
          config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.base_value=0.0125 \
          config.OPTIMIZER.param_schedulers.lr.values=[0.0125,0.00125,0.000125,0.0000125,0.00000125] \
          config.OPTIMIZER.param_schedulers.lr.milestones=[30,60,90,100] \
          config.DISTRIBUTED.NUM_NODES=4 \
          config.TEST_EVERY_NUM_EPOCH=1 \
          config.MODEL.WEIGHTS_INIT.APPEND_PREFIX=trunk._feature_blocks. \
          config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME='' \
          config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/path/to/resnet50-0676ba61.pth
      ```
      
- **Step3: Inference on DollarStreet dataset**

  For the model that predicts label (for instance: finetuned model after Step2 above), perform the _inference_ on the DollarStreet to generate files: a) model label predictions for each image, b) model confidence score for each prediction. Using VISSL, this can be done with following example command:

  ```bash
  python tools/run_distributed_engines.py \
      config=extract_label_predictions/extract_predictions_resnet_in1k_8gpu \
      +config/fairness/extract_label_predictions/datasets=dollar_street_in22k_val \
      +config/fairness/extract_label_predictions/models/dollar_street=resnext50 \
      config.DISTRIBUTED.NUM_NODES=1 \
      config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
      config.TEST_ONLY=True \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/path/to/resnet50-0676ba61_finetuned.torch
   ```

- **Step4: Calculate the metrics**

  Using the predictions, scores data dumped in previous step, you can calculate the simple metrics by adapting the following script to your inputs:
  
  ```bash
  python projects/fairness_indicators/geographical_diversity/inference_geographical_diversity.py
  ```
