## Indicator1: Harmful Label association

- **Step1: Setup Data**

  The indicator performs inference on OpenImages MIAP and Casual Conversations dataset. Optionally, if your model doesn't predict labels, then the model needs to be finetuned to predict labels. The finetuning can be done on a subset of ImageNet-22K classes. 

   -  __[Optional][Finetuning] Setting up ImageNet-22k subset__: ImageNet-22K dataset has 21,842 classes which map to the labels in wordnet. The full ImageNet-22K taxonomy is available [here](https://github.com/atong01/Imagenet-Tensorflow/blob/master/model/imagenet_synset_to_human_label_map.txt). Subsample the dataset corresponding to the [619 classes in taxonomy](https://dl.fbaipublicfiles.com/vissl/fairness/label_association/imagenet_to_idx_labels_map.json). Note that the classes with prefix `1k_` indicate these classes are part of ImageNet-1K dataset as well.

   - __[Inference] Setup of Casual Conversations (CC)__: Following the [CC dataset guidelines](https://ai.facebook.com/datasets/casual-conversations-dataset/), obtain access to the dataset. We setup the _mini-test_ split of the dataset which has 2982 videos. We follow the [CC paper Section 4.1](https://scontent-iad3-2.xx.fbcdn.net/v/t39.8562-6/10000000_1032795703933429_5521369258245261270_n.pdf?_nc_cat=111&ccb=1-5&_nc_sid=ae5e01&_nc_ohc=_S3fHxQzy6sAX9C7nMJ&_nc_ht=scontent-iad3-2.xx&oh=00_AT-wqHBSuj1fqg4yn8hK4U_ar57IUOjSQ776fiLGu_2FWw&oe=6208F25B) for inference on face crops. Please refer to our research [Appendix B.1](**TODO**) for further details. We parse the dataset to generate a single _.json_ file with the format:
      ``` 
      [
        {"age": "20", "gender": "Female", "img_path": "/path/to/img1.jpg", "skin-type": "5", "subject_id": "1"},
        {....},
        ...
      ]
      ```
   

   - __[Inference] Setup of OpenImages MIAP__: Following the [OpenImages MIAP guidelines](https://storage.googleapis.com/openimages/open_images_extended_miap/Open%20Images%20Extended%20-%20MIAP%20-%20Data%20Card.pdf), download the dataset. We setup the _test_ split of dataset which contains 22,590 unique images where each image has multiple bounded boxes. Please refer to our research [Appendix B.2](**TODO**) for further details. We parse the dataset to generate a single _.json_ file with the format:
      ``` 
      [
        {"AgePresentation": "Young", "Confidence": "1", "GenderPresentation": "Unknown", "IsDepictionOf": "0", "IsGroupOf": "0", "IsInsideOf": "0", "IsOccluded": "0", "IsTruncated": "0", "LabelName": "/m/01g317", "bbox": ["886.5607679999999", "302.212474", "1016.448", "639.179403"], "path": "/path/to/img1.jpg"},
        {....},
        ...
      ]
      ```

- **Step2 [Optional]: Finetune the model** 

   This step is only needed if you need to finetune your model such that it can generate label predictions (if it doesn't already). 
   - You can use VISSL for this purpose. [Install VISSL](https://github.com/facebookresearch/vissl/blob/main/INSTALL.md).

   - Make your model compatible with VISSL [following these guidelines](https://vissl.readthedocs.io/en/latest/evaluations/load_models.html).

   - Make the input data format compatible by [following these guidelines](https://vissl.readthedocs.io/en/latest/vissl_modules/data.html#using-data).

   - Finetune your model. Example command (for [ResNet-50 supervised torchvision model](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L28)):

      ```bash
      python tools/run_distributed_engines.py \
          config=fairness/finetune/in22k_subset/eval_resnet_8gpu_transfer_in22k_subset_fulltune \
          +config/fairness/finetune/in22k_subset/models=resnext50_in22k_subset \
          +config/fairness/finetune/in22k_subset/dataset=in22k_subset \
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

- **Step3: Inference on the Casual Conversations and Openimages MIAP**

  For the model that predicts label (for instance: finetuned model after Step2 above), perform the _inference_ on the CC and OpenImages MIAP datasets to generate files: a) model label predictions for each image, b) model confidence score for each prediction. Using VISSL, this can be done with following example command:
  
  - Inference on CC

      ```bash
      python tools/run_distributed_engines.py \
          config=extract_label_predictions/extract_predictions_resnet_in1k_8gpu \
          +config/fairness/extract_label_predictions/datasets=casual_conversations_face_crops_mini \
          +config/fairness/extract_label_predictions/models/in22k_subset=resnext50_in22k_subset \
          config.DISTRIBUTED.NUM_NODES=1 \
          config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
          config.TEST_ONLY=True \
          config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/path/to/resnet50-0676ba61_finetuned.torch
      ```
      
  - Inference on OpenImages MIAP
      ```bash
      python tools/run_distributed_engines.py \
          config=extract_label_predictions/extract_predictions_resnet_in1k_8gpu \
          +config/fairness/extract_label_predictions/datasets=openimages_miap_test \
          +config/fairness/extract_label_predictions/models/in22k_subset=resnext50_in22k_subset \
          config.DISTRIBUTED.NUM_NODES=1 \
          config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
          config.TEST_ONLY=True \
          config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/path/to/resnet50-0676ba61_finetuned.torch
      ```

- **Step4: Calculate the metrics**
  Using the predictions, scores data dumped in previous step, you can calculate the simple metrics:
  - Inference on CC
    
    ```bash
    python projects/fairness_indicators/harmful_label_associations/inference_label_assoc_casual_conversations.py
    ```
  
  - Inference on OpenImages MIAP

    ```bash
    python projects/fairness_indicators/harmful_label_associations/inference_label_assoc_openimages_miap_test.py
    ```
