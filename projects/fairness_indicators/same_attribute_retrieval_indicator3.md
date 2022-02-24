## Indicator3: Similarity based attribute retrieval

- **Step1: Setup Data**

  This indicator performs similarity search which requires a _Database_(UTK-Faces) and _Queries_(Casual Conversations). 
 
  - _Setup UTK Faces (Database)_: Users can [download UTK Faces dataset](https://susanqq.github.io/UTKFace/). We provide the [full metadata for UTK faces](https://dl.fbaipublicfiles.com/vissl/fairness/similarity_search/utk_dataset_metadata.json) corresponding the images in this dataset. You can use [this script](https://github.com/facebookresearch/vissl/tree/main/extra_scripts/datasets/create_utk_faces_filelist.py) to convert the downloaded dataset into the input format for VISSL dataloader.

  - _Setup Casual Conversations (Queries)_: Following the [CC dataset guidelines](https://ai.facebook.com/datasets/casual-conversations-dataset/), obtain access to the dataset. We setup the _mini-test_ split of the dataset which has 2982 videos. We follow the [CC paper Section 4.1](https://scontent-iad3-2.xx.fbcdn.net/v/t39.8562-6/10000000_1032795703933429_5521369258245261270_n.pdf?_nc_cat=111&ccb=1-5&_nc_sid=ae5e01&_nc_ohc=_S3fHxQzy6sAX9C7nMJ&_nc_ht=scontent-iad3-2.xx&oh=00_AT-wqHBSuj1fqg4yn8hK4U_ar57IUOjSQ776fiLGu_2FWw&oe=6208F25B) for inference on face crops. Please refer to our research [Appendix B.1](**TODO**) for further details. We parse the dataset to generate a single _.json_ file with the format:
      ``` 
      [
        {"age": "20", "gender": "Female", "img_path": "/path/to/img1.jpg", "skin-type": "5", "subject_id": "1"},
        {....},
        ...
      ]
      ```

- **Step2: Extract Model features on UTK Faces and CC**
  
  After setting up data, the next step is to extract the model features on both these datasets. You can use VISSL for this:
  
  - You can use VISSL for this purpose. [Install VISSL](https://github.com/facebookresearch/vissl/blob/main/INSTALL.md).

  - Make your model compatible with VISSL [following these guidelines](https://vissl.readthedocs.io/en/latest/evaluations/load_models.html).

  - Make the input data format compatible by [following these guidelines](https://vissl.readthedocs.io/en/latest/vissl_modules/data.html#using-data).

  - Extract features. Example command (for [ResNet-50 supervised torchvision model](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L28)):
    ```bash
    python tools/run_distributed_engines.py \
        config=feature_extraction/extract_resnet_in1k_8gpu \
        +config/fairness/nearest_neighbor/models=resnext50_supervised \
        +config/fairness/nearest_neighbor/datasets=utk_train_cc_face_crops_scale1pt5_gender \
        config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=32 \
        config.DATA.TEST.BATCHSIZE_PER_REPLICA=32 \
        config.DISTRIBUTED.NUM_NODES=1 \
        config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
        config.LOG_FREQUENCY=1 \
        config.TEST_MODEL=True \
        config.TEST_ONLY=False \
        config.EXTRACT_FEATURES.CHUNK_THRESHOLD=-1 \
        config.MODEL.WEIGHTS_INIT.APPEND_PREFIX=trunk._feature_blocks. \
        config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME='' \
        config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/path/to/resnet50-0676ba61.pth
    ```

- **Step3: Calculate the metric**

  Using the features dumped, you can do the similarity search and calculate Precision@K metrics using the following script. Note that you should adapt the inputs of the script to your inputs.
  ```bash
  python projects/fairness_indicators/similarity_search/inference_casual_conversations.py
  ```
