# Helpful scripts for data preparation and model conversion

We provide several helpful scripts to prepare data, to convert VISSL models to detectron2 compatible models or to convert [caffe2 models](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md) to VISSL compatible models.

<br>

## Data preparation

VISSL supports benchmarks inspired by the [VTAB](https://arxiv.org/pdf/1910.04867.pdf) and [CLIP](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf) papers, for which the datasets do not directly exist but are transformations of existing dataset.

To run these benchmarks, the following data preparation scripts are mandatory:

- `create_clevr_count_data_files.py`: to create a dataset from [CLEVR](https://arxiv.org/abs/1612.06890) where the goal is to count the number of object in the scene
- `create_ucf101_data_files.py`: to create an image action recognition dataset from the video action recognition dataset [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) by extracting the middle frame

### Preparing CLEVR/Counts data files

Download the full dataset by visiting [Stanford CLEVR website](https://cs.stanford.edu/people/jcjohns/clevr/) and clicking on [Download CLEVR v1.0 (18 GB)](https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip) dataset.
Expand the archive.

The resulting folder should have the following structure:

```bash
CLEVR_v1.0/
    COPYRIGHT.txt 
    LICENSE.txt
    README.txt 
    images/
        train/
            ... 75000 images ...
        val/
            ... 15000 images ...
        test/
            ... 15000 images ...
    questions/
        CLEVR_test_questions.json
        CLEVR_train_questions.json
        CLEVR_val_questions.json
    scenes/
        CLEVR_train_scenes.json
        CLEVR_val_scenes.json
```

Run the script (where `/path/to/CLEVR_v1.0/` is the path to the expanded archive):

```bash
python extra_scripts/create_clevr_count_data_files.py \
    -i /path/to/CLEVR_v1.0/ \
    -o /output_path/clevr_count
```

The folder `/output_path/clevr_count` now contains the CLEVR/Counts dataset. The last step is to set this path in `dataset_catalog.json` and you are good to go:

```
"clevr_count_folder": {
    "train": ["/checkpoint/qduval/datasets/clevr_count/train", "<lbl_path>"],
    "val": ["/checkpoint/qduval/datasets/clevr_count/val", "<lbl_path>"]
}
```


### Preparing UCF101/image data files

Download the full dataset by visiting the [UCF101 website](https://www.crcv.ucf.edu/data/UCF101.php):

- Click on [The UCF101 data set can be downloaded by "clicking here"](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar) to retrieve the data (all the videos).
- Click on [The Train/Test Splits for Action Recognition on UCF101 data set can be downloaded by clicking here](https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip) to retrieve the splits.

Expand both archives in the same folder, say `/path/to/ucf101`.

The resulting folder should have the following structure:

```bash
ucf101/
    data/
        ... 13320 videos ...
    ucfTrainTestlist/
        classInd.txt
        testlist01.txt
        testlist02.txt
        testlist03.txt
        trainlist01.txt
        trainlist02.txt
        trainlist03.txt
```

Run the following commands (where `/path/to/ucf101` is the path of the folder above):

```bash
# To create the training split

python extra_scripts/create_ucf101_data_files.py \
    -d /path/to/ucf101/data \
    -a /path/to/ucf101/ucfTrainTestlist/trainlist01.txt \
    -o /output_path/ucf101/train

# To create the test split

python extra_scripts/create_ucf101_data_files.py \
    -d /path/to/ucf101/data \
    -a /path/to/ucf101/ucfTrainTestlist/testlist01.txt \
    -o /output_path/ucf101/test
```

The folder `/output_path/ucf101` now contains the UCF101 image action recognition dataset. The last step is to set this path in `dataset_catalog.json` and you are good to go:

```
"ucf101_folder": {
    "train": ["/checkpoint/qduval/vissl/ucf101/train", "<lbl_path>"],
    "val": ["/checkpoint/qduval/vissl/ucf101/test", "<lbl_path>"]
}
```

<br>

## Data preparation (optional)
The following scripts are optional as VISSL's `dataset_catalog.py` supports reading the downloaded data directly for most of these. For all the datasets below, we assume that the datasets are in the format as described [here](../vissl/data/README.md).

### Preparing COCO data files

```bash
python extra_scripts/create_coco_data_files.py \
    --json_annotations_dir /path/to/coco/annotations/ \
    --output_dir /tmp/vissl/datasets/coco/ \
    --train_imgs_path /path/to/coco/train2014 \
    --val_imgs_path /path/to/coco/val2014
```


### Preparing VOC data files

0. for VOC2007
data_source_dir='/mnt/fair/VOC2007/'
output_dir='/path/to/my/output/dir/voc2007/'

1. for VOC2012
data_source_dir='/mnt/fair/VOC2012/'
output_dir='/path/to/my/output/dir/voc2012'

- For VOC2007 dataset:
```bash
python extra_scripts/create_voc_data_files.py \
    --data_source_dir /path/to/VOC2007/ \
    --output_dir /tmp/vissl/datasets/voc07/
```

- For VOC2012 dataset:

```bash
python extra_scripts/create_voc_data_files.py \
    --data_source_dir /path/to/VOC2012/ \
    --output_dir /tmp/vissl/datasets/voc12/
```


### Preparing ImageNet and Places{205, 365} data files

```bash
python extra_scripts/create_imagenet_data_files.py \
    --data_source_dir /path/to/imagenet_full_size/ \
    --output_dir /tmp/vissl/datasets/imagenet1k/
```

```bash
python extra_scripts/create_imagenet_data_files.py \
    --data_source_dir /path/to/places205/ \
    --output_dir /tmp/vissl/datasets/places205/
```

```bash
python extra_scripts/create_imagenet_data_files.py \
    --data_source_dir /path/to/places365/ \
    --output_dir /tmp/vissl/datasets/places365/
```

### Low-shot data sampling
Low-shot image classification is one of the benchmark tasks in the [paper](https://arxiv.org/abs/1905.01235). VISSL support low-shot sampling and benchmarking on the PASCAL VOC dataset only.

We train on `trainval` split of VOC2007 dataset which has 5011 images and 20 classes.
Hence the labels are of shape 5011 x 20. We generate 5 independent samples (for a given low-shot value `k`) by essentially generating 5 independent target files. For each class, we randomly
pick the positive `k` samples and `19 * k` negatives. Rest of the samples are ignored. We perform low-shot image classification on various different layers on the model (AlexNet, ResNet-50). The targets `targets_data_file` is usually obtained by extracting features for a given layer. Below command generates 5 samples for various `k` values:

```bash
python extra_scripts/create_voc_low_shot_samples.py \
    --targets_data_file /path/to/voc/numpy_targets.npy \
    --output_path /tmp/vissl/datasets/voc07/low_shot/labels/ \
    --k_values "1,2,4,8,16,32,64,96" \
    --num_samples 5
```

<br>

## Generating Jigsaw Permutations for varying problem complexity
We provide scripts to change problem complexity of Jigsaw approach (as an axis of scaling in [paper](https://arxiv.org/abs/1905.01235)).


For the problem of Jigsaw, we vary the number of permutations used
to solve the jigsaw task. In the [paper](https://arxiv.org/abs/1905.01235), the permutations used `∈` [100, 2000, 10000]. We provide these permutations files for download [here](../MODEL_ZOO.md). To generate the permutations, use the command below:

```bash
python extra_scripts/generate_jigsaw_permutations.py \
    --output_dir /tmp/vissl//jigsaw_perms/ \
    -- N 2000
```

<br>

## Converting Models VISSL -> {Detectron2, ClassyVision, TorchVision}
We provide scripts to convert VISSL models to [Detectron2](https://github.com/facebookresearch/detectron2) and [ClassyVision](https://github.com/facebookresearch/ClassyVision) compatible models.

### Converting to Detectron2
All the ResNe(X)t models in VISSL can be converted to Detectron2 weights using following command:

```bash
python extra_scripts/convert_vissl_to_detectron2.py \
    --input_model_file <input_model>.pth  \
    --output_model <d2_model>.torch \
    --weights_type torch \
    --state_dict_key_name classy_state_dict
```

### Converting to ClassyVision
All the ResNe(X)t models in VISSL can be converted to Detectron2 weights using following command:

```bash
python extra_scripts/convert_vissl_to_classy_vision.py \
    --input_model_file <input_model>.pth  \
    --output_model <d2_model>.torch \
    --state_dict_key_name classy_state_dict
```

### Converting to TorchVision
All the ResNe(X)t models in VISSL can be converted to Torchvision weights using following command:

```bash
python extra_scripts/convert_vissl_to_torchvision.py \
    --model_url_or_file <input_model>.pth  \
    --output_dir /path/to/output/dir/ \
    --output_name <my_converted_model>.torch
```

<br>

## Converting Caffe2 models -> VISSL
We provide conversion of all the [caffe2 models](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md) in the [paper](https://arxiv.org/abs/1905.01235)

### ResNet-50 models to VISSL

All the models have been added to `ICCV19_MODEL_ZOO_FB.md`.

Jigsaw model:

```bash
python extra_scripts/convert_caffe2_to_torchvision_resnet.py \
    --c2_model <model>.pkl \
    --output_model <pth_model>.torch \
    --jigsaw True --bgr2rgb True
```

Colorization model:

```
python extra_scripts/convert_caffe2_to_torchvision_resnet.py \
    --c2_model <model>.pkl \
    --output_model <pth_model>.torch \
    --bgr2rgb False
```

Supervised model:

```
python extra_scripts/convert_caffe2_to_pytorch_rn50.py \
    --c2_model <model>.pkl \
    --output_model <pth_model>.torch \
    --bgr2rgb True
```

### AlexNet models to VISSL

AlexNet Jigsaw models:
```
python extra_scripts/convert_caffe2_to_vissl_alexnet.py \
    --weights_type caffe2 \
    --model_name jigsaw \
    --bgr2rgb True \
    --input_model_weights <model.pkl> \
    --output_model <pth_model>.torch
```

AlexNet Colorization models:
```
python extra_scripts/convert_caffe2_to_vissl_alexnet.py \
    --weights_type caffe2 \
    --model_name colorization \
    --input_model_weights <model.pkl> \
    --output_model <pth_model>.torch
```

AlexNet Supervised models:
```
python extra_scripts/convert_caffe2_to_vissl_alexnet.py \
    --weights_type caffe2 \
    --model_name supervised \
    --bgr2rgb True \
    --input_model_weights <model.pkl> \
    --output_model <pth_model>.torch
```

<br>

## Converting Models ClassyVision -> VISSL
We provide scripts to convert [ClassyVision](https://github.com/facebookresearch/ClassyVision) models to [VISSL](https://github.com/facebookresearch/vissl) compatible models.


```bash
python extra_scripts/convert_classy_vision_to_vissl_resnet.py \
    --input_model_file <input_model>.pth  \
    --output_model <d2_model>.torch \
    --depth 50
```

<br>

## Converting Official RotNet and DeepCluster models -> VISSL

AlexNet RotNet model:
```
python extra_scripts/convert_caffe2_to_vissl_alexnet.py \
    --weights_type torch \
    --model_name rotnet \
    --input_model_weights <model> \
    --output_model <pth_model>.torch
```

AlexNet DeepCluster model:
```
python extra_scripts/convert_alexnet_models.py \
    --weights_type torch \
    --model_name deepcluster \
    --input_model_weights <model> \
    --output_model <pth_model>.torch
```
