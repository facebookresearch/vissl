# Helpful scripts for data preparation and model conversion

We provide several helpful scripts to prepare data, to convert VISSL models to detectron2 compatible models or to convert [caffe2 models](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md) to VISSL compatible models.

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

## Generating Jigsaw Permutations for varying problem complexity
We provide scripts to change problem complexity of Jigsaw approach (as an axis of scaling in [paper](https://arxiv.org/abs/1905.01235)).


For the problem of Jigsaw, we vary the number of permutations used
to solve the jigsaw task. In the [paper](https://arxiv.org/abs/1905.01235), the permutations used `∈` [100, 2000, 10000]. We provide these permutations files for download [here](../MODEL_ZOO.md). To generate the permutations, use the command below:

```bash
python extra_scripts/generate_jigsaw_permutations.py \
    --output_dir /tmp/vissl//jigsaw_perms/ \
    -- N 2000
```

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

## Converting Models ClassyVision -> VISSL
We provide scripts to convert [ClassyVision](https://github.com/facebookresearch/ClassyVision) models to [VISSL](https://github.com/facebookresearch/vissl) compatible models.


```bash
python extra_scripts/convert_classy_vision_to_vissl_resnet.py \
    --input_model_file <input_model>.pth  \
    --output_model <d2_model>.torch \
    --depth 50
```


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
