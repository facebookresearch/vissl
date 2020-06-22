## Input Parameter Values for various datasets
Below are the input values to the scripts for various datasets. The instructions
for running the scripts are also listed in the end.

### Preparing COCO data files

json_annotations_dir='/mnt/vol/gfsai-east/ai-group/datasets/json_dataset_annotations/coco'
output_dir='/mnt/vol/gfsai-east/ai-group/users/prigoyal/caffe2/resnet/gen/coco/'
train_imgs_path='/data/local/packages/ai-group.coco_train2014/prod/coco_train2014'
val_imgs_path='/data/local/packages/ai-group.coco_val2014/prod/coco_val2014'


### Preparing VOC data files

0. for VOC2007
data_source_dir='/mnt/fair/VOC2007/'
output_dir='/mnt/vol/gfsai-east/ai-group/users/prigoyal/caffe2/resnet/gen/voc2007/'

1. for VOC2012
data_source_dir='/mnt/fair/VOC2012/'
output_dir='/mnt/vol/gfsai-east/ai-group/users/prigoyal/caffe2/resnet/gen/voc2012'


### Preparing ImageNet and Places205 data files

0. For ImageNet-1K
You might need to install imagenet1k dataset on your devgpu if the dataset does
not exist already. To install, follow the below instructions:

* Edit `/data/local/packages/packages.txt` and add `ai-group.imagenet-full-size`

* Run `sudo chefctl -i` and wait for the chef run to finish. After the chef run,
your data will be under `/data/local/packages/ai-group.imagenet-full-size`


data_source_dir='/data/local/packages/ai-group.imagenet-full-size/prod/imagenet_full_size/'
output_dir='/mnt/vol/gfsai-east/ai-group/users/prigoyal/caffe2/resnet/gen/imagenet1k/'

1. For Places-205 (see https://fb.quip.com/2BH7AoRi7uVg)
There exists a squashfs image that can be mounted to the local folder (for fast
reading):
```
mkdir /tmp/places205 && squashfuse /mnt/vol/gfsai-east/ai-group/users/imisra/datasets/places205.img /tmp/places205
```
data_source_dir='/tmp/places205/'
output_dir='/mnt/vol/gfsai-east/ai-group/users/prigoyal/caffe2/resnet/gen/places205/'


## Building and running

0. Build using command
```
buck build @mode/dev-nosan deeplearning/projects/ssl_framework/extra_scripts/...
```

## Create Data input files

0. For COCO data:
Assuming the output dir to be /home/prigoyal/local/coco/

```
buck-out/gen/deeplearning/projects/ssl_framework/extra_scripts/create_coco_data_files.par --json_annotations_dir /mnt/vol/gfsai-east/ai-group/datasets/json_dataset_annotations/coco --output_dir /home/prigoyal/local/vissl/coco/ --train_imgs_path /data/local/packages/ai-group.coco_train2014/prod/coco_train2014 --val_imgs_path /data/local/packages/ai-group.coco_val2014/prod/coco_val2014
```

0. For VOC2007 dataset:
Combining train and val and output dir to be /home/prigoyal/local/voc07
```
buck-out/gen/deeplearning/projects/ssl_framework/extra_scripts/create_voc_data_files.par --data_source_dir /mnt/fair/VOC2007/ --output_dir /home/prigoyal/local/vissl/voc07/
```

0. For VOC2012 dataset:
```
buck-out/gen/deeplearning/projects/ssl_framework/extra_scripts/create_voc_data_files.par --data_source_dir /mnt/fair/VOC2012/ --output_dir /home/prigoyal/local/vissl/voc12/
```

0. For ImageNet-1k dataset:
```
buck-out/gen/deeplearning/projects/ssl_framework/extra_scripts/create_imagenet_data_files.par --data_source_dir /data/local/packages/ai-group.imagenet-full-size/prod/imagenet_full_size/ --output_dir /home/prigoyal/local/vissl/imagenet1k/
```

0. For Places-205 dataset:
```
buck-out/gen/deeplearning/projects/ssl_framework/extra_scripts/create_imagenet_data_files.par --data_source_dir /tmp/places205/ --output_dir /home/prigoyal/local/vissl/places205/
```

## Converting PyTorch RN50 weights to VISSL weights

* To load from a file:
```
buck-out/gen/deeplearning/projects/ssl_framework/extra_scripts/convert_torchvision_resnet_weights.par --model_url_or_file /home/prigoyal/local/resnet50-19c8e357.pth --output_dir /home/prigoyal/local/
```

* To load from a url:
```
buck-out/gen/deeplearning/projects/ssl_framework/extra_scripts/convert_torchvision_resnet_weights.par --model_url_or_file https://download.pytorch.org/models/resnet50-19c8e357.pth --output_dir /home/prigoyal/local/
```

## Pickle ICCV'19 Caffe2 ResNet-50 models to PyTorch

All the models have been added to `ICCV19_MODEL_ZOO_FB.md`.

Jigsaw model:

```
buck-out/gen/deeplearning/projects/ssl_framework/extra_scripts/convert_caffe2_to_pytorch_rn50.par --c2_model <model>.pkl --output_model <pth_model>.torch --jigsaw True --bgr2rgb True
```

Colorization model:

```
buck-out/gen/deeplearning/projects/ssl_framework/extra_scripts/convert_caffe2_to_pytorch_rn50.par --c2_model <model>.pkl --output_model <pth_model>.torch --bgr2rgb False
```

Supervised model:

```
buck-out/gen/deeplearning/projects/ssl_framework/extra_scripts/convert_caffe2_to_pytorch_rn50.par --c2_model <model>.pkl --output_model <pth_model>.torch --bgr2rgb True
```

## Convert various AlexNet models to PyTorch compatible with VISSL

All AlexNet models have been added to `ICCV19_MODEL_ZOO_FB.md`

AlexNet Jigsaw models:
```
buck-out/gen/deeplearning/projects/ssl_framework/extra_scripts/convert_alexnet_models.par --weights_type caffe2 --model_name jigsaw --bgr2rgb True --input_model_weights <model.pkl> --output_model <pth_model>.torch
```

AlexNet Colorization models:
```
buck-out/gen/deeplearning/projects/ssl_framework/extra_scripts/convert_alexnet_models.par --weights_type caffe2 --model_name colorization --input_model_weights <model.pkl> --output_model <pth_model>.torch
```

AlexNet Supervised models:
```
buck-out/gen/deeplearning/projects/ssl_framework/extra_scripts/convert_alexnet_models.par --weights_type caffe2 --model_name supervised --bgr2rgb True --input_model_weights <model.pkl> --output_model <pth_model>.torch
```

AlexNet RotNet model:
```
buck-out/gen/deeplearning/projects/ssl_framework/extra_scripts/convert_alexnet_models.par --weights_type torch --model_name rotnet --input_model_weights <model> --output_model <pth_model>.torch
```

AlexNet DeepCluster model:
```
buck-out/gen/deeplearning/projects/ssl_framework/extra_scripts/convert_alexnet_models.par --weights_type torch --model_name deepcluster --input_model_weights <model> --output_model <pth_model>.torch
```

## Convert various VISSL ResNet models to Detectron2

All the ResNet models that are VISSL compatible can be converted to Detectron2 weights using following command:

```
buck-out/gen/deeplearning/projects/ssl_framework/extra_scripts/convert_vissl_resnet_to_detectron2.par --input_model_file <input_model>.pth  --output_model <d2_model>.torch --weights_type torch --state_dict_key_name model_state_dict
```


## Generating Jigsaw Permutations

```
buck-out/gen/deeplearning/projects/ssl_framework/extra_scripts/generate_jigsaw_permutations.par --output_dir <output_dir_path>
```
