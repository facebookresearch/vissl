# Helpful scripts for data preparation and model conversion

We provide several helpful scripts to prepare data, to convert VISSL models to detectron2 compatible models or to convert [caffe2 models](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md) to VISSL compatible models.

<br>

## Data preparation

VISSL supports benchmarks inspired by the [VTAB](https://arxiv.org/pdf/1910.04867.pdf) and [CLIP](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf) papers, for which the datasets do not directly exist but are transformations of existing dataset.

To run these benchmarks, the following data preparation scripts are mandatory:

- `extra_scripts/datasets/create_clevr_count_data_files.py`: to create a `disk_filelist` dataset from [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) where the goal is to count the number of object in the scene
- `extra_scripts/datasets/create_clevr_dist_data_files.py`: to create a `disk_filelist` dataset from [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) where the goal is to estimate the distance to the closest object in the scene
- `extra_scripts/datasets/create_dsprites_location_data_files.py`: to create a `disk_folder` dataset from [dSprites](https://github.com/deepmind/dsprites-dataset) where the goal is to estimate the x coordinate of the sprite on the scene
- `extra_scripts/datasets/create_dsprites_orientation_data_files.py`: to create a `disk_folder` dataset from [dSprites](https://github.com/deepmind/dsprites-dataset) where the goal is to estimate the orientation of the sprite on the scene
- `extra_scripts/datasets/create_euro_sat_data_files.py`: to transform the [EUROSAT](https://github.com/phelber/eurosat) dataset to the `disk_folder` format
- `extra_scripts/datasets/create_food101_data_files.py`: to transform the [FOOD101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101) dataset to the `disk_folder` format
- `extra_scripts/datasets/create_imagenet_ood_data_files.py`: to create test sets in `disk_filelist` format for Imagenet based on [Imagenet-A](https://github.com/hendrycks/natural-adv-examples) and [Imagenet-R](https://github.com/hendrycks/imagenet-r)
- `extra_scripts/datasets/create_kitti_dist_data_files.py`: to create a `disk_folder` dataset from [KITTI](http://www.cvlibs.net/datasets/kitti/) where the goal is to estimate the distance of the closest car, van or truck
- `extra_scripts/datasets/create_patch_camelyon_data_files.py`: to transform the [PatchCamelyon](https://github.com/basveeling/pcam) dataset to the `disk_folder` format
- `extra_scripts/datasets/create_small_norb_azimuth_data_files.py` to create a `disk_folder` dataset from [Small NORB](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/) where the goal is to find the azimuth or the photographed object
- `extra_scripts/datasets/create_small_norb_elevation_data_files.py` to create a `disk_folder` dataset from [Small NORB](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/) where the goal is to predict the elevation in the image
- `extra_scripts/datasets/create_sun397_data_files.py` to transform the [SUN397](https://vision.princeton.edu/projects/2010/SUN/) dataset to the `disk_filelist` format
- `extra_scripts/datasets/create_ucf101_data_files.py`: to create a `disk_folder` image action recognition dataset from the video action recognition dataset [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) by extracting the middle frame

### Unified data preparation interface

All of these scripts follow the same easy to use interface:

```
python create_[***]_data_files.py -i /path/to/input_datset -o /path/to/tranformed/dataset -d
```

- `-i` gives the path to the official dataset format
- `-o` gives the path to the output transformed dataset (the one to feed to VISSL)
- `-d` (optional) automatically downloads the dataset in the input path

Scripts producing a `disk_filelist` format will create the following structure:

```
output_folder/
    train_images.npy  # Paths to the train images
    train_labels.npy  # Labels for each of the train images
    val_images.npy    # Paths to the val images
    val_labels.npy    # Labels for each of the val images
```

These files should be referenced in the `dataset_catalog.json` like so:

```json
"dataset_filelist": {
    "train": ["/path/to/train_images.npy", "/path/to/train_labels.npy"],
    "val": ["/path/to/val_images.npy", "/path/to/val_labels.npy"]
},
```

Scripts producing a `disk_folder` format will create the following structure:

```
train/
    label1/
        image_1.jpeg
        image_2.jpeg
        ...
    label2/
        image_x.jpeg
        image_y.jpeg
        ...
    ...
val/
    label1/
        image_1.jpeg
        image_2.jpeg
        ...
    label2/
        image_x.jpeg
        image_y.jpeg
        ...
    ...
```

These files should be referenced in the `dataset_catalog.json` like so:

```json
"dataset_folder": {
    "train": ["/path/to/dataset/train", "<ignored>"],
    "val": ["/path/to/dataset/val", "<ignored>"]
},
```

The following sections will describe each of these data preparation scripts in detail.

### Preparing CLEVR/Counts data files

#### Automatic download

Run the `create_clevr_count_data_files.py` script with the `-d` option as follows:

```bash
python extra_scripts/datasets/create_clevr_count_data_files.py \
    -i /path/to/clevr/ \
    -o /output_path/to/clevr_count
    -d
```

The folder `/output_path/clevr_count` now contains the CLEVR/Counts `disk_filelist` dataset.
The last step is to set this path in `dataset_catalog.json` and you are good to go:

```
"clevr_count_filelist": {
    "train": ["/output_path/to/clevr_count/train_images.npy", "/output_path/to/clevr_count/train_labels.npy"],
    "val": ["/output_path/to/clevr_count/val_images.npy", "/output_path/to/clevr_count/val_labels.npy"]
},
```

#### Manual download

Download the full dataset by visiting [CLEVR website](https://cs.stanford.edu/people/jcjohns/clevr/) and clicking on [Download CLEVR v1.0 (18 GB)](https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip) dataset.
Expand the archive.

The resulting folder should have the following structure:

```bash
/path/to/clevr/
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

Run the script where `/path/to/clevr/` is the path of the folder containing the `CLEVR_v1.0` folder:

```bash
python extra_scripts/datasets/create_clevr_count_data_files.py \
    -i /path/to/clevr/ \
    -o /output_path/to/clevr_count
```

The folder `/output_path/clevr_count` now contains the CLEVR/Counts dataset.

### Preparing CLEVR/Dist data files

Follow the exact same steps as for the preparation of the CLEVR/Count dataset described above, but use `create_clevr_dist_data_files.py` instead of `create_clevr_count_data_files.py`.

Once the dataset is prepared and available at `/path/to/clevr_dist`, the last step is to set this path in `dataset_catalog.json` and you are good to go:

```
"clevr_dist_filelist": {
    "train": ["/path/to/clevr_dist/train_images.npy", "/path/to/clevr_dist/train_labels.npy"],
    "val": ["/path/to/clevr_dist/val_images.npy", "/path/to/clevr_dist/val_labels.npy"]
},
```

### Preparing the dSprites/location data files

Run the `create_dsprites_location_data_files.py` script with the `-d` option as follows:

```bash
python extra_scripts/datasets/create_dsprites_location_data_files.py \
    -i /path/to/dsprites/ \
    -o /output_path/to/dsprites_loc
    -d
```

The folder `/output_path/to/dsprites_loc` now contains the dSprites/location `disk_folder` dataset.
The last step is to set this path in `dataset_catalog.json` and you are good to go:

```
"dsprites_loc_folder": {
    "train": ["/output_path/to/dsprites_loc/train", "<ignored>"],
    "val": ["/output_path/to/dsprites_loc/val", "<ignored>"]
},
```

### Preparing the dSprites/orientation data files

Run the `create_dsprites_orientation_data_files.py` script with the `-d` option as follows:

```bash
python extra_scripts/datasets/create_dsprites_orientation_data_files.py \
    -i /path/to/dsprites/ \
    -o /output_path/to/dsprites_orient
    -d
```

The folder `/output_path/to/dsprites_orient` now contains the dSprites/orientation `disk_folder` dataset.
The last step is to set this path in `dataset_catalog.json` and you are good to go:

```
"dsprites_orient_folder": {
    "train": ["/output_path/to/dsprites_orient/train", "<ignored>"],
    "val": ["/output_path/to/dsprites_orient/val", "<ignored>"]
},
```

### Preparing the Caltech101 data files

Run the `create_caltech101_data_files.py` script with the `-d` option as follows:

```bash
python extra_scripts/datasets/create_caltech101_data_files.py \
    -i /path/to/caltech101/ \
    -o /output_path/to/caltech101
    -d
```

The folder `/output_path/to/caltech101` now contains the Caltech101 `disk_folder` dataset.
The last step is to set this path in `dataset_catalog.json` and you are good to go:

```
"caltech101_folder": {
    "train": ["/output_path/to/caltech101/train", "<ignored>"],
    "val": ["/output_path/to/caltech101/test", "<ignored>"]
},
```

### Preparing the DTD data files

Run the `create_dtd_data_files.py` script with the `-d` option as follows:

```bash
python extra_scripts/datasets/create_dtd_data_files.py \
    -i /path/to/dtd/ \
    -o /output_path/to/dtd
    -d
```

The folder `/output_path/to/dtd` now contains the DTD `disk_folder` dataset.
The last step is to set this path in `dataset_catalog.json` and you are good to go:

```
"dtd_folder": {
    "train": ["/output_path/to/dtd/train", "<ignored>"],
    "val": ["/output_path/to/dtd/test", "<ignored>"]
},
```

### Preparing the EuroSAT data files

Run the `create_euro_sat_data_files.py` script with the `-d` option as follows:

```bash
python extra_scripts/datasets/create_euro_sat_data_files.py \
    -i /path/to/euro_sat/ \
    -o /output_path/to/euro_sat
    -d
```

The folder `/output_path/to/euro_sat` now contains the EuroSAT `disk_folder` dataset.
The last step is to set this path in `dataset_catalog.json` and you are good to go:

```
"euro_sat_folder": {
    "train": ["/output_path/to/euro_sat/train", "<ignored>"],
    "val": ["/output_path/to/euro_sat/val", "<ignored>"]
},
```

### Preparing the Food-101 data files

Run the `create_food101_data_files.py` script with the `-d` option as follows:

```bash
python extra_scripts/datasets/create_food101_data_files.py \
    -i /path/to/food101/ \
    -o /output_path/to/food101
    -d
```

The folder `/output_path/to/food101` now contains the Food101 `disk_folder` dataset.
The last step is to set this path in `dataset_catalog.json` and you are good to go:

```
"food101_folder": {
    "train": ["/output_path/to/food101/train", "<ignored>"],
    "val": ["/output_path/to/food101/val", "<ignored>"]
},
```


### Preparing the FGVC Aircrafts data files

Run the `create_fgvc_aircraft_data_files.py` script with the `-d` option as follows:

```bash
python extra_scripts/datasets/create_fgvc_aircraft_data_files.py \
    -i /path/to/aircrafts/ \
    -o /output_path/to/aircrafts
    -d
```

The folder `/output_path/to/aircrafts` now contains the FGVC Aircrafts `disk_folder` dataset.
The last step is to set this path in `dataset_catalog.json` and you are good to go:

```
"aircrafts_folder": {
    "train": ["/output_path/to/aircrafts/trainval", "<ignored>"],
    "val": ["/output_path/to/aircrafts/test", "<ignored>"]
},
```

### Preparing the GTSRB data files

Run the `create_gtsrb_data_files.py` script with the `-d` option as follows:

```bash
python extra_scripts/datasets/create_gtsrb_data_files.py \
    -i /path/to/gtsrb/ \
    -o /output_path/to/gtsrb
    -d
```

The folder `/output_path/to/gtsrb` now contains the GTSRB `disk_folder` dataset.
The last step is to set this path in `dataset_catalog.json` and you are good to go:

```
"gtsrb_folder": {
    "train": ["/output_path/to/gtsrb/train", "<ignored>"],
    "val": ["/output_path/to/gtsrb/test", "<ignored>"]
},
```

### Preparing the Imagenet-A and Imagenet-R data files

Run the `create_imagenet_ood_data_files.py` script with the `-d` option as follows:

```bash
python extra_scripts/datasets/create_imagenet_ood_data_files.py \
    -i /path/to/input_folder/ \
    -o /path/to/output/
    -d
```

After running this script:
- The folder `/path/to/input_folder/` will contain the expanded `imagenet-a` and `imagenet-r` dataset in their original format
- The folder `/path/to/output/imagenet-a` will contain the Imagenet-A `disk_filelist` to provide to VISSL
- The folder `/path/to/output/imagenet-r` will contain the Imagenet-R `disk_filelist` to provide to VISSL

Note: all these folders are necessary as the `disk_filelist` format references images and does not copy them. Deleting `/path/to/input_folder/` will result in an error during training.

The last step is to set these paths in `dataset_catalog.json` and you are good to go:

```
"imagenet_a_filelist": {
    "train": ["<not_used>", "<not_used>"],
    "val": ["/path/to/output/imagenet-a/test_images.npy", "/path/to/output/imagenet-a/test_labels.npy"]
},
"imagenet_r_filelist": {
    "train": ["<not_used>", "<not_used>"],
    "val": ["/path/to/output/imagenet-r/test_images.npy", "/path/to/output/imagenet-r/test_labels.npy"]
},
```

### Preparing iNaturalist2018 data files

#### Automatic download

Run the `create_inaturalist2018_data_files.py` script with the `-d` option as follows:

```bash
python extra_scripts/datasets/create_inaturalist2018_data_files.py \
    -i /path/to/inaturalist2018/ \
    -o /output_path/to/inaturalist2018
    -d
```

The folder `/output_path/to/inaturalist2018` now contains the inaturalist2018 `disk_filelist` dataset.
The last step is to set this path in `dataset_catalog.json` and you are good to go:

```
"inaturalist2018_filelist": {
    "train": ["/output_path/to/inaturalist2018/train_images.npy", "/output_path/to/inaturalist2018/train_labels.npy"],
    "val": ["/output_path/to/inaturalist2018/val_images.npy", "/output_path/to/inaturalist2018/val_labels.npy"]
},
```

#### Manual download

Download the full dataset by visiting the [Inaturalist competion Github](https://github.com/visipedia/inat_comp/tree/master/2018#data) and downloading the "All training and validation images [120GB]", "Training annotations [26MB]", and "Validation annotations [26MB]" into the same directory, for example: "/path/to/inaturalist2018/". Expand each `.tar` archive.

The resulting folder should have the following structure:

```bash
/path/to/inaturalist2018/
    train_val2018/
        Actinopterygii/
            2229
                ... Images for class 2229 ...
            2230
                ... Images for class 2230 ...
            ...
        Amphibia
        ... All 14 "super categories" ...
    train2018.json
    val2018.json
```

Run the script where `/path/to/inaturalist2018/` is the path of the folder containing the expanded tars:

```bash
python extra_scripts/datasets/create_inaturalist2018_data_files.py \
    -i /path/to/inaturalist2018/ \
    -o /output_path/to/inaturalist2018
```

The folder `/output_path/to/inaturalist2018` now contains the inaturalist2018 `disk_filelist` .

### Preparing the KITTI/distance data files

Run the `create_kitti_dist_data_files.py` script with the `-d` option as follows:

```bash
python extra_scripts/datasets/create_kitti_dist_data_files.py \
    -i /path/to/kitti/ \
    -o /output_path/to/kitti_distance
    -d
```

The folder `/output_path/to/kitti_distance` now contains the KITTI/distance `disk_folder` dataset.
The last step is to set this path in `dataset_catalog.json` and you are good to go:

```
"kitti_dist_folder": {
    "train": ["/output_path/to/kitti_distance/train", "<ignored>"],
    "val": ["/output_path/to/kitti_distance/val", "<ignored>"]
},
```

### Preparing the Oxford Flowers data files

Run the `create_oxford_flowers_data_files.py` script with the `-d` option as follows:

```bash
python extra_scripts/datasets/create_oxford_flowers_data_files.py \
    -i /path/to/flowers/ \
    -o /output_path/to/flowers
    -d
```

The folder `/output_path/to/flowers` now contains the Oxford Pets `disk_folder` dataset.
The last step is to set this path in `dataset_catalog.json` and you are good to go:

```
"flowers_folder": {
    "train": ["/output_path/to/flowers/train", "<ignored>"],
    "val": ["/output_path/to/flowers/test", "<ignored>"]
},
```

### Preparing the Oxford Pets data files

Run the `create_oxford_pets_data_files.py` script with the `-d` option as follows:

```bash
python extra_scripts/datasets/create_oxford_pets_data_files.py \
    -i /path/to/pets/ \
    -o /output_path/to/pets
    -d
```

The folder `/output_path/to/pets` now contains the Oxford Pets `disk_folder` dataset.
The last step is to set this path in `dataset_catalog.json` and you are good to go:

```
"oxford_pets_folder": {
    "train": ["/output_path/to/pets/train", "<ignored>"],
    "val": ["/output_path/to/pets/test", "<ignored>"]
},
```


### Preparing the Patch Camelyon data files

Run the `create_patch_camelyon_data_files.py` script with the `-d` option as follows:

```bash
python extra_scripts/datasets/create_patch_camelyon_data_files.py \
    -i /path/to/pcam/ \
    -o /output_path/to/pcam
    -d
```

The folder `/output_path/to/pcam` now contains the Patch Camelyon `disk_folder` dataset.
The last step is to set this path in `dataset_catalog.json` and you are good to go:

```
"pcam_folder": {
    "train": ["/output_path/to/pcam/train", "<ignored>"],
    "val": ["/output_path/to/pcam/val", "<ignored>"]
},
```

### Preparing the SmallNORB/azimuth data files

Run the `create_small_norb_azimuth_data_files.py` script with the `-d` option as follows:

```bash
python extra_scripts/datasets/create_small_norb_azimuth_data_files.py \
    -i /path/to/snorb/ \
    -o /output_path/to/snorb_azimuth
    -d
```

The folder `/output_path/to/snorb_azimuth` now contains the SmallNORB/azimuth `disk_folder` dataset.
The last step is to set this path in `dataset_catalog.json` and you are good to go:

```
"small_norb_azimuth_folder": {
    "train": ["/output_path/to/snorb_azimuth/train", "<ignored>"],
    "val": ["/output_path/to/snorb_azimuth/val", "<ignored>"]
},
```

### Preparing the SmallNORB/elevation data files

Run the `create_small_norb_elevation_data_files.py` script with the `-d` option as follows:

```bash
python extra_scripts/datasets/create_small_norb_elevation_data_files.py \
    -i /path/to/snorb/ \
    -o /output_path/to/snorb_elevation
    -d
```

The folder `/output_path/to/snorb_elevation` now contains the SmallNORB/elevation `disk_folder` dataset.
The last step is to set this path in `dataset_catalog.json` and you are good to go:

```
"small_norb_elevation_folder": {
    "train": ["/output_path/to/snorb_elevation/train", "<ignored>"],
    "val": ["/output_path/to/snorb_elevation/val", "<ignored>"]
},
```

### Preparing the Stanford Cars data files

Run the `create_stanford_cars_data_files.py` script with the `-d` option as follows:

```bash
python extra_scripts/datasets/create_stanford_cars_data_files.py \
    -i /path/to/cars/ \
    -o /output_path/to/cars
    -d
```

The folder `/output_path/to/cars` now contains the Stanford Cars `disk_folder` dataset.
The last step is to set this path in `dataset_catalog.json` and you are good to go:

```
"stanford_cars_folder": {
    "train": ["/output_path/to/cars/train", "<ignored>"],
    "val": ["/output_path/to/cars/val", "<ignored>"]
},
```

### Preparing the SUN397 data files

Run the `create_sun397_data_files.py` script with the `-d` option as follows:

```bash
python extra_scripts/datasets/create_sun397_data_files.py \
    -i /path/to/sun397/ \
    -o /path/to/sun397/ \
    -d
```

The folder `/path/to/sun397/` now contains the Stanford Cars `disk_filelist` dataset.
The last step is to set this path in `dataset_catalog.json` and you are good to go:

```
"sun397_filelist": {
    "train": ["/checkpoint/qduval/datasets/sun397/train_images.npy", "/checkpoint/qduval/datasets/sun397/train_labels.npy"],
    "val": ["/checkpoint/qduval/datasets/sun397/val_images.npy", "/checkpoint/qduval/datasets/sun397/val_labels.npy"]
},
```


### Preparing UCF101/image data files

#### Automatic download

Run the `create_ucf101_data_files.py` script with the `-d` option as follows:

```bash
python extra_scripts/datasets/create_ucf101_data_files.py \
    -i /path/to/ucf101/ \
    -o /output_path/to/ucf101
    -d
```

The folder `/output_path/ucf101` now contains the UCF101 image action recognition `disk_folder` dataset.
The last step is to set this path in `dataset_catalog.json` and you are good to go:

```
"ucf101_folder": {
    "train": ["/output_path/to/ucf101/train", "<ignored>"],
    "val": ["/output_path/to/ucf101/val", "<ignored>"]
},
```

#### Manual download

Download the full dataset by visiting the [UCF101 website](https://www.crcv.ucf.edu/data/UCF101.php):

- Click on [The UCF101 data set can be downloaded by "clicking here"](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar) to retrieve the data (all the videos).
- Click on [The Train/Test Splits for Action Recognition on UCF101 data set can be downloaded by clicking here](https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip) to retrieve the splits.

Expand both archives in the same folder, say `/path/to/ucf101`.
The resulting folder should have the following structure:

```bash
ucf101/
    UCF-101/
        ApplyEyeMakeup/
            ... videos ...
        ApplyLipstick/
            ... videos ...
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
python extra_scripts/datasets/create_ucf101_data_files.py \
    -i /path/to/ucf101/ \
    -o /output_path/to/ucf101
```

The folder `/output_path/ucf101` now contains the UCF101 image action recognition dataset.

<br>

## Data preparation (optional)

The following scripts are optional as VISSL's `dataset_catalog.py` supports reading the downloaded data directly for most of these. For all the datasets below, we assume that the datasets are in the format as described [here](../vissl/data/README.md).

### Preparing COCO data files

```bash
python extra_scripts/datasets/create_coco_data_files.py \
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
python extra_scripts/datasets/create_voc_data_files.py \
    --data_source_dir /path/to/VOC2007/ \
    --output_dir /tmp/vissl/datasets/voc07/
```

- For VOC2012 dataset:

```bash
python extra_scripts/datasets/create_voc_data_files.py \
    --data_source_dir /path/to/VOC2012/ \
    --output_dir /tmp/vissl/datasets/voc12/
```


### Preparing ImageNet and Places{205, 365} data files

```bash
python extra_scripts/datasets/create_imagenet_data_files.py \
    --data_source_dir /path/to/imagenet_full_size/ \
    --output_dir /tmp/vissl/datasets/imagenet1k/
```

```bash
python extra_scripts/datasets/create_imagenet_data_files.py \
    --data_source_dir /path/to/places205/ \
    --output_dir /tmp/vissl/datasets/places205/
```

```bash
python extra_scripts/datasets/create_imagenet_data_files.py \
    --data_source_dir /path/to/places365/ \
    --output_dir /tmp/vissl/datasets/places365/
```

### Low-shot data sampling
Low-shot image classification is one of the benchmark tasks in the [paper](https://arxiv.org/abs/1905.01235). VISSL support low-shot sampling and benchmarking on the PASCAL VOC dataset only.

We train on `trainval` split of VOC2007 dataset which has 5011 images and 20 classes.
Hence the labels are of shape 5011 x 20. We generate 5 independent samples (for a given low-shot value `k`) by essentially generating 5 independent target files. For each class, we randomly
pick the positive `k` samples and `19 * k` negatives. Rest of the samples are ignored. We perform low-shot image classification on various different layers on the model (AlexNet, ResNet-50). The targets `targets_data_file` is usually obtained by extracting features for a given layer. Below command generates 5 samples for various `k` values:

```bash
python extra_scripts/datasets/create_voc_low_shot_samples.py \
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

## Converting FSDP checkpoints

VISSL allows to pretrain algorithms such as SwAV with FullyShardedDataParallel (FSDP) instead of DistributedDataParallel (DDP) to reduce the amount of GPU memory used during training and therefore train much bigger models.

FSDP models will save sharded checkpoints (one parameter file for each model shard, i.e. one parameter file for each GPU) instead of consolidated checkpoints (which contain all the weights).

While this allow faster training, it also couples the evaluation with the pre-training by forcing the benchmarking code to use exactly as many GPUs as the pre-training did. In some cases (linear evaluation), we would like to reduce the number of GPUs used during evaluation.

The `convert_sharded_checkpoint.py` script allow to transform sharded training checkpoints, containing parameters and optimizer states, into evaluation checkpoints containing only the parameter of the trunk to evaluate them, and compatible with any number of GPUs.

You can use the script like so:

```
python extra_scripts/convert_sharded_checkpoint.py \
    -i path/to/fsdp_checkpoint.torch \
    -o path/to/eval_checkpoint.torch \
    -t consolidated
```

The resulting `eval_checkpoint.torch` will be usable in DDP mode or FSDP mode, with any number of GPU.

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
