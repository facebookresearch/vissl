# SEER: Technical Model Documentation

## Table of Contents
- [Purpose](#Purpose)
- [Training Data](#Pretraining-Data)
- [Evaluation Data](#Evaluation-Data)
- [Evaluation Metrics](#Evaluation-Metrics)
- [Interpretability](#Interpretability)
- [Reproducibility](#Reproducibility)

## Purpose

We trained the model for research purposes to understand and answer how to develop the next generation of computer vision systems that are better, fairer and more flexible. SEER model itself produces image embeddings. The model can be used for detecting similar images from a given image. The model can also be adapted to perform image classification ie. given an input image, predict the objects / places / things present in the image. 

The model needs to undergo an additional step involving training on the data that the ML developer is interested in to be used for classification purposes and other research tasks for which an image embedding is necessary as a preliminary step. 


#### Architecture

[RegNetY](https://openaccess.thecvf.com/content_CVPR_2020/papers/Radosavovic_Designing_Network_Design_Spaces_CVPR_2020_paper.pdf)

#### Last Model Update

January 2022

#### Inference Type

Embeddings only. Needs to be adapted via fine tuning to make any label predictions and the model cannot make label predictions without additional finetuning for a specific task.

#### License

The use of Model is restricted to the [License](./MODEL_LICENSE.md).

#### Intended Use Case(s)

Model can be fine tuned to accomplish any computer vision tasks that require embeddings such as instance retrieval, ranking, classification, etc. 

#### Limitations and Known Issues

None. Re-iterating that the model is trained on Instagram data. The use of model is subject to the License attached.

## Pretraining Data

#### Pretraining Data Description
Model is trained with self-supervised learning to learn generic high quality visual representations using random unfiltered globally-sourced images from Instagram .
<details>
  <summary>Data Sets</summary>
  
- **Open Source Data Sets Used**: N/A
- **Private Datasets Used**:  IG images
- **Public Datasets Used**: N/A
</details>

<details>
  <summary>Issues</summary>
  
- **Representation Issues**: To have a representative dataset, we used globally-sourced images. We are evaluating fairness and robustness of the model for fairness for different gender/skintone/age, geo-diversity (recognizing a concept across the world correctly), hate speech detection and inaccurate label predictions.
- **Labeling Issues**: Not applicable as we don't use labels for training the model.
</details>

## Pretraining Evaluation Metrics

#### Performance Metrics
Log Loss, SwAV loss (see: [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/pdf/2006.09882.pdf)) during training. 

#### Last Evaluation Date
January 2022

## Evaluation Data

#### Evaluation Dataset Description
We evaluate on a number of Public datasets. See: [Vision Models are More Robust and Fair
when Pretrained in the Wild without Supervision](TODO link) which shows evaluations done on public datasets. For the fairness evaluations, we use Casual Conversations, OpenImages MIAP (containing more diverse images that represent close-to real-world scenarios) and Dollar Street dataset (contains images representing objects from all over the world). These datasets are used for inference only and no training is done on these datasets.
<details>
  <summary>Data Sets</summary>
  
- **Private Datasets Used**: N/A
- **Public Datasets Used**: ImageNet, Pascal VOC07, Places205, iNaturalist18, Oxford Flowers, SUN397, Food-101, Caltech-101, Oxford Pets, Stanford Cars, Cifar-10, Cifar-100, FGVC Aircrafts, STL-10, DTD (textures), UCF-101, Kinetics700, KITTI Distance, CLEVR count, CLEVR Distance, dSprites Orientation, dSprites Location, small Norm Elevation, EuroSat, RESISC45, Hateful Memes, MNIST, SVHN, GTSRB, PatchCamelyon, CopyDays, ImageNet-Adversarial, ImageNet-v2, ImageNet-Rendition, ImageNet-Sketch, ImageNet-Real Labels, ObjectNet, Casual Conversations, OpenImages MIAP, Dollar Street.

</details>

<details>
  <summary>Issues</summary>
  
- **Representation Issues**: To address the representation issue, we used globally-sourced images. We further evaluate fairness and robustness for the model for different gender/skintone/age, geo-diversity (recognizing a concept across the world correctly), hate speech detection and inaccurate label predictions.
- **Labeling Issues**: We used publicly released datasets directly
</details>

## Evaluation Metrics

#### Performance Metrics
Top1 accuracy, mean average precision during model evaluation on downstream tasks. For Fairness evaluations, we use the metrics for the 3 fairness indicators as described in the research: [Fairness Indicators for Systematic Assessments of Visual Feature Extractors](TODO link).

#### Last Evaluation Date
January 2022

## Interpretability
#### Feature Importance
N/A (Captum was not used for this model)

## Reproducibility

<details>
  <summary>Unbounded Hyperparameters</summary>
  
  - This model is a series of models. Please see [Self-supervised Pretraining of Visual Features in the Wild](https://arxiv.org/pdf/2103.01988.pdf) for more information.
  </details>


<details>
  <summary>Bounded Hyperparameters</summary>
  
- `batchsize`= 8192
- `learning rate` = 0.3 for `batchsize` of 256 so we scale LR for `batchsize` of 8192 to 9.6
- `lr_scheduler` = linear warmup then cosine schedule
- `num_epochs` = 1
- `weight decay` = 1e-5
- `lr_scheduler_decay` = cosine decay
- `gpu` = 512 V100_32GB
- `fp16` = We use mixed precision with O1 precision from Apex
- `SwAV multi crops` = 2x224 + 4x196
</details>

<details>
  <summary>Bounded Hyperparameters (Regnet 10B)</summary>

- `batchsize`= 7936 (16 local * 496 GPUs)
- `learning rate` = 0.3 for `batchsize` of 256 so we scale LR for `batchsize` of 7936 to 9.3
- `lr_scheduler` = linear warmup then cosine schedule
- `num_epochs` = 1
- `weight decay` = 1e-5
- `gpu` = 496 A100 40GB
- `fp16` = We use mixed precision from Pytorch
- `SwAV multi crops` = 2x160 + 4x96
</details>


#### Range of Hyperparameters
N/A

<br />

[Back To Top](#SEER-Self-supervised-model-on-images-in-the-wild)
