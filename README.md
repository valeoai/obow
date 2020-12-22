# **Online Bag-of-Visual-Words Generation for Unsupervised Representation Learning**

![OBoW](./img/obow_overview.png)

This is a PyTorch implementation of the OBoW paper:   
**Title:**      "Online Bag-of-Visual-Words Generationfor Unsupervised Representation Learning"    
**Authors:**     S. Gidaris, A. Bursuc, G. Puy, N. Komodakis, M. Cord, and P. Pérez  

If you use the OBoW code or framework in your research, please consider citing:

```
@article{gidaris2020obow,
  title={Online Bag-of-Visual-Words Generation for Unsupervised Representation Learning},
  author={Gidaris, Spyros and Bursuc, Andrei and Komodakis, Nikos and Cord, Matthieu and P{\'e}rez, Patrick},
  journal={arXiv preprint arXiv:2012.xxxx},
  year={2020}
}
```

### **License**
This code is released under the MIT License (refer to the LICENSE file for details).

## **Preparation**

### **Pre-requisites**
* Python 3.7
* Pytorch >= 1.3.1 (tested with 1.3.1)
* CUDA 10.0 or higher

### **Installation**

**(1)** Clone the repo:
```bash
$ git clone https://github.com/valeoai/obow
```


**(2)** Install this repository and the dependencies using pip:
```bash
$ pip install -e ./obow
```


With this, you can edit the obow code on the fly and import function
and classes of obow in other projects as well.   

**(3)** Optional. To uninstall this package, run:
```bash
$ pip uninstall obow
```


**(4)** Create *experiment* directory:
```bash
$ cd obow
$ mkdir ./experiments
```


You can take a look at the [Dockerfile](./Dockerfile) if you are uncertain
about the steps to install this project.


## **Download our ResNet50 pre-trained model**

| Method | Epochs | Batch-size | Dataset | ImageNet linear acc. | Links to pre-trained weights |
|----------------|-------------------|---------------------|--------------------|--------------------|--------------------|
| OBoW | 200 | 256 | ImageNet | 73.8 | [entire model](https://github.com/valeoai/obow/releases/download/v0.1.0/ImageNetFull_ResNet50_OBoW_full.zip) / [only feature extractor](https://github.com/valeoai/obow/releases/download/v0.1.0/ImageNetFull_ResNet50_OBoW_full_feature_extractor.zip) |


To download our ResNet50 pre-trained model from the command line run:

```bash
# Run from the OBoW directory
$ mkdir ./experiments/ImageNetFull
$ cd ./experiments/ImageNetFull

# To download all model files
$ wget https://github.com/valeoai/obow/releases/download/v0.1.0/ImageNetFull_ResNet50_OBoW_full.zip
$ unzip ImageNetFull_ResNet50_OBoW_full.zip

# To download only the student feature extractor in torchvision-like format
$ wget https://github.com/valeoai/obow/releases/download/v0.1.0/ImageNetFull_ResNet50_OBoW_full_feature_extractor.zip
$ unzip ImageNetFull_ResNet50_OBoW_full_feature_extractor.zip

$ cd ../../
```

## **Experiments: Training and evaluating ImageNet self-supervised features.**

### **Train a ResNet50-based OBoW model (full solution) on the ImageneNet dataset.**

```bash
# Run from the obow directory
# Train the OBoW model.
$ python main_obow.py --config=ImageNetFull/ResNet50_OBoW_full --workers=32 -p=250 --dst-dir=./experiments/ --data-dir=/datasets_local/ImageNet --multiprocessing-distributed --dist-url='tcp://127.0.0.1:4444'
```   


Here with `--data-dir=/datasets_local/ImageNet` it is assumed that the ImageNet
dataset is at the location `/datasets_local/ImageNet`.
The configuration file for running the above experiment, which is specified by
the `--config` argument, is located at: `./config/ImageNetFull/ResNet50_OBoW_full.py`.
Note that all the experiment configuration files are placed in the `./config/`
directory. The data of this experiment, such as checkpoints and logs, will be
stored at `./experiments/ImageNetFull/ResNet50_OBoW_full`.   

### **Evaluate on the ImageNet linear classification protocol**

Train an ImageNet linear classification model on top of frozen features learned by student of the OBoW model.
```bash
# Run from the obow directory
# Train and evaluate a linear classifier for the 1000-way ImageNet classification task.
$ python main_linear_classification.py --config=ImageNetFull/ResNet50_OBoW_full --workers=32 -p=250 -b 1024 --wd 0.0 --lr 10.0 --epochs 100 --cos-schedule --dataset ImageNet --name "ImageNet_LinCls_b1024_wd0lr10_e100" --dst-dir=./experiments/ --data-dir=/datasets_local/ImageNet --multiprocessing-distributed --dist-url='tcp://127.0.0.1:4444'
```   


The data of this experiment, such as checkpoints and logs, will be
stored at `./experiments/ImageNetFull/ResNet50_OBoW_full/ImageNet_LinCls_b1024_wd0lr10_e100`.

### **Evaluate on the Places205 linear classification protocol**

Train an Places205 linear classification model on top of frozen features extracted from the OBoW model.
```bash
# Run from the obow directory
# Train and evaluate a linear classifier for the 205-way Places205 classification task.
$ python main_linear_classification.py --config=ImageNetFull/ResNet50_OBoW_full --dataset Places205 --batch-norm --workers=32 -p=500 -b 256 --wd 0.00001 --lr 0.01 --epochs 28 --schedule 10 20 --name "Places205_LinCls_b256_wd1e4lr0p01_e28" --dst-dir=./experiments/ --data-dir=/datasets_local/Places205 --multiprocessing-distributed --dist-url='tcp://127.0.0.1:4444'
```


The data of this experiment, such as checkpoints and logs, will be
stored at `./experiments/ImageNetFull/ResNet50_OBoW_full/Places205_LinCls_b256_wd1e4lr0p01_e28`.

### **ImageNet semi-supervised evaluation setting.**

```bash
# Run from the obow directory
# Fine-tune with 1% of ImageNet annotated images.
$ python main_semisupervised.py --config=ImageNetFull/ResNet50_OBoW_full --workers=32 -p=50  --dst-dir=./experiments/ --data-dir=/datasets_local/ImageNet --multiprocessing-distributed --dist-url='tcp://127.0.0.1:4444' --percentage 1 --lr=0.0002 --lr-head=0.5 --lr-decay=0.2 --wd=0.0 --epochs=40 --schedule 24 32 --name="semi_supervised_prc1_wd0_lr0002lrp5_e40"
# Fine-tune with 10% of ImageNet annotated images.
$ python main_semisupervised.py --config=ImageNetFull/ResNet50_OBoW_full --workers=32 -p=50  --dst-dir=./experiments/ --data-dir=/datasets_local/ImageNet --multiprocessing-distributed --dist-url='tcp://127.0.0.1:4444' --percentage 10 --lr=0.0002 --lr-head=0.5 --lr-decay=0.2 --wd=0.0 --epochs=20 --schedule 12 16 --name="semi_supervised_prc10_wd0_lr0002lrp5_e20"
```


The data of these experiments, such as checkpoints and logs, will be
stored at `./experiments/ImageNetFull/ResNet50_OBoW_full/semi_supervised_prc1_wd0_lr0002lrp5_e40` and
`./experiments/ImageNetFull/ResNet50_OBoW_full/semi_supervised_prc10_wd0_lr0002lrp5_e20`
(for the 1% and 10% settings respectively).


### **Convert to torchvision format.**

The ResNet50 model that we trained is stored in a different format than that of the torchvision ResNe50 model.
The following command converts it to the torchvision format.

```bash
$ python main_obow.py --config=ImageNetFull/ResNet50_OBoW_full --dst-dir=./experiments/ --data-dir=/datasets_local/ImageNet --multiprocessing-distributed --dist-url='tcp://127.0.0.1:4444' --convert-to-torchvision
```

### **Pascal VOC07 Classification evaluation.**

First convert from the torchvision format to the caffe2 format (see command above).
```bash
# Run from the obow directory
python utils/convert_pytorch_to_caffe2.py --pth_model ./experiments/ImageNetFull/ResNet50_OBoW_full/tochvision_resnet50_student_K8192_epoch200.pth.tar --output_model ./experiments/ImageNetFull/ResNet50_OBoW_full/caffe2_resnet50_student_K8192_epoch200_bgr.pkl --rgb2bgr True
```

For the following steps you need first to download and install [fair_self_supervision_benchmark](https://github.com/facebookresearch/fair_self_supervision_benchmark).

```bash
# Run from the fair_self_supervision_benchmark directory
$ python setup.py install
$ python -c 'import self_supervision_benchmark'
# Step 1: prepare datatset.
$ mkdir obow_ep200
$ mkdir obow_ep200/voc
$ mkdir obow_ep200/voc/voc07
$ python extra_scripts/create_voc_data_files.py --data_source_dir /datasets_local/VOC2007/ --output_dir ./obow_ep200/voc/voc07/
# Step 2: extract features from voc2007
$ mkdir obow_ep200/ssl-benchmark-output
$ mkdir obow_ep200/ssl-benchmark-output/extract_features_gap
$ mkdir obow_ep200/ssl-benchmark-output/extract_features_gap/data
# ==> Extract pool5 features from the train split.
$ python tools/extract_features.py \
    --config_file [obow directory path]/utils/configs/benchmark_tasks/image_classification/voc07/resnet50_supervised_extract_gap_features.yaml \
    --data_type train \
    --output_file_prefix trainval \
    --output_dir ./obow_ep200/ssl-benchmark-output/extract_features_gap/data \
    NUM_DEVICES 1 TEST.BATCH_SIZE 64 TRAIN.BATCH_SIZE 64 \
    TEST.PARAMS_FILE [obow directory path]/experiments/obow/ImageNetFull/ResNet50_OBoW_full/caffe2_resnet50_student_K8192_epoch200_bgr.pkl \
    TRAIN.DATA_FILE ./obow_ep200/voc/voc07/train_images.npy \
    TRAIN.LABELS_FILE ./obow_ep200/voc/voc07/train_labels.npy
# ==> Extract pool5 features from the test split.
$ python tools/extract_features.py \
    --config_file [obow directory path]/utils/configs/benchmark_tasks/image_classification/voc07/resnet50_supervised_extract_gap_features.yaml \
    --data_type test \
    --output_file_prefix test \
    --output_dir ./obow_ep200/ssl-benchmark-output/extract_features_gap/data \
    NUM_DEVICES 1 TEST.BATCH_SIZE 64 TRAIN.BATCH_SIZE 64 \
    TEST.PARAMS_FILE [obow directory path]/experiments/obow/ImageNetFull/ResNet50_OBoW_full/caffe2_resnet50_student_K8192_epoch200_bgr.pkl \
    TRAIN.DATA_FILE ./obow_ep200/voc/voc07/test_images.npy TEST.DATA_FILE ./obow_ep200/voc/voc07/test_images.npy \
    TRAIN.LABELS_FILE ./obow_ep200/voc/voc07/test_labels.npy TEST.LABELS_FILE ./obow_ep200/voc/voc07/test_labels.npy
# Step 4: Train and test linear svms.
# ==> Train linear svms.
$ mkdir obow_ep200/ssl-benchmark-output/extract_features_gap/data/voc07_svm
$ mkdir obow_ep200/ssl-benchmark-output/extract_features_gap/data/voc07_svm/svm_pool5bn
$ python tools/svm/train_svm_kfold.py \
    --data_file ./obow_ep200/ssl-benchmark-output/extract_features_gap/data/trainval_pool5_bn_features.npy \
    --targets_data_file ./obow_ep200/ssl-benchmark-output/extract_features_gap/data/trainval_pool5_bn_targets.npy \
    --costs_list "0.05,0.1,0.3,0.5,1.0,3.0,5.0" \
    --output_path ./obow_ep200/ssl-benchmark-output/extract_features_gap/data/voc07_svm/svm_pool5bn/  
# ==> Test the linear svms.
$ python tools/svm/test_svm.py \
    --data_file ./obow_ep200/ssl-benchmark-output/extract_features_gap/data/test_pool5_bn_features.npy \
    --targets_data_file ./obow_ep200/ssl-benchmark-output/extract_features_gap/data/test_pool5_bn_targets.npy \
    --costs_list "0.05,0.1,0.3,0.5,1.0,3.0,5.0" \
    --output_path ./obow_ep200/ssl-benchmark-output/extract_features_gap/data/voc07_svm/svm_pool5bn/    
```

### **Pascal VOC07+12 Object Detection evaluation.**

1. First install [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

2. Convert a pre-trained model from the torchvision format to the caffe2 format required by Detectron2 (see command above). 

3. Put dataset under "./datasets" directory, following the [directory structure](https://github.com/facebookresearch/detectron2/tree/master/datasets)
	 requried by Detectron2.
   
4. Copy the [config file](https://github.com/valeoai/obow/tree/main/utils/configs/benchmark_tasks/object_detection) in the Detectron2 repo `configs/PascalVOC-Detection`.

5. In Detectron2 launch the `train_net.py` script to reproduce the object detection experiments on Pascal VOC:

```bash
python tools/train_net.py --num-gpus 8 --config-file configs/PascalVOC-Detection/pascal_voc_0712_faster_rcnn_R_50_C4_BoWNetpp_K8192.yaml
```

## **Other experiments: Training using 20% of ImageNet and ResNet18.**

A single gpu is enough for the following experiments.

### **ResNet18-based OBoW vanilla solution.**

```bash
# Run from the obow directory
# Train the model.
$ python main_obow.py --config=ImageNet20/ResNet18_OBoW_vanilla --workers=16 --dst-dir=./experiments/ --data-dir=/datasets_local/ImageNet
# Few-shot evaluation.
$ python main_obow.py --config=ImageNet20/ResNet18_OBoW_vanilla --workers=16 --episodes 200 --fewshot-q 1 --fewshot-n 50 --fewshot-k 1 5 --evaluate --start-epoch=-1 --dst-dir=./experiments/ --data-dir=/datasets_local/ImageNet
# Linear classification evaluation. Note the following command precaches the extracted features at root/local_storage/spyros/cache/obow.
$ python main_linear_classification.py --config=ImageNet20/ResNet18_OBoW_vanilla --workers=16 -b 256 --wd 0.000002 --dataset ImageNet --name "ImageNet_LinCls_precache_b256_lr10p0wd2e6" --precache --lr 10.0 --epochs 50 --schedule 15 30 45 --subset=260 --dst-dir=./experiments/ --data-dir=/datasets_local/ImageNet --cache-dir=/root/local_storage/spyros/cache/obow
```

### **ResNet18-based OBoW full solution.**

```bash
# Run from the obow directory
# Train the model.
$ python main_obow.py --config=ImageNet20/ResNet18_OBoW_full --workers=16 --dst-dir=./experiments/ --data-dir=/datasets_local/ImageNet
# Few-shot evaluation.
$ python main_obow.py --config=ImageNet20/ResNet18_OBoW_full --workers=16 --episodes 200 --fewshot-q 1 --fewshot-n 50 --fewshot-k 1 5 --evaluate --start-epoch=-1 --dst-dir=./experiments/ --data-dir=/datasets_local/ImageNet
# Linear classification evaluation. Note the following command precaches the extracted features at root/local_storage/spyros/cache/obow.
$ python main_linear_classification.py --config=ImageNet20/ResNet18_OBoW_full --workers=16 -b 256 --wd 0.000002 --dataset ImageNet --name "ImageNet_LinCls_precache_b256_lr10p0wd2e6" --precache --lr 10.0 --epochs 50 --schedule 15 30 45 --subset=260 --dst-dir=./experiments/ --data-dir=/datasets_local/ImageNet --cache-dir=/root/local_storage/spyros/cache/obow
```

### **Download the ResNet18-based OBoW models pre-trained on 20% of ImageNet.**

```bash
# Run from the OBoW directory
$ mkdir ./experiments/ImageNet20
$ cd ./experiments/ImageNet20

# To download the full OBoW version
$ wget https://github.com/valeoai/obow/releases/download/v0.1.0/ImageNet20_ResNet18_OBoW_full.zip
$ unzip ImageNet20_ResNet18_OBoW_full.zip

# To download the vanilla OBoW version
$ wget https://github.com/valeoai/obow/releases/download/v0.1.0/ImageNet20_ResNet18_OBoW_vanilla.zip
$ unzip ImageNet20_ResNet18_OBoW_vanilla.zip

$ cd ../../
```
