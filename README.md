# Solar Cell Defect Classification with Convolutional Neural Networks
## Introduction

This project focuses on classifying defects in solar cells using convolutional neural networks (CNNs). We implement various CNN architectures using PyTorch to detect two types of defects: cracks and inactive regions.


## Dataset

The dataset comprises electroluminescence images of solar modules, capturing cracks and inactive regions. Cracks vary in size, while inactive regions result from disconnections caused by cracks. The dataset preprocessing pipeline handles loading, normalization, augmentation, and class-balanced sampling.

![Left: Crack on a polycristalline module; Middle: Inactive region; Right: Cracks and
inactive regions on a monocristalline module](https://github.com/AliSajadzadeh/SolarCellClassification/blob/main/Defective%20Solar%20cells.png)

Left: Crack on a polycristalline module; Middle: Inactive region; Right: Cracks and
inactive regions on a monocristalline module

## Architectures

Two CNN architectures are implemented: ResNet18 and AlexNet. ResNet18 consists of residual blocks with skip connections, while AlexNet has been modified to use batch normalization instead of local response normalization.

## Training

The training process involves alternating between training epochs on the training set and evaluating performance on the validation set. The training continues until a stopping criterion is met.


## How to Use

 Ensure TensorFlow is installed.
 Run the provided scripts for data preprocessing, model training, and evaluation.
Adjust hyperparameters as needed.
Monitor training progress and evaluate model performance.

Refer to the code comments for detailed instructions and explanations.
