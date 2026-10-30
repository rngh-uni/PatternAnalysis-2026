# COMP3710 Melanoma Detection using Siamese Network - s4612969

## Description

Using Siamese Neural Network to classify possible melanoma and benign skin conditions. The results will be compared to a U-Net trained on the same dataset.

The intended users for the model are doctors and pathologists who want to be able to detect cancer cells early. The model should be able to detect obvious melanoma cases whilst flagging borderline cases to relevant experts for manual review.



## Environment Setup

### 1. Getting the Data

- Download image data from https://www.kaggle.com/datasets/roua91/isic-melanoma-dataset-resized-to-224x224/data
- Extract the compressed contents to ./Siamese_Melanoma_robert

### 2. Installing Dependencies

- Create a python venv and install the listed dependencies in requirements.txt

### 3. Running Training

```bash
python3 train.py
```

## Dataset Overview

- Source: ISIC 2020 Challenge Dataset (224x224 Resized version)
- Link: https://www.kaggle.com/datasets/roua91/isic-melanoma-dataset-resized-to-224x224/data
- Split by patient ID rather than just by image ID to prevent contamination of test set
- Details:
    - Image dimensions: 224x224
    - Classes: 0 for benign ; 1 for malignant
    - Total images: 24453 | 19227 benign | 24453 malignant
    - CSV file: Contains image IDs, labels, and patient IDs

## Model Architecture

The model being used is a Siamese Neural Network, which uses the same weights on two different inputs to determine their similarity. This can be used for the classification of melanoma since the model should be able to differentiate the malignant tumors for benign blemishes.

The distance metric that the model outputs can be used as a confidence value for determining what samples should be declared melanoma outright, and which borderline cases should be inspected by a human.

Triplet loss will be used.

A Resnet-18 subnetwork structure was chosen since it is effective for capturing intricate details from images.

### Risks

- Siamese networks can become very large, and computationally expensive to train, especially as the amount of training images grows.


## Logs

- 26/10/2026 | 1st Training run failed
- 27/10/2026 | 2nd Training run failed
- 29/10/2026 | 3rd Training run failed

# Recognition Tasks
Various recognition tasks solved in deep learning frameworks.

Tasks may include:
* Image Segmentation
* Object detection
* Graph node classification
* Image super resolution
* Disease classification
* Generative modelling with StyleGAN and Stable Diffusion