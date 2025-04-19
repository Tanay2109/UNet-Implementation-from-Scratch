# U-Net Implementation from Scratch
This repository contains a PyTorch implementation of the U-Net architecture from scratch for semantic segmentation, inspired by the research paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Olaf Ronneberger, Philipp Fischer, and Thomas Brox. The model consists of an encoder-decoder structure with skip connections, enabling precise segmentation by combining low-level spatial information with high-level contextual details.

# 📄 File Descriptions

model.py: Defines the U-Net model architecture. Contains the building blocks of the network such as double convolution layers, downsampling (encoder), upsampling (decoder), skip connections, the bottleneck and the main function.

train.py: Handles the full training pipeline including forward/backward propagation, loss calculation, optimizer steps, checkpointing, and validation.

dataset.py: Defines a custom PyTorch Dataset class for loading image-mask pairs and applying transformations using the Albumentations library for the CarvanaDataset used in this project.

utils.py: Includes helper functions such as saving/loading model checkpoints, saving predicted segmentation masks, and computing metrics like accuracy and Dice score.

# 🧠 UNet Architecture & Applications

U-Net is a fully convolutional neural network that consists of:

- Encoder (Contracting Path): Extracts semantic features using stacked convolutional layers followed by MaxPool2d for downsampling. Each downsampling step captures higher-level features with increasing receptive fields.

- Decoder (Expanding Path): Reconstructs the spatial dimensions using TransposeConv2d (upsampling). It refines feature maps for pixel-level predictions.

- Skip Connections: At each level, feature maps from the encoder are concatenated with the corresponding decoder layer. This allows the model to preserve spatial information and combine low-level details with high-level semantics, leading to more accurate segmentation boundaries.

Applications: Biomedical image segmentation, Satellite image analysis, Autonomous driving (road segmentation), Agricultural field mapping

![image](https://github.com/user-attachments/assets/f54ac8b8-7edb-437b-a7e9-ab53e52d3b15)


# 🔧 Main Functions Used

- DoubleConv() – Two consecutive convolutional layers with BatchNorm and ReLU.

- MaxPool2d() – Downsampling operation to reduce spatial dimensions in the encoder.

- TransposeConv2d() – Upsampling operation in the decoder.

- SkipConnections[] – Feature maps passed from encoder to decoder to retain spatial information.

# 📊 Results

Dataset:

In this project, the Carvana Image Masking Challenge dataset from Kaggle is used. This dataset contains high-resolution images of cars taken from a fixed angle, intended for binary image segmentation—specifically, segmenting the car from the background.

The dataset structure is as follows:

- train/: RGB training images of cars

- train_masks/: Corresponding binary masks for training images

- val/: Validation images (subset of the training set)

- val_masks/: Ground truth masks for validation images

Each mask identifies the car by labeling car pixels as 255 (foreground) and the background as 0.
Preprocessing and data augmentation techniques are applied using Albumentations to improve model robustness and generalization capability.

Results:

After training for 5 epochs with learning_rate = 1e-4 and batch_size = 16, the following results were obtained:

- Accuracy: 98.76%

- Dice Score: 0.9713

Predicted segmentation masks are saved in the saved_images/ folder for qualitative evaluation.
