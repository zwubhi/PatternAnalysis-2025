## TITLE: Alzheimer's Disease Classification using ConvNeXt
## NAME: Subhitcha Senthilvel  
## STUDENT ID: s4919098

## PROBLEM AND ALGORITHM DESCRIPTION:
The project implements a ConvNeXt-based CNN, trained from scratch using PyTorch, to classify MRI brain images into two categories: Alzheimer’s Disease (AD) and Normal Control (NC). ConvNeXt is a modernized CNN architecture inspired by design principles of Vision Transformers (ViTs), with inherent depthwise separable convolutions, residual connections, and layer normalization for better efficiency and representation learning. Given 2D brain MRI slices, the model learns spatial and structural patterns that distinguish between normal and diseased brains. ConvNeXt-style CNN is implemented in PyTorch, trained from scratch. The algorithm implements modern CNN design with data augmentation, CutMix, label smoothing, stochastic depth, and cosine LR scheduling to improve generalization and stability. Without using any pretraining models, this algorithm directly learns from the raw MRI data to extract discriminative imaging features and provide accurate, objective support for the diagnosis of Alzheimer's disease.

## MODEL ARCHITECTURE:
**Input:** 3×256×256 (grayscale MRI converted to RGB)
**Stem:** Conv2d(3 → 96, kernel=4, stride=4) -> LayerNorm2d(96)
**Stages:** (ConvNeXt-style, depths = [4, 4, 18, 4], dims = [96, 192, 384, 768])
**Stage 1:** 4 × ConvNeXt Blocks (dim=96) -> Downsample Conv2d(96 → 192, kernel=2, stride=2)
**Stage 2:** 4 × ConvNeXt Blocks (dim=192) -> Downsample Conv2d(192 → 384, kernel=2, stride=2)
**Stage 3:** 18 × ConvNeXt Blocks (dim=384) -> Downsample Conv2d(384 → 768, kernel=2, stride=2)
**Stage 4:** 4 × ConvNeXt Blocks (dim=768)
**Classification Head:** Global Average Pooling -> LayerNorm(768) -> Linear(768 → 2)
**Block internals:** Depthwise 7×7 conv → LayerNorm2d → Pointwise MLP (×4 expand) with GELU → Dropout(0.2) → residual + stochastic depth
**Loss:** Ceoss-entropy with Label Smoothing (ε = 0.1)
**Activation function:** Gaussian Error Linear Unit (GELU)
**Regularization:** CutMix (p=0.5), Random Erasing, Stochastic Depth, Augmentations
**Optimizer:** AdamW (lr = 1e-4)
**LR Scheduler:** CosineAnnealingLR
**Early Stopping:** Patience = 5
**Batch size:** 64
**Epochs:** 100

## PRE-PROCESSING:
**Channeling:** All MRI images are converted to 3-channel RGB to match the ConvNeXt input format.
**Image Resizing:** Images are resized to 256 × 256 pixels for uniform input dimensions.
**Normalization:** Pixel values are normalized to scale intensities between –1 and 1.
**Data Augmentation:**
Random horizontal flip
Random rotation (±15°)
RandomAffine(translate=10%)
Color jitter (brightness, contrast, saturation, hue)
Random erasing (simulate occlusions)
CutMix (blend two images for regularization)
**Dataset Split:** A 70/30 stratified split is applied on the training data to create training and validation sets, preserving class balance.
**Test Data:** Loaded separately with only resizing and normalization (no augmentation).

## DATA SPLITS: 
Train set: 70%
Validation set: 30% (from train set)
Test set: 100% (independent test data)

## RESULTS:
Early stopping at epoch 88
Final Train Loss: 0.5111
Final Train Accuracy: 80.62%
Final Validation Loss: 0.3935
Final Validation Accuracy: 94.33%
Final Test Accuracy: 76.16%
ROC AUC: 0.8390
Confusion Matrix:
[[2790 1670]
 [ 476 4064]]
Accuracy: 0.7616
Sensitivity (Recall): 0.8952
Specificity: 0.6256
Precision: 0.7088
F1 Score: 0.7911

## LIBRARIES:
os, argparse, typing, numpy, torch, torch.nn, torch.utils.data, torch.optim, torch.optim.lr_scheduler, torch.amp, torchvision.datasets, torchvision.transforms, torchvision.ops, sklearn.model_selection, sklearn.metrics, matplotlib.pyplot, PIL, tqdm

## EXAMPLE INPUTS:
1) Enter image path to classify: /content/drive/MyDrive/PR_PROJ/ADNI/AD_NC/test/AD/388976_96.jpeg
2) Enter image path to classify: /content/drive/MyDrive/PR_PROJ/ADNI/AD_NC/test/NC/1185714_93.jpeg

## OUTPUT:
1) Predicted: AD 
Confidence: 89.95%
2) Predicted: NC
Confidence: 90.82%

## PLOTS:
https://raw.githubusercontent.com/zwubhi/PatternAnalysis-2025/topic-recognition/recognition/ADNI-ConvNeXt-Subhitcha/figures/Output.png


## NOTE:
Achieved accuracy of 87.28% with pretrained on ImageNet. Preferred this instead with 76.16% accuracy, since it was built from scratch.

## REFERENCES:
https://medium.com
https://www.geeksforgeeks.org
https://chatgpt.com
https://github.com
https://www.youtube.com
https://arxiv.org