# Alzheimer's Disease Classification using ConvNeXt

**Name:** Subhitcha Senthilvel  
**Student ID:** s4919098  

---

## Problem and Algorithm Description

This project implements a **ConvNeXt-based Convolutional Neural Network (CNN)**, trained **from scratch using PyTorch**, to classify MRI brain images into two categories:
- **Alzheimer’s Disease (AD)**
- **Normal Control (NC)**

ConvNeXt is a modernized CNN architecture inspired by the design principles of **Vision Transformers (ViTs)**.  
It integrates:
- Depthwise separable convolutions  
- Residual connections  
- Layer normalization  

These features enhance representational power and training efficiency.  

Given 2D brain MRI slices, the model learns spatial and structural patterns that distinguish between normal and diseased brains.  
The implementation includes:
- Data augmentation  
- CutMix  
- Label smoothing  
- Stochastic depth  
- Cosine learning rate scheduling  

The model was **trained from scratch**, without any pretrained weights, to directly learn discriminative imaging features from raw MRI data — providing objective support for Alzheimer's diagnosis.

---

## Model Architecture

Component             | Description                                                               
---------             | -----------                                                             
**Input**             | 3×256×256 (grayscale MRI converted to RGB) 
**Stem**              | Conv2d(3 → 96, kernel=4, stride=4) → LayerNorm2d(96) 
**Stages**            | ConvNeXt-style, depths = [4, 4, 18, 4], dims = [96, 192, 384, 768] 
**Stage 1**           | 4 × ConvNeXt Blocks (dim=96) → Downsample Conv2d(96 → 192) 
**Stage 2**           | 4 × ConvNeXt Blocks (dim=192) → Downsample Conv2d(192 → 384) 
**Stage 3**           | 18 × ConvNeXt Blocks (dim=384) → Downsample Conv2d(384 → 768) 
**Stage 4**           | 4 × ConvNeXt Blocks (dim=768) 
**Head**              | Global Avg Pool → LayerNorm(768) → Linear(768 → 2) 
**Loss**              | Cross-Entropy with Label Smoothing (ε = 0.1) 
**Activation**        | GELU 
**Regularization**    | CutMix (p=0.5), Random Erasing, Stochastic Depth 
**Optimizer**         | AdamW (lr = 1e-4) 
**Scheduler**         | CosineAnnealingLR 
**Early Stopping**    | Patience = 5 
**Batch Size**        | 64 
**Epochs**            | 100 

---

## Preprocessing

- **Channeling:** Convert grayscale MRIs to 3-channel RGB  
- **Resizing:** 256×256 pixels  
- **Normalization:** Scale pixel intensities to [-1, 1]  
- **Augmentation:**
  - Random horizontal flip  
  - Random rotation (±15°)  
  - RandomAffine (translate = 10%)  
  - Color jitter (brightness, contrast, saturation, hue)  
  - Random erasing (simulate occlusions)  
  - CutMix (blend two images for regularization)  
- **Split:** 70/30 stratified split for training and validation  
- **Test Data:** Independent test set (resized + normalized, no augmentations)

---

## Data Splits

Set               | Percentage    | Description 
---               | -----------   | -----------
**Train**         | 70%           | Training subset 
**Validation**    | 30%           | From training data (stratified) 
**Test**          | 100%          | Independent test set 

---

## Results

Metric                      | Value 
------                      | ------
**Early Stopping Epoch**    | 88 
**Final Train Loss**        | 0.5111 
**Train Accuracy**          | 80.62% 
**Validation Loss**         | 0.3935 
**Validation Accuracy**     | 94.33% 
**Test Accuracy**           | 76.16% 
**ROC AUC**                 | 0.8390 
**Sensitivity (Recall)**    | 0.8952 
**Specificity**             | 0.6256 
**Precision**               | 0.7088 
**F1 Score**                | 0.7911 
**Confusion Matrix:**
[[2790 1670]
[ 476 4064]]


---

## Libraries Used

`os`, `argparse`, `typing`, `numpy`, `torch`, `torch.nn`, `torch.utils.data`,  
`torch.optim`, `torch.optim.lr_scheduler`, `torch.amp`,  
`torchvision.datasets`, `torchvision.transforms`, `torchvision.ops`,  
`sklearn.model_selection`, `sklearn.metrics`, `matplotlib.pyplot`, `PIL`, `tqdm`

---

## Example Inputs

1. `/content/drive/MyDrive/PR_PROJ/ADNI/AD_NC/test/AD/388976_96.jpeg`  
2. `/content/drive/MyDrive/PR_PROJ/ADNI/AD_NC/test/NC/1185714_93.jpeg`

---

## Example Outputs

Input       | Predicted | Confidence 
-----       | --------- | ----------
AD Image    | AD        | **89.95%** 
NC Image    | NC        | **90.82%** 

---

## Plots

### Model Output
![Model Output](https://raw.githubusercontent.com/zwubhi/PatternAnalysis-2025/topic-recognition/recognition/ADNI-ConvNeXt-Subhitcha/figures/Output.png)

### Evaluation Metrics
![Evaluation Metrics](https://raw.githubusercontent.com/zwubhi/PatternAnalysis-2025/topic-recognition/recognition/ADNI-ConvNeXt-Subhitcha/figures/Evaluation_Metrics.png)

### Confusion Matrix
![Confusion Matrix](https://raw.githubusercontent.com/zwubhi/PatternAnalysis-2025/topic-recognition/recognition/ADNI-ConvNeXt-Subhitcha/figures/Confusion_Matrix.png)

### ROC Curve
![ROC Curve](https://raw.githubusercontent.com/zwubhi/PatternAnalysis-2025/topic-recognition/recognition/ADNI-ConvNeXt-Subhitcha/figures/ROC_Curve.png)

### Loss vs Epoch
![Loss vs Epoch](https://raw.githubusercontent.com/zwubhi/PatternAnalysis-2025/topic-recognition/recognition/ADNI-ConvNeXt-Subhitcha/figures/Loss%20VS%20Epoch.png)

### Accuracy vs Epoch
![Accuracy vs Epoch](https://raw.githubusercontent.com/zwubhi/PatternAnalysis-2025/topic-recognition/recognition/ADNI-ConvNeXt-Subhitcha/figures/Accuracy%20VS%20Epoch.png)

### Sample output 1
![Sample output 1](https://raw.githubusercontent.com/zwubhi/PatternAnalysis-2025/topic-recognition/recognition/ADNI-ConvNeXt-Subhitcha/figures/Prediction_AD.png)

### Sample output 2
![Sample output 2](https://raw.githubusercontent.com/zwubhi/PatternAnalysis-2025/topic-recognition/recognition/ADNI-ConvNeXt-Subhitcha/figures/Prediction_NC.png)

---

## Note

Achieved **87.28% accuracy** using an **ImageNet-pretrained ConvNeXt model**.  
However, the **scratch-trained version (76.16%)** was preferred for originality and architecture understanding.

---

## References

- [Medium](https://medium.com)  
- [GeeksforGeeks](https://www.geeksforgeeks.org)  
- [ChatGPT](https://chat.openai.com)  
- [GitHub](https://github.com)  
- [YouTube](https://www.youtube.com)  
- [arXiv](https://arxiv.org)

---