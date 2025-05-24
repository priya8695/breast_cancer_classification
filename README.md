# Breast Mass detection and its Classification as Benign or Malignant using Deep Learning Framework

## Project Description

This project presents an automated Computer-Aided Detection (CAD) system for breast mass classification using deep learning techniques. The system is designed to assist radiologists in the early detection of breast cancer by performing three key functions: suspicious region identification, mass/no-mass detection, and benign/malignant mass classification.



## Key Features

- **Fully Convolutional Deep Hierarchical Saliency Network (FCDHSNet)** - Detects suspicious regions in mammograms with high accuracy
- **Multi-Feature Integration** - Combines 2D Discrete Wavelet Transform (DWT) spectral features with CNN spatial features for improved classification
- **Transfer Learning** - Capability to adapt pre-trained models to smaller datasets
- **End-to-End Automation** - Complete pipeline from mammogram preprocessing to final mass classification

## Performance Metrics

| Metric                      | DDSM    | INbreast |
|-----------------------------|---------|----------|
| Sensitivity (Detection)     | 0.94    | 0.96     |
| Classification Accuracy     | 98.05%  | 98.14%   |
| False Positives per Image   | 0.024   | 0.026    |

## Project Structure

### Three-Stage Processing Pipeline

1. **Suspicious Region Identification**
   - Preprocessing: 
     - Contrast enhancement using CLAHE
     - Pectoral mass removal
   - FCDHSNet Architecture:
     - Coarse detection network (modified VGG16)
     - Finer detection network with RCL blocks

2. **Mass/No-Mass Detection**
   - ROI segmentation using centroid information
   - CNN architecture:
     - Input: 4-channel (wavelet subbands + ROI)
     - 4 convolutional + 4 max-pooling layers
     - Trained with categorical cross-entropy

3. **Mass Classification**
   - Same CNN architecture as mass/no-mass detector
   - Binary classification (benign/malignant)
   - Uses augmented training data (4 rotations per image)

## Datasets

| Dataset       | Cases | Abnormal | Characteristics |
|---------------|-------|----------|-----------------|
| [DDSM](link)  | 2,620 | 1,863    | Scanned film mammograms |
| [INbreast](link) | 410 | 115      | Full-field digital mammograms |

## Results

The proposed framework demonstrates:
- **High sensitivity**: 94-96% detection rate
- **Low false positives**: <0.03 FP/image
- **Superior accuracy**: >98% classification
- **Efficient transfer learning**: Effective adaptation from DDSM to INbreast

```python
# Example usage
from cad_system import BreastCAD

model = BreastCAD(weights='ddsm_weights.h5')
results = model.predict('mammogram.dcm')
