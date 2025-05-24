# Breast Mass Classification using Deep Learning

![Mammogram Analysis Pipeline](https://via.placeholder.com/800x400?text=Mammogram+Analysis+Pipeline) *(placeholder for system diagram)*

## Project Description

An automated Computer-Aided Detection (CAD) system for breast mass classification using deep learning techniques. The system assists radiologists in early breast cancer detection through three key functions: suspicious region identification, mass/no-mass detection, and benign/malignant classification.

## Key Features

- 🔍 **Fully Convolutional Deep Hierarchical Saliency Network (FCDHSNet)** - Detects suspicious regions with high sensitivity
- 🌈 **Multi-modal Feature Integration** - Combines 2D Discrete Wavelet Transform (DWT) spectral features with CNN spatial features
- 🔄 **Transfer Learning** - Adapts pre-trained models efficiently to smaller datasets
- 🤖 **End-to-End Automation** - Processes raw mammograms through final classification without manual intervention

## Performance Metrics

| Metric                      | DDSM    | INbreast |
|-----------------------------|---------|----------|
| Sensitivity (Detection)     | 0.94    | 0.96     |
| Classification Accuracy     | 98.05%  | 98.14%   |
| False Positives per Image   | 0.024   | 0.026    |

## Project Architecture

### Three-Stage Processing Pipeline

```mermaid
graph LR
    A[Raw Mammogram] --> B((Preprocessing))
    B --> C[Suspicious Region Detection]
    C --> D[Mass/No-Mass Classification]
    D --> E[Benign/Malignant Diagnosis]
    E --> F[Clinical Report]
