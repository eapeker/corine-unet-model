# CORINE Level-2 Land Cover Classification from Sentinel-2 Images Using U-Net Deep Learning Model

This repository contains a deep learning model for CORINE Land Cover Level 2 classification using an advanced U-Net architecture with spatial attention and ASPP blocks.

## Model Description

The model implements a semantic segmentation approach to classify satellite imagery into 15 CORINE Land Cover Level 2 classes. Special features include:

- Residual convolutional blocks
- Spatial attention modules
- Atrous Spatial Pyramid Pooling (ASPP)
- Combined Jaccard and Cross-Entropy loss

## Repository Structure

- `corine_level2_unet_model.py`: Main Python model implementation
- `requirements.txt`: Dependencies required to run the code
- `figures/`: Visualizations of model performance and results

## Requirements

To install the required dependencies:

pip install -r requirements.txt

## Citation

@article{peker2025corine,
  title={CORINE Level-2 Land Cover Classification from Sentinel-2 Images Using U-Net Deep Learning Model},
  author={Peker, Emin Atabey and Çalışkan, Murat and Bora, Eser and İnan, Çiğdem and Uçaner, M. Erkan},
  journal={International Journal of Environment and Geoinformatics},
  note={Under review},
  year={2025}
}

## Contact

For questions or feedback, please contact: eminatabey.peker@tarimorman.gov.tr
