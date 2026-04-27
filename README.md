# Vision Transformers: Implementation, Visualization, and Fine-Tuning

This repository contains a deep dive into the architecture and optimization of Vision Transformers (ViT). Based on the requirements of the pdf file, it includes scratch implementations of core components and experimental comparisons of efficient fine-tuning methods on the CIFAR-10 dataset.

## Topics Covered

- **Attention Mechanisms**: Implementation of single-head and multi-head self-attention modules from scratch using PyTorch.
- **Interpretability**: Extraction of attention maps using forward hooks and visualization of CLS-to-patch attention.
- **Attention Rollout**: Implementation of the attention rollout algorithm to visualize information flow across multiple Transformer layers.
- **Positional Encodings**: From-scratch implementation and visualization of sinusoidal positional encodings.
- **Fine-Tuning Strategies**:
  - **Linear Probing**: Training only the classification head with a frozen backbone.
  - **Full Fine-Tuning**: Retraining the entire model for domain-specific convergence.
- **Performance Analysis**: Numerical comparison of trainable parameters, training time, and accuracy across different tuning methods.

## How to Use

1. **Environment**: Ensure you have `torch`, `torchvision`, `matplotlib`, and `PIL` installed.
2. **Dataset**: The notebook is configured to use the CIFAR-10 dataset.
3. **Execution**: Open `vision_transformers_cifar_10.ipynb` and run the cells.

> **Important**: The notebook is designed to be executed sequentially. Please **run the cells in the order that they appear** to ensure that all classes, functions, and variables are properly initialized before being used in training or visualization.

## Results Summary
The experiments compare the efficiency of freezing the backbone versus full fine-tuning. While full fine-tuning allows for deeper adaptation, methods like linear probing offer significant efficiency gains in terms of trainable parameters and training time.
