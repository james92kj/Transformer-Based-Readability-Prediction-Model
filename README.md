# Transformer-Based Readability Prediction Model

This repository contains an implementation of a readability prediction model using transformer architectures with custom attention mechanisms.

## Project Overview

The project implements a readability prediction model using RoBERTa as the base transformer model, enhanced with custom multi-head attention mechanisms. The model is designed to predict readability scores for text passages.

## Model Architecture

### Key Components:

1. **Base Model**: 
   - RoBERTa-base transformer
   - Token classification head modified for regression

2. **Custom Attention Mechanisms**:
   - Implementation of both single-head and multi-head attention blocks
   - Multi-head attention with configurable number of heads (default: 8)
   - Dropout for regularization (0.1)
   - Layer normalization for stability

3. **CRPTokenModel**:
   - Custom model architecture combining transformer outputs with attention
   - Token-level feature extraction
   - Attention-based feature aggregation
   - Final regression head

## Training Setup

### Configuration:
- Maximum sequence length: 300 tokens
- Batch sizes:
  - Training: 3 samples per device
  - Evaluation: 2 samples per device
- Training epochs: 2
- 5-fold cross-validation
- Optimizer: Adam with initial learning rate 0.003
- Learning rate scheduler: Cosine annealing with minimum LR 1e-5

### Model Saving Strategy:
- Implements top-k model saving based on metrics
- Saves models with timestamp and performance metrics in filename
- Maintains JSON logs of training history
- Automatically manages storage by keeping only top performing models

## Performance Metrics:
- Primary metrics:
  - Mean Squared Error (MSE) Loss
  - R² Score
  - Root Mean Square Error (RMSE)
  - Mean Absolute Deviation (MAD)
- Tracks both training and validation metrics

## Future Improvements
- Experiment with different transformer backbones
- Implement more attention mechanisms
- Add data augmentation techniques
- Explore ensemble methods
- Add inference pipeline

## Notes
- The model uses custom attention mechanisms to better capture readability features
- Implements efficient memory management for large-scale training
- Includes comprehensive logging and model saving strategies
