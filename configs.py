"""Configuration settings for generative models comparison."""

import os
import torch

# Project structure
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Dataset configurations
DATASET_CONFIG = {
    'name': 'cifar10',
    'image_size': 32,
    'num_channels': 3,
    'batch_size': 64
}

# Model configurations
MODEL_CONFIG = {
    'latent_dim': 100,
    'hidden_dim': 64,
    'time_steps': 1000,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# Training configurations
TRAIN_CONFIG = {
    'epochs': 50,
    'lr': 0.0002,
    'beta': 0.5,
    'beta1': 0.5,
    'beta2': 0.999,
    'weight_decay': 0.01
}

# Evaluation configurations
EVAL_CONFIG = {
    'num_samples': 1000,
    'fid_batch_size': 64,
    'inception_splits': 10
}