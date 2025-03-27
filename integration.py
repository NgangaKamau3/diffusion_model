import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import logging
import random
from datetime import datetime

# Import the diffusion model implementation
from main_model import DiffusionModel, UNet, Config, ImageDataset, train_diffusion_model, download_dataset
from eval import DiffusionEvaluator, evaluate_diffusion_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("diffusion_experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Diffusion Model Training and Evaluation")
    
    # Dataset arguments
    parser.add_argument("--data_url", type=str, default="http://example.com/cfs/m4392/G25",
                        help="Base URL for dataset download")
    parser.add_argument("--dataset_name", type=str, default="dataset.tar.gz",
                        help="Name of the dataset archive file")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory to store the dataset")
    
    # Model parameters
    parser.add_argument("--image_size", type=int, default=32,
                        help="Image size for model input")
    parser.add_argument("--num_channels", type=int, default=3,
                        help="Number of image channels")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--time_steps", type=int, default=1000,
                        help="Number of diffusion steps")
    parser.add_argument("--beta_start", type=float, default=1e-4,
                        help="Starting value for beta schedule")
    parser.add_argument("--beta_end", type=float, default=0.02,
                        help="Ending value for beta schedule")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--sample_interval", type=int, default=5,
                        help="Interval for generating samples during training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Evaluation parameters
    parser.add_argument("--n_eval_samples", type=int, default=1000,
                        help="Number of samples to generate for evaluation")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only run evaluation using a pre-trained model")
    parser.add_argument("--model_path", type=str, default="checkpoints/diffusion_final.pth",
                        help="Path to pre-trained model for evaluation")
    
    # Additional parameters
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save results")
    
    return parser.parse_args()

def update_config_from_args(args):
    """Update the Config object with command line arguments"""
    config = Config()
    
    # Update config attributes
    config.data_dir = args.data_dir
    config.image_size = args.image_size
    config.num_channels = args.num_channels
    config.batch_size = args.batch_size
    config.time_steps = args.time_steps
    config.beta_start = args.beta_start
    config.beta_end = args.beta_end
    config.epochs = args.epochs
    config.lr = args.lr
    config.sample_interval = args.sample_interval
    
    return config

def main():
    """Main function to orchestrate training and evaluation"""
    args = parse_arguments()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"diffusion_experiment_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Update directories in args
    args.data_dir = os.path.join(output_dir, "data")
    args.output_dir = output_dir
    
    # Create subdirectories
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "evaluation"), exist_ok=True)
    
    # Save experiment configuration
    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Update config with command line arguments
    config = update_config_from_args(args)
    
    # Download dataset
    logger.info("Downloading and preparing dataset...")
    download_dataset(url_base=args.data_url, dataset_name=args.dataset_name)
    
    if not args.eval_only:
        # Train diffusion model
        logger.info("Starting diffusion model training...")
        diffusion = train_diffusion_model(config)
        
        # Save final model
        model_path = os.path.join(output_dir, "checkpoints", "diffusion_final.pth")
        diffusion.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
    else:
        # Initialize model for evaluation only
        logger.info(f"Initializing model for evaluation from {args.model_path}")
        diffusion = DiffusionModel(config)
        diffusion.load_model(args.model_path)
    
    # Evaluate model
    logger.info("Starting model evaluation...")
    evaluation_results = evaluate_diffusion_model(
        diffusion, 
        config, 
        n_eval_samples=args.n_eval_samples
    )
    
    # Save evaluation results
    eval_results_path = os.path.join(output_dir, "evaluation", "evaluation_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(evaluation_results, f, indent=4)
    
    logger.info(f"Evaluation results saved to {eval_results_path}")
    logger.info("Experiment completed successfully!")

if __name__ == "__main__":
    main()