import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm

# Import model implementations from previous artifacts
from models import VAE, Generator, Discriminator
from training_utils import train_vae, train_gan, ComparativeModelAnalysis
from evaluation_utils import evaluate_vae_complexity_bias, demonstrate_mode_collapse, plot_complexity_bias
from trainers import DiffusionTrainer

# Simple Diffusion Model Implementation
class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DiffusionModel, self).__init__()
        
        # Define U-Net-like architecture
        # Encoder
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim*2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim*2, hidden_dim*2, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim*2, hidden_dim*4, 3, stride=2, padding=1)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim*4),
            nn.SiLU(),
            nn.Linear(hidden_dim*4, hidden_dim*4)
        )
        
        # Middle
        self.mid_block1 = nn.Conv2d(hidden_dim*4, hidden_dim*4, 3, padding=1)
        self.mid_block2 = nn.Conv2d(hidden_dim*4, hidden_dim*4, 3, padding=1)
        
        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(hidden_dim*8, hidden_dim*2, 4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(hidden_dim*4, hidden_dim*2, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(hidden_dim*4, hidden_dim, 4, stride=2, padding=1)
        self.conv6 = nn.Conv2d(hidden_dim*2, hidden_dim, 3, padding=1)
        self.conv7 = nn.Conv2d(hidden_dim, input_dim, 3, padding=1)
        
        self.act = nn.SiLU()
        
    def forward(self, x, t):
        # Initial encoding
        h1 = self.act(self.conv1(x))
        h2 = self.act(self.conv2(h1))
        h3 = self.act(self.conv3(h2))
        h4 = self.act(self.conv4(h3))
        
        # Time embedding and addition
        temb = self.time_embed(t.view(-1, 1)).view(-1, h4.shape[1], 1, 1)
        h4 = h4 + temb
        
        # Middle blocks with residual connections
        h_mid = self.act(self.mid_block1(h4))
        h_mid = self.mid_block2(h_mid) + h4
        
        # Decoder with skip connections
        h = torch.cat([h_mid, h4], dim=1)
        h = self.act(self.up1(h))
        h = torch.cat([h, h3], dim=1)
        h = self.act(self.conv5(h))
        
        h = torch.cat([h, h2], dim=1)
        h = self.act(self.up2(h))
        h = torch.cat([h, h1], dim=1)
        h = self.act(self.conv6(h))
        
        # Output layer
        return self.conv7(h)

# Main function
def generate_gan_samples(generator, n_samples, latent_dim, device):
    """Generate samples from a trained GAN model."""
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)
        samples = generator(z)
    return samples

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)  # Normalize to [-1, 1]
    ])
    
    if args.dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif args.dataset == 'celebA':
        dataset = torchvision.datasets.CelebA(root='./data', split='train', download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Model parameters
    input_channels = 3  # RGB
    hidden_dim = 64
    latent_dim = 100
    
    # Initialize models
    print("Initializing models...")
    diffusion_model = DiffusionModel(input_channels, hidden_dim).to(device)
    vae = VAE(input_channels, hidden_dim, latent_dim).to(device)
    generator = Generator(latent_dim, hidden_dim, input_channels).to(device) 
    discriminator = Discriminator(input_channels, hidden_dim).to(device)
    
    # Initialize trainer
    diffusion_trainer = DiffusionTrainer(diffusion_model, args.timesteps, device)
    
    # Train diffusion model
    if args.train_diffusion:
        print("Training diffusion model...")
        diffusion_optimizer = optim.Adam(diffusion_model.parameters(), lr=args.lr)
        diffusion_losses = diffusion_trainer.train(dataloader, args.epochs, diffusion_optimizer)
        
        # Plot diffusion training losses
        plt.figure(figsize=(10, 6))
        plt.plot(diffusion_losses)
        plt.title('Diffusion Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('results/diffusion_training_loss.png')
        plt.close()
    
    # Train VAE
    if args.train_vae:
        print("Training VAE...")
        vae_optimizer = optim.Adam(vae.parameters(), lr=args.lr)
        train_loss_history, kl_loss_history, recon_loss_history = train_vae(
            vae, dataloader, vae_optimizer, device, epochs=args.epochs, beta=args.beta)
        
        # Plot VAE training metrics
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(train_loss_history)
        plt.title('VAE Total Loss')
        plt.subplot(3, 1, 2)
        plt.plot(recon_loss_history)
        plt.title('VAE Reconstruction Loss')
        plt.subplot(3, 1, 3)
        plt.plot(kl_loss_history)
        plt.title('VAE KL Divergence Loss')
        plt.tight_layout()
        plt.savefig('results/vae_training.png')
        plt.close()
    
    # Train GAN
    if args.train_gan:
        print("Training GAN...")
        g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        d_losses, g_losses, diversity_scores = train_gan(
            generator, discriminator, dataloader, g_optimizer, d_optimizer, device, epochs=args.epochs, latent_dim=latent_dim)
        
        # Plot GAN training metrics
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(d_losses)
        plt.title('GAN Discriminator Loss')
        plt.subplot(3, 1, 2)
        plt.plot(g_losses)
        plt.title('GAN Generator Loss')
        plt.subplot(3, 1, 3)
        plt.plot(diversity_scores)
        plt.title('GAN Diversity Score')
        plt.tight_layout()
        plt.savefig('results/gan_training.png')
        plt.close()
    
    # Comparative analysis
    if args.comparative_analysis:
        print("Conducting comparative analysis...")
        # Generate samples from all models
        diffusion_samples = diffusion_trainer.sample(500)
        
        # Generate VAE samples
        with torch.no_grad():
            z = torch.randn(500, latent_dim).to(device)
            vae_samples = vae.decode(z)
            
        gan_samples = generate_gan_samples(generator, 500, latent_dim, device)
        
        # Get real samples for comparison
        real_samples = next(iter(dataloader))
        real_samples = real_samples.to(device)
        
        # Comparative analysis
        generated_samples = {
            'diffusion': diffusion_samples,
            'vae': vae_samples,
            'gan': gan_samples
        }
        
        analyzer = ComparativeModelAnalysis(
            diffusion_model=diffusion_trainer, 
            vae_model=vae, 
            gan_model=generator, 
            latent_dim=latent_dim, 
            device=device
        )
        results = analyzer.run_statistical_tests(real_samples, generated_samples)
        analyzer.plot_comparison_results()
        analyzer.visualize_latent_spaces(real_samples, generated_samples)
        analyzer.create_visual_grid_comparison(real_samples, generated_samples)
    
    # Demonstrate model limitations
    if args.demonstrate_limitations:
        print("Demonstrating model limitations...")
        # VAE complexity bias
        complexity_scores, recon_errors = evaluate_vae_complexity_bias(vae, dataloader, device)
        plot_complexity_bias(complexity_scores, recon_errors)
        
        # GAN mode collapse
        mode_collapse_diversity = demonstrate_mode_collapse(
            generator, discriminator, dataloader, device, latent_dim=latent_dim)
    
    print("All tasks completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generative Models Comparison")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "celebA"], 
                        help="Dataset to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.5, help="Beta parameter for VAE")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train_diffusion", action="store_true", help="Train diffusion model")
    parser.add_argument("--train_vae", action="store_true", help="Train VAE")
    parser.add_argument("--train_gan", action="store_true", help="Train GAN")
    parser.add_argument("--comparative_analysis", action="store_true", help="Perform comparative analysis")
    parser.add_argument("--demonstrate_limitations", action="store_true", help="Demonstrate model limitations")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of timesteps for diffusion model")
    
    args = parser.parse_args()
    main(args)