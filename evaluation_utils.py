import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def evaluate_vae_complexity_bias(vae, dataloader, device):
    """Evaluate VAE's bias towards simple patterns."""
    complexity_scores = []
    recon_errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device) if isinstance(batch, (tuple, list)) else batch.to(device)
            recon_x, _, _ = vae(x)
            
            # Calculate reconstruction error
            recon_error = torch.nn.functional.mse_loss(recon_x, x, reduction='none').mean(dim=[1,2,3])
            
            # Calculate complexity score (using gradient magnitude as proxy)
            complexity = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean(dim=[1,2,3])
            
            complexity_scores.extend(complexity.cpu().numpy())
            recon_errors.extend(recon_error.cpu().numpy())
    
    return np.array(complexity_scores), np.array(recon_errors)

def demonstrate_mode_collapse(generator, discriminator, dataloader, device, latent_dim=100, n_samples=1000):
    """Demonstrate and measure GAN mode collapse."""
    generator.eval()
    
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)
        fake_samples = generator(z)
        
        # Calculate pairwise distances as a diversity metric
        fake_flat = fake_samples.view(n_samples, -1)
        distances = torch.pdist(fake_flat)
        diversity_score = distances.mean().item()
    
    return diversity_score

def plot_complexity_bias(complexity_scores, recon_errors):
    """Plot the relationship between image complexity and reconstruction error."""
    plt.figure(figsize=(10, 6))
    plt.scatter(complexity_scores, recon_errors, alpha=0.5)
    plt.xlabel('Image Complexity')
    plt.ylabel('Reconstruction Error')
    plt.title('VAE Complexity Bias Analysis')
    plt.savefig('results/complexity_bias.png')
    plt.close()