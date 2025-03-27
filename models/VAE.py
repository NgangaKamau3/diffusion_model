# %% [markdown]
"""
# Variational Autoencoder (VAE) Implementation
This notebook implements a VAE for image generation with complexity bias analysis.
"""

# %% [markdown]
"""
## Import Libraries
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# %% [markdown]
"""
## VAE Architecture
"""

# %%
class VAE(nn.Module):
    """Variational Autoencoder implementation"""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim*4, hidden_dim*8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        
        # Flattened dimension calculation
        self.flatten_dim = hidden_dim * 8 * 4 * 4  # for 64x64 input
        
        # Latent space
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_var = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim*8, hidden_dim*4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(hidden_dim*2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(hidden_dim, input_dim, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Constrain output to [0, 1]
        )
        
    def encode(self, x):
        """Encode input to latent space parameters mu and log_var"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """Decode latent vector to reconstruction"""
        z = self.decoder_input(z)
        z = z.view(z.size(0), -1, 4, 4)  # Reshape to match decoder input dimensions
        reconstruction = self.decoder(z)
        return reconstruction
    
    def forward(self, x):
        """Forward pass through the VAE"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

# %% [markdown]
"""
## Training Functions
"""

# %%
def train_vae(model, dataloader, optimizer, device, epochs=50, beta=1.0):
    """Train VAE with KL annealing to demonstrate complexity bias"""
    model.train()
    train_loss_history = []
    kl_loss_history = []
    recon_loss_history = []
    
    for epoch in range(epochs):
        train_loss = 0
        kl_loss_total = 0
        recon_loss_total = 0
        
        # KL annealing - increase beta gradually
        current_beta = min(beta, beta * (epoch + 1) / (epochs // 2))
        
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, log_var = model(data)
            
            # Reconstruction loss
            recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # Total loss with beta weighting
            loss = recon_loss + current_beta * kl_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            kl_loss_total += kl_loss.item()
            recon_loss_total += recon_loss.item()
        
        avg_loss = train_loss / len(dataloader.dataset)
        avg_kl = kl_loss_total / len(dataloader.dataset)
        avg_recon = recon_loss_total / len(dataloader.dataset)
        
        train_loss_history.append(avg_loss)
        kl_loss_history.append(avg_kl)
        recon_loss_history.append(avg_recon)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, '
              f'Recon Loss: {avg_recon:.4f}, KL Loss: {avg_kl:.4f}, Beta: {current_beta:.4f}')
    
    return train_loss_history, kl_loss_history, recon_loss_history

# %% [markdown]
"""
## Generation and Evaluation Functions
"""

# %%
def generate_vae_samples(model, num_samples, latent_dim, device):
    """Generate samples from the trained VAE"""
    model.eval()
    with torch.no_grad():
        # Sample from standard normal distribution
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decode(z)
    return samples

def evaluate_vae_complexity_bias(model, dataloader, device):
    """Evaluate VAE complexity bias"""
    model.eval()
    complexity_scores = []
    recon_errors = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            
            # Measure image complexity (using gradient magnitude as proxy)
            dx = torch.abs(data[:, :, :, 1:] - data[:, :, :, :-1]).sum(dim=(1, 2, 3))
            dy = torch.abs(data[:, :, 1:, :] - data[:, :, :-1, :]).sum(dim=(1, 2, 3))
            complexity = (dx + dy) / (data.size(2) * data.size(3))
            
            # Get reconstructions
            recon, _, _ = model(data)
            
            # Calculate per-sample reconstruction error
            recon_error = F.mse_loss(recon, data, reduction='none').sum(dim=(1, 2, 3))
            
            complexity_scores.extend(complexity.cpu().numpy())
            recon_errors.extend(recon_error.cpu().numpy())
    
    return np.array(complexity_scores), np.array(recon_errors)

# %% [markdown]
"""
## Visualization Functions
"""

# %%
def plot_complexity_bias(complexity_scores, recon_errors):
    """Plot reconstruction error vs. image complexity"""
    plt.figure(figsize=(10, 6))
    plt.scatter(complexity_scores, recon_errors, alpha=0.5)
    plt.xlabel('Image Complexity')
    plt.ylabel('Reconstruction Error')
    plt.title('VAE Complexity Bias: Reconstruction Error vs. Image Complexity')
    
    # Add trend line
    z = np.polyfit(complexity_scores, recon_errors, 1)
    p = np.poly1d(z)
    plt.plot(sorted(complexity_scores), p(sorted(complexity_scores)), 'r-')
    
    plt.tight_layout()
    plt.savefig('vae_complexity_bias.png')
    plt.close()