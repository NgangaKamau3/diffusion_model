# %% [markdown]
"""
# Diffusion Model Implementation
This notebook implements a diffusion model for image generation using PyTorch.
"""

# %% [markdown]
"""
## Import Libraries
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# %% [markdown]
"""
## Model Architecture
"""

# %%
class DiffusionModel(nn.Module):
    def __init__(self, 
                 time_steps=1000,
                 img_channels=3,
                 img_size=32,
                 hidden_dims=[32, 64, 128, 256],
                 device='cuda'):
        super().__init__()
        
        self.time_steps = time_steps
        self.img_size = img_size
        self.device = device
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0])
        )
        
        # U-Net architecture
        # Encoder
        self.encoder = nn.ModuleList()
        in_channels = img_channels
        for dim in hidden_dims:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, dim, 3, padding=1),
                    nn.GroupNorm(8, dim),
                    nn.SiLU(),
                    nn.Conv2d(dim, dim, 3, padding=1),
                    nn.GroupNorm(8, dim),
                    nn.SiLU(),
                    nn.AvgPool2d(2)
                )
            )
            in_channels = dim
            
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1], 3, padding=1),
            nn.GroupNorm(8, hidden_dims[-1]),
            nn.SiLU(),
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1], 3, padding=1),
            nn.GroupNorm(8, hidden_dims[-1]),
            nn.SiLU()
        )
        
        # Decoder
        self.decoder = nn.ModuleList()
        hidden_dims.reverse()
        for i, dim in enumerate(hidden_dims[1:], 1):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i-1]*2, dim, 2, stride=2),
                    nn.GroupNorm(8, dim),
                    nn.SiLU(),
                    nn.Conv2d(dim, dim, 3, padding=1),
                    nn.GroupNorm(8, dim),
                    nn.SiLU()
                )
            )
            
        # Output layer
        self.output = nn.Conv2d(hidden_dims[-1], img_channels, 1)

# %% [markdown]
"""
## Diffusion Process
"""

# %%
class DiffusionTrainer:
    def __init__(self, model, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.beta = torch.linspace(beta_start, beta_end, model.time_steps)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
    def diffuse(self, x, t):
        """Add noise to the input according to the diffusion schedule"""
        noise = torch.randn_like(x)
        return (
            torch.sqrt(self.alpha_bar[t])[:, None, None, None] * x + 
            torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None] * noise,
            noise
        )
    
    def train_step(self, x, optimizer):
        """Single training step"""
        self.model.train()
        optimizer.zero_grad()
        
        # Sample time steps
        t = torch.randint(0, self.model.time_steps, (x.shape[0],),
                         device=self.model.device)
        
        # Add noise
        noisy_x, target_noise = self.diffuse(x, t)
        
        # Predict noise
        pred_noise = self.model(noisy_x, t)
        
        # Calculate loss
        loss = F.mse_loss(pred_noise, target_noise)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()

# %% [markdown]
"""
## Sampling Process
"""

# %%
def sample(self, batch_size=16):
    """Generate samples from noise"""
    self.model.eval()
    with torch.no_grad():
            # Start from noise
            x = torch.randn(batch_size, 3, self.model.img_size, 
                          self.model.img_size, device=self.model.device)
            
            # Reverse diffusion process
    for t in tqdm(reversed(range(self.model.time_steps)), 
                     desc='Sampling'):
            t_batch = torch.full((batch_size,), t, device=self.model.device)
            
            # Predict noise
            predicted_noise = self.model(x, t_batch)
            
            # Remove noise
            alpha = self.alpha[t]
            alpha_bar = self.alpha_bar[t]
            beta = self.beta[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            
            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * predicted_noise
            ) + torch.sqrt(beta) * noise
        
    return x

# %% [markdown]
"""
## Training Loop
"""

# %%
def train(model, dataloader, trainer, optimizer, epochs, device):
    """Training loop for the diffusion model"""
    losses = []
    
    for epoch in range(epochs):
        epoch_losses = []
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
        
        for batch in progress_bar:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(device)
            
            loss = trainer.train_step(batch, optimizer)
            epoch_losses.append(loss)
            
            progress_bar.set_postfix({'loss': sum(epoch_losses)/len(epoch_losses)})
            
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1} average loss: {avg_loss:.4f}')
        
        # Generate samples every 5 epochs
        if (epoch + 1) % 5 == 0:
            samples = trainer.sample(batch_size=4)
            plot_samples(samples, epoch+1)
    
    return losses

# %% [markdown]
"""
## Visualization Functions
"""

# %%
def plot_samples(samples, epoch=None):
    """Plot generated samples"""
    samples = samples.cpu().detach()
    samples = (samples + 1) / 2  # Denormalize
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < samples.shape[0]:
            img = samples[i].permute(1, 2, 0)
            ax.imshow(img)
            ax.axis('off')
    
    if epoch is not None:
        plt.suptitle(f'Generated Samples - Epoch {epoch}')
    plt.tight_layout()
    plt.show()
# %%

