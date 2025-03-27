# %% [markdown]
"""
# Generative Adversarial Network (GAN) Implementation
This notebook implements a GAN with mode collapse detection and analysis.
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
from scipy.stats import entropy

# %% [markdown]
"""
## GAN Architecture
"""

# %%
class Generator(nn.Module):
    """Generator network"""
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: latent_dim
            nn.ConvTranspose2d(latent_dim, hidden_dim * 8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(True),
            # Size: (hidden_dim*8) x 4 x 4
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),
            # Size: (hidden_dim*4) x 8 x 8
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            # Size: (hidden_dim*2) x 16 x 16
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            # Size: (hidden_dim) x 32 x 32
            nn.ConvTranspose2d(hidden_dim, output_dim, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
            # Output: output_dim x 64 x 64
        )
    
    def forward(self, z):
        # Reshape input: B x latent_dim -> B x latent_dim x 1 x 1
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.main(z)

class Discriminator(nn.Module):
    """Discriminator network"""
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: input_dim x 64 x 64
            nn.Conv2d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: hidden_dim x 32 x 32
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (hidden_dim*2) x 16 x 16
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (hidden_dim*4) x 8 x 8
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (hidden_dim*8) x 4 x 4
            nn.Conv2d(hidden_dim * 8, 1, kernel_size=4, stride=1, padding=0),
            # Output: 1 x 1 x 1
        )
    
    def forward(self, img):
        return self.main(img).view(-1, 1)

# %% [markdown]
"""
## Training Functions
"""

# %%
def train_gan(generator, discriminator, dataloader, optimG, optimD, device, epochs=100, latent_dim=100):
    """Train GAN and monitor for mode collapse"""
    generator.train()
    discriminator.train()
    
    # History for tracking
    d_losses = []
    g_losses = []
    diversity_scores = []
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(64, latent_dim, device=device)
    
    # Label tensors
    real_label = 1
    fake_label = 0
    
    for epoch in range(epochs):
        for batch_idx, real_data in enumerate(dataloader):
            batch_size = real_data.size(0)
            real_data = real_data.to(device)
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with real data
            discriminator.zero_grad()
            label = torch.full((batch_size,), real_label, device=device, dtype=torch.float)
            output = discriminator(real_data).view(-1)
            errD_real = F.binary_cross_entropy_with_logits(output, label)
            errD_real.backward()
            D_x = torch.sigmoid(output).mean().item()
            
            # Train with fake data
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake.detach()).view(-1)
            errD_fake = F.binary_cross_entropy_with_logits(output, label)
            errD_fake.backward()
            D_G_z1 = torch.sigmoid(output).mean().item()
            
            errD = errD_real + errD_fake
            optimD.step()
            
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # Fake labels are real for generator cost
            output = discriminator(fake).view(-1)
            errG = F.binary_cross_entropy_with_logits(output, label)
            errG.backward()
            D_G_z2 = torch.sigmoid(output).mean().item()
            
            optimG.step()
            
            # Output training stats
            # Save losses for plotting
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, '
                      f'Loss_D: {errD.item():.4f}, Loss_G: {errG.item():.4f}, '
                      f'D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')
                d_losses.append(errD.item())
                g_losses.append(errG.item())
        
        # Calculate diversity score after each epoch to monitor mode collapse
        diversity = calculate_diversity_score(generator, latent_dim, device)
        diversity_scores.append(diversity)
        print(f'Epoch {epoch+1}/{epochs}, Diversity Score: {diversity:.4f}')
        
        # Generate and save images
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
                grid = make_grid(fake, normalize=True)
                plt.figure(figsize=(10, 10))
                plt.imshow(grid.permute(1, 2, 0))
                plt.axis('off')
                plt.savefig(f'gan_samples_epoch_{epoch+1}.png')
                plt.close()
    
    return d_losses, g_losses, diversity_scores

# %% [markdown]
"""
## Generation and Evaluation Functions
"""

# %%
def generate_gan_samples(generator, num_samples, latent_dim, device):
    """Generate samples from the trained GAN"""
    generator.eval()
    with torch.no_grad():
        # Sample from standard normal distribution
        z = torch.randn(num_samples, latent_dim, device=device)
        samples = generator(z)
    return samples

def calculate_diversity_score(generator, latent_dim, device, num_samples=1000, num_bins=50):
    """Calculate diversity score to detect mode collapse"""
    generator.eval()
    samples = []
    
    with torch.no_grad():
        for _ in range(5):  # Generate in smaller batches
            z = torch.randn(200, latent_dim, device=device)
            sample = generator(z)
            # Extract features - for simplicity, we'll use mean activation across channels
            features = sample.mean(dim=(2, 3))  # Average over spatial dimensions
            samples.append(features)
    
    # Combine all samples
    features = torch.cat(samples, dim=0)
    features = features.cpu().numpy()
    
    # Calculate distribution entropy for each feature dimension
    entropies = []
    for dim in range(features.shape[1]):
        hist, _ = np.histogram(features[:, dim], bins=num_bins, density=True)
        hist = hist + 1e-8  # Avoid log(0)
        ent = entropy(hist)
        entropies.append(ent)
    
    # Average entropy across features
    diversity_score = np.mean(entropies)
    return diversity_score

# %% [markdown]
"""
## Mode Collapse Analysis
"""

# %%
def birthday_paradox_test(generator, latent_dim, device, num_tests=100, sample_size=100):
    """Implement the birthday paradox test"""
    generator.eval()
    collision_counts = []
    
    with torch.no_grad():
        for _ in range(num_tests):
            # Generate samples
            z = torch.randn(sample_size, latent_dim, device=device)
            samples = generator(z)
            
            # Convert to feature vectors (use mean activation)
            features = samples.mean(dim=(2, 3)).cpu().numpy()
            
            # Count collisions (images that are nearly identical)
            collisions = 0
            for i in range(sample_size):
                for j in range(i + 1, sample_size):
                    # Calculate L2 distance
                    dist = np.linalg.norm(features[i] - features[j])
                    if dist < 0.1:  # Threshold for "same" image
                        collisions += 1
            
            collision_counts.append(collisions)
    
    # Expected collisions under no mode collapse
    expected_collisions = sample_size * (sample_size - 1) / (2 * (2**latent_dim))
    actual_collisions = np.mean(collision_counts)
    
    return actual_collisions, expected_collisions

def demonstrate_mode_collapse(generator, discriminator, dataloader, device, latent_dim=100):
    """Deliberately induce mode collapse"""
    generator.train()
    discriminator.train()
    
    # Use SGD optimizer with high learning rate for generator (prone to mode collapse)
    optimG = optim.SGD(generator.parameters(), lr=0.1)
    optimD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    diversity_scores = []
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(64, latent_dim, device=device)
    
    for epoch in range(20):  # Fewer epochs to demonstrate faster collapse
        for batch_idx, real_data in enumerate(dataloader):
            if batch_idx > 100:  # Limit batches per epoch
                break
                
            batch_size = real_data.size(0)
            real_data = real_data.to(device)
            
            # Train discriminator minimally
            if batch_idx % 5 == 0:  # Update D less frequently
                discriminator.zero_grad()
                # Real data
                output_real = discriminator(real_data).view(-1)
                errD_real = torch.mean(F.relu(1.0 - output_real))
                errD_real.backward()
                
                # Fake data
                noise = torch.randn(batch_size, latent_dim, device=device)
                fake = generator(noise)
                output_fake = discriminator(fake.detach()).view(-1)
                errD_fake = torch.mean(F.relu(1.0 + output_fake))
                errD_fake.backward()
                
                optimD.step()
            
            # Train generator aggressively (promotes mode collapse)
            for _ in range(3):  # Update G more frequently
                generator.zero_grad()
                noise = torch.randn(batch_size, latent_dim, device=device)
                fake = generator(noise)
                output = discriminator(fake).view(-1)
                errG = -torch.mean(output)  # Non-saturating loss
                errG.backward()
                optimG.step()
        
        # Calculate diversity score
        diversity = calculate_diversity_score(generator, latent_dim, device)
        diversity_scores.append(diversity)
        print(f'Epoch {epoch+1}/20, Diversity Score: {diversity:.4f}')
        
        # Generate and save images
        with torch.no_grad():
            fake = generator(fixed_noise).detach().cpu()
            grid = make_grid(fake, normalize=True)
            plt.figure(figsize=(10, 10))
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis('off')
            plt.savefig(f'mode_collapse_epoch_{epoch+1}.png')
            plt.close()
    
    # Plot diversity scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(diversity_scores) + 1), diversity_scores)
    plt.xlabel('Epoch')
    plt.ylabel('Diversity Score')
    plt.title('GAN Mode Collapse: Decreasing Diversity Over Time')
    plt.savefig('gan_mode_collapse.png')
    plt.close()
    
    return diversity_scores