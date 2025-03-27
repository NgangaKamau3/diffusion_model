# %% [markdown]
"""
# Comparative Analysis of Generative Models
This notebook implements comparative analysis tools for VAE, GAN, and Diffusion models.
"""

# %% [markdown]
"""
## Import Libraries and Dependencies
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import wasserstein_distance
import scipy.linalg
import os
from models import VAE, Generator, Discriminator  # Import needed classes from models.py file

# %% [markdown]
"""
## Sample Generation Functions
"""

# %%
def generate_vae_samples(vae_model, num_samples, latent_dim, device):
    """Generate samples from a VAE model"""
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = vae_model.decode(z)
    return samples

def generate_gan_samples(gan_model, num_samples, latent_dim, device):
    """Generate samples from a GAN model"""
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = gan_model(z)
    return samples

# %% [markdown]
"""
## Comparative Analysis Class
"""

# %%
class ComparativeModelAnalysis:
    def __init__(self, diffusion_model=None, vae_model=None, gan_model=None, 
                 latent_dim=100, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.diffusion_model = diffusion_model
        self.vae_model = vae_model
        self.gan_model = gan_model
        self.latent_dim = latent_dim
        self.device = device
        self.test_results = {}
    
    def generate_all_samples(self, num_samples=1000):
        """Generate samples from all available models"""
        samples = {}
        
        if self.vae_model is not None:
            vae_samples = generate_vae_samples(self.vae_model, num_samples, self.latent_dim, self.device)
            samples['vae'] = vae_samples
        
        if self.gan_model is not None:
            gan_samples = generate_gan_samples(self.gan_model, num_samples, self.latent_dim, self.device)
            samples['gan'] = gan_samples
            
        if self.diffusion_model is not None:
            diffusion_samples = self.diffusion_model.sample(num_samples)
            samples['diffusion'] = diffusion_samples
            
        return samples
    
    def compute_fid_score(self, real_features, generated_features):
        """Compute Fréchet Inception Distance"""
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
        
        # Calculate squared difference between means
        diff = mu1 - mu2
        
        # Calculate covariance sqrt term
        covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # Compute FID
        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid
    
    def compute_mmd(self, real_samples, generated_samples):
        """Compute Maximum Mean Discrepancy with RBF kernel"""
        # Flatten images
        real_flat = real_samples.reshape(real_samples.shape[0], -1)
        gen_flat = generated_samples.reshape(generated_samples.shape[0], -1)
        
        # Convert to numpy arrays
        real_np = real_flat.cpu().detach().numpy()
        gen_np = gen_flat.cpu().detach().numpy()
        
        # Compute kernel matrices
        def rbf_kernel(x, y, sigma=1.0):
            """RBF kernel between x and y"""
            n, m = x.shape[0], y.shape[0]
            dist_matrix = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    dist_matrix[i, j] = np.sum((x[i] - y[j]) ** 2)
            return np.exp(-dist_matrix / (2 * sigma ** 2))
        
        # Sample subset for computational efficiency
        n_samples = min(500, real_np.shape[0], gen_np.shape[0])
        real_subset = real_np[:n_samples]
        gen_subset = gen_np[:n_samples]
        
        # Compute kernel matrices
        K_XX = rbf_kernel(real_subset, real_subset)
        K_YY = rbf_kernel(gen_subset, gen_subset)
        K_XY = rbf_kernel(real_subset, gen_subset)
        
        # Compute MMD
        mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
        return mmd
    
    def compute_inception_score(self, generated_samples, n_splits=10):
        """Compute Inception Score for generated samples"""
        # This would typically use a pretrained Inception model
        # For simplicity, we'll use a proxy measure based on sample diversity
        
        # Reshape images to 3D tensors
        samples = generated_samples.cpu().detach()
        
        # Split dataset into n_splits
        split_size = samples.size(0) // n_splits
        scores = []
        
        for i in range(n_splits):
            split = samples[i * split_size:(i + 1) * split_size]
            
            # Compute features (simple proxy: average pooling)
            features = split.mean(dim=(2, 3))  # Spatial average
            
            # Compute entropy of the average prediction
            p_y = torch.softmax(features.mean(dim=0), dim=0)
            entropy1 = -torch.sum(p_y * torch.log(p_y + 1e-8))
            
            # Compute average entropy of individual predictions
            p_yx = torch.softmax(features, dim=1)
            entropy2 = -torch.sum(p_yx * torch.log(p_yx + 1e-8), dim=1).mean()
            
            # Compute KL divergence: entropy1 - entropy2
            kl = entropy1 - entropy2
            scores.append(torch.exp(kl).item())
        
        # Return mean and standard deviation of scores
        return np.mean(scores), np.std(scores)
    
    def compute_wasserstein_distance(self, real_samples, generated_samples):
        """Compute Wasserstein distance between real and generated distributions"""
        # Flatten images
        real_flat = real_samples.reshape(real_samples.shape[0], -1)
        gen_flat = generated_samples.reshape(generated_samples.shape[0], -1)
        
        # Convert to numpy
        real_np = real_flat.cpu().detach().numpy()
        gen_np = gen_flat.cpu().detach().numpy()
        
        # For computational efficiency, compute distance on PCA-reduced data
        pca = PCA(n_components=50)
        real_pca = pca.fit_transform(real_np)
        gen_pca = pca.transform(gen_np)
        
        # Compute average wasserstein distance across dimensions
        w_distances = []
        for i in range(real_pca.shape[1]):
            w_distances.append(wasserstein_distance(real_pca[:, i], gen_pca[:, i]))
        
        return np.mean(w_distances)
    
    def run_statistical_tests(self, real_data, generated_samples):
        """Run all statistical tests for model comparison"""
        results = {}
        
        # Compute FID for each model
        if generated_samples.get('diffusion') is not None:
            results['diffusion_fid'] = self.compute_fid_score(real_data, generated_samples['diffusion'])
        if generated_samples.get('vae') is not None:
            results['vae_fid'] = self.compute_fid_score(real_data, generated_samples['vae'])
        if generated_samples.get('gan') is not None:
            results['gan_fid'] = self.compute_fid_score(real_data, generated_samples['gan'])
        
        # Compute MMD for each model
        if generated_samples.get('diffusion') is not None:
            results['diffusion_mmd'] = self.compute_mmd(real_data, generated_samples['diffusion'])
        if generated_samples.get('vae') is not None:
            results['vae_mmd'] = self.compute_mmd(real_data, generated_samples['vae'])
        if generated_samples.get('gan') is not None:
            results['gan_mmd'] = self.compute_mmd(real_data, generated_samples['gan'])
        
        # Compute Inception Score for each model
        if generated_samples.get('diffusion') is not None:
            results['diffusion_is'] = self.compute_inception_score(generated_samples['diffusion'])
        if generated_samples.get('vae') is not None:
            results['vae_is'] = self.compute_inception_score(generated_samples['vae'])
        if generated_samples.get('gan') is not None:
            results['gan_is'] = self.compute_inception_score(generated_samples['gan'])
        
        # Compute Wasserstein distance for each model
        if generated_samples.get('diffusion') is not None:
            results['diffusion_wd'] = self.compute_wasserstein_distance(real_data, generated_samples['diffusion'])
        if generated_samples.get('vae') is not None:
            results['vae_wd'] = self.compute_wasserstein_distance(real_data, generated_samples['vae'])
        if generated_samples.get('gan') is not None:
            results['gan_wd'] = self.compute_wasserstein_distance(real_data, generated_samples['gan'])
        
        self.test_results = results
        return results
    
    def plot_comparison_results(self):
        """Plot comparative analysis results"""
        if not self.test_results:
            print("No test results available. Run statistical tests first.")
            return
            
        # Extract metrics for each model
        models = set()
        for key in self.test_results.keys():
            model = key.split('_')[0]
            models.add(model)
        models = list(models)
        
        metrics = ['fid', 'mmd', 'is', 'wd']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if metric == 'is':
                # Inception Score (higher is better)
                values = []
                for model in models:
                    key = f"{model}_{metric}"
                    if key in self.test_results:
                        values.append(self.test_results[key][0])  # Mean value
                    else:
                        values.append(0)
            else:
                # Other metrics (lower is better)
                values = []
                for model in models:
                    key = f"{model}_{metric}"
                    if key in self.test_results:
                        values.append(self.test_results[key])
                    else:
                        values.append(0)
            
            axes[i].bar(models, values)
            axes[i].set_title(f"{metric.upper()} Comparison")
            axes[i].set_xlabel("Model")
            axes[i].set_ylabel("Score")
            
            # Add value annotations
            for j, v in enumerate(values):
                axes[i].text(j, v, f"{v:.4f}", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.close()
        
    def visualize_latent_spaces(self, real_data, generated_samples):
        """Visualize and compare the latent spaces of different models"""
        # Extract features using PCA
        flatten_data = lambda x: x.reshape(x.size(0), -1).cpu().detach().numpy()
        
        real_flat = flatten_data(real_data)
        
        # Apply PCA
        pca = PCA(n_components=50)
        real_pca = pca.fit_transform(real_flat)
        
        # Apply t-SNE to reduce to 2D for visualization
        tsne = TSNE(n_components=2, random_state=42)
        real_tsne = tsne.fit_transform(real_pca)
        
        # Process each model's samples
        model_tsne = {}
        for model, samples in generated_samples.items():
            flat = flatten_data(samples)
            pca_result = pca.transform(flat)
            tsne_result = tsne.fit_transform(pca_result)
            model_tsne[model] = tsne_result
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Plot real data
        plt.scatter(real_tsne[:, 0], real_tsne[:, 1], c='black', marker='x', alpha=0.7, label='Real Data')
        
        # Plot generated samples
        colors = ['blue', 'red', 'green']
        for i, (model, coords) in enumerate(model_tsne.items()):
            plt.scatter(coords[:, 0], coords[:, 1], c=colors[i], alpha=0.5, label=f'{model.capitalize()} Generated')
        
        plt.legend()
        plt.title('t-SNE Visualization of Real and Generated Data Distributions')
        plt.tight_layout()
        plt.savefig('latent_space_comparison.png')
        plt.close()
        
    def create_visual_grid_comparison(self, real_data, generated_samples, num_samples=25):
        """Create a visual grid comparing samples from different models"""
        # Select a subset of samples
        real_subset = real_data[:num_samples]
        
        # Create a figure with subplots
        fig, axes = plt.subplots(len(generated_samples) + 1, num_samples, 
                               figsize=(2*num_samples, 2*(len(generated_samples) + 1)))
        
        # Plot real data in first row
        for i in range(num_samples):
            img = real_subset[i].cpu().detach()
            if img.shape[0] == 1:  # Grayscale
                axes[0, i].imshow(img[0], cmap='gray')
            else:  # RGB
                img = img.permute(1, 2, 0)  # CHW -> HWC
                axes[0, i].imshow(img)
            axes[0, i].axis('off')
        axes[0, 0].set_ylabel('Real', rotation=0, labelpad=40)
        
        # Plot generated samples in subsequent rows
        for j, (model, samples) in enumerate(generated_samples.items(), 1):
            model_subset = samples[:num_samples]
            for i in range(num_samples):
                img = model_subset[i].cpu().detach()
                if img.shape[0] == 1:  # Grayscale
                    axes[j, i].imshow(img[0], cmap='gray')
                else:  # RGB
                    img = img.permute(1, 2, 0)  # CHW -> HWC
                    axes[j, i].imshow(img)
                axes[j, i].axis('off')
            axes[j, 0].set_ylabel(model.capitalize(), rotation=0, labelpad=40)
        
        plt.tight_layout()
        plt.savefig('visual_sample_comparison.png')
        plt.close()

# %% [markdown]
"""
## Training Functions
"""

# %%
def train_vae(vae, dataset_loader, optimizer, device, epochs=30, beta=0.5):
    """Train a VAE model and return training metrics"""
    train_loss_history = []
    kl_loss_history = []
    recon_loss_history = []
    
    vae.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_kl = 0
        epoch_recon = 0
        for batch_idx, data in enumerate(dataset_loader):
            if isinstance(data, (tuple, list)):
                data = data[0]
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = vae(data)
            recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + beta * kl_loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_kl += kl_loss.item()
            epoch_recon += recon_loss.item()
        
        train_loss_history.append(epoch_loss / len(dataset_loader.dataset))
        kl_loss_history.append(epoch_kl / len(dataset_loader.dataset))
        recon_loss_history.append(epoch_recon / len(dataset_loader.dataset))
        
    return train_loss_history, kl_loss_history, recon_loss_history

def train_gan(generator, discriminator, dataloader, g_optimizer, d_optimizer, device, epochs=50, latent_dim=100):
    """Train a GAN model and return training metrics"""
    d_losses = []
    g_losses = []
    diversity_scores = []
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0
        diversity_score = 0

        for batch_idx, data in enumerate(dataloader):
            if isinstance(data, (tuple, list)):
                data = data[0]
            real_data = data.to(device)
            batch_size = real_data.size(0)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            label_real = torch.ones(batch_size, 1).to(device)
            label_fake = torch.zeros(batch_size, 1).to(device)

            output_real = discriminator(real_data)
            d_loss_real = criterion(output_real, label_real)

            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_data = generator(noise)
            output_fake = discriminator(fake_data.detach())
            d_loss_fake = criterion(output_fake, label_fake)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            output_fake = discriminator(fake_data)
            g_loss = criterion(output_fake, label_real)
            g_loss.backward()
            g_optimizer.step()

            # Calculate diversity score (simple feature-based metric)
            with torch.no_grad():
                diversity_score += torch.std(fake_data.view(batch_size, -1), dim=0).mean().item()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

        d_losses.append(epoch_d_loss / len(dataloader))
        g_losses.append(epoch_g_loss / len(dataloader))
        diversity_scores.append(diversity_score / len(dataloader))

    return d_losses, g_losses, diversity_scores

# %% [markdown]
"""
## Model Limitation Analysis
"""

# %%
def evaluate_vae_complexity_bias(vae, dataset_loader, device, num_samples=1000):
    """Evaluate VAE's bias towards simple/complex patterns"""
    complexity_scores = []
    recon_errors = []
    
    with torch.no_grad():
        for batch in dataset_loader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(device)
            
            # Get reconstructions
            recon_batch, _, _ = vae(batch)
            
            # Compute reconstruction error
            error = F.mse_loss(recon_batch, batch, reduction='none').mean(dim=[1,2,3])
            
            # Compute complexity score (using gradient magnitude as proxy)
            complexity = torch.abs(batch[:, :, 1:, :] - batch[:, :, :-1, :]).mean(dim=[1,2,3])
            
            complexity_scores.extend(complexity.cpu().numpy())
            recon_errors.extend(error.cpu().numpy())
            
            if len(complexity_scores) >= num_samples:
                break
    
    return np.array(complexity_scores), np.array(recon_errors)

def demonstrate_mode_collapse(generator, discriminator, dataloader, device, latent_dim=100, num_samples=1000):
    """Analyze GAN mode collapse by measuring diversity in generated samples"""
    generator.eval()
    with torch.no_grad():
        # Generate samples
        noise = torch.randn(num_samples, latent_dim).to(device)
        fake_samples = generator(noise)
        
        # Compute pairwise distances as a diversity metric
        fake_flat = fake_samples.view(num_samples, -1)
        distances = torch.pdist(fake_flat)
        
        # Calculate diversity metrics
        diversity_score = distances.mean().item()
        std_dev = distances.std().item()
        
    return diversity_score

# %% [markdown]
"""
## Main Demonstration Function
"""

# %%
def demonstrate_model_limitations(dataset_loader, device):
    """Main function to demonstrate VAE complexity bias and GAN mode collapse"""
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # Model parameters
    input_channels = 3  # RGB
    hidden_dim = 64
    latent_dim = 100
    
    print("Initializing models...")
    # Initialize models
    vae = VAE(input_channels, hidden_dim, latent_dim).to(device)
    generator = Generator(latent_dim, hidden_dim, input_channels).to(device)
    discriminator = Discriminator(input_channels, hidden_dim).to(device)
    
    print("Training VAE...")
    # Train VAE
    vae_optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    train_loss_history, kl_loss_history, recon_loss_history = train_vae(
        vae, dataset_loader, vae_optimizer, device, epochs=30, beta=0.5)
    
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
    
    print("Training GAN...")
    # Train GAN
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_losses, g_losses, diversity_scores = train_gan(
        generator, discriminator, dataset_loader, g_optimizer, d_optimizer, device, epochs=50, latent_dim=latent_dim)
    
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
    
    print("Evaluating VAE complexity bias...")
    # Demonstrate VAE complexity bias
    complexity_scores, recon_errors = evaluate_vae_complexity_bias(vae, dataset_loader, device)
    
    # Plot complexity bias
    plt.figure(figsize=(10, 6))
    plt.scatter(complexity_scores, recon_errors, alpha=0.5)
    plt.xlabel('Complexity Score')
    plt.ylabel('Reconstruction Error')
    plt.title('VAE Complexity Bias Analysis')
    plt.savefig('results/complexity_bias.png')
    plt.close()
    
    print("Demonstrating GAN mode collapse...")
    # Demonstrate GAN mode collapse
    mode_collapse_diversity = demonstrate_mode_collapse(
        generator, discriminator, dataset_loader, device, latent_dim=latent_dim)
    
    print("Conducting comparative analysis...")
    # Generate samples from both models
    vae_samples = generate_vae_samples(vae, 500, latent_dim, device)
    gan_samples = generate_gan_samples(generator, 500, latent_dim, device)
    
    # Get real samples for comparison
    real_samples = next(iter(dataset_loader))
    real_samples = real_samples.to(device)
    
    # Comparative analysis
    generated_samples = {
        'vae': vae_samples,
        'gan': gan_samples
    }
    
    analyzer = ComparativeModelAnalysis(vae_model=vae, gan_model=generator, latent_dim=latent_dim, device=device)
    results = analyzer.run_statistical_tests(real_samples, generated_samples)
    analyzer.plot_comparison_results()
    analyzer.visualize_latent_spaces(real_samples, generated_samples)
    analyzer.create_visual_grid_comparison(real_samples, generated_samples)
    
    print("Analysis complete. Results saved in the 'results' directory.")
    return results

# %% [markdown]
"""
## Execute Analysis
To run the analysis, use:
```python
dataset_loader = # your dataset loader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = demonstrate_model_limitations(dataset_loader, device)
"""
# %%
if __name__ == "__main__":
    # Initialize dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    dataset_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = demonstrate_model_limitations(dataset_loader, device)