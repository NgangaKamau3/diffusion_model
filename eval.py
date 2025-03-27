import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance
import os
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.models import resnet18
import torchvision.transforms as transforms

class DiffusionEvaluator:
    def __init__(self, real_images, generated_images, device='cuda'):
        """
        Initialize evaluator with real and generated image batches
        
        Args:
            real_images: Tensor of shape (N, C, H, W) with pixel values normalized to [0, 1]
            generated_images: Tensor of shape (M, C, H, W) with pixel values normalized to [0, 1]
            device: Computation device
        """
        self.device = device
        self.real_images = real_images.to(device)
        self.generated_images = generated_images.to(device)
        
        # Extract high-level features using a pretrained model
        self.feature_extractor = resnet18(pretrained=True).to(device)
        # Remove the final classification layer
        self.feature_extractor.fc = torch.nn.Identity()
        self.feature_extractor.eval()
        
    def extract_features(self, images):
        """Extract high-dimensional features from images using the pretrained model"""
        with torch.no_grad():
            features = self.feature_extractor(images)
        return features
    
    def compute_fid(self):
        """
        Compute Fréchet Inception Distance between real and generated images
        
        A simplified version focusing on feature distribution comparison
        """
        # Extract features
        real_features = self.extract_features(self.real_images)
        gen_features = self.extract_features(self.generated_images)
        
        # Calculate mean and covariance
        mu_real = torch.mean(real_features, dim=0).cpu().numpy()
        sigma_real = torch.cov(real_features.T).cpu().numpy()
        
        mu_gen = torch.mean(gen_features, dim=0).cpu().numpy()
        sigma_gen = torch.cov(gen_features.T).cpu().numpy()
        
        # Calculate FID
        diff = mu_real - mu_gen
        
        # Numerical stability for covariance matrix operations
        eps = 1e-6
        sigma_real_regularized = sigma_real + np.eye(sigma_real.shape[0]) * eps
        sigma_gen_regularized = sigma_gen + np.eye(sigma_gen.shape[0]) * eps
        
        # Calculate the product of covariances
        covmean, _ = stats.linalg.sqrtm(sigma_real_regularized @ sigma_gen_regularized, disp=False)
        
        # Check if complex
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # Calculate FID
        fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
        
        return float(fid)
    
    def compute_inception_score(self):
        """
        Compute a modified inception score using feature distribution entropy
        """
        # Extract features
        gen_features = self.extract_features(self.generated_images)
        
        # Normalize features
        normalized_features = F.softmax(gen_features, dim=1)
        
        # Calculate p(y)
        py = torch.mean(normalized_features, dim=0)
        
        # Calculate KL divergence
        kl_divs = []
        for i in range(normalized_features.size(0)):
            py_i = normalized_features[i]
            kl_div = torch.sum(py_i * torch.log(py_i / py + 1e-10))
            kl_divs.append(kl_div.item())
        
        # Calculate Inception Score
        inception_score = np.exp(np.mean(kl_divs))
        
        return inception_score
    
    def compute_density_stats(self):
        """
        Compute density statistics using nearest neighbor distances
        """
        # Extract features
        real_features = self.extract_features(self.real_images).cpu().numpy()
        gen_features = self.extract_features(self.generated_images).cpu().numpy()
        
        # Compute nearest neighbors for generative quality (precision)
        nn_gen_to_real = NearestNeighbors(n_neighbors=1).fit(real_features)
        distances_gen_to_real, _ = nn_gen_to_real.kneighbors(gen_features)
        precision = np.mean(distances_gen_to_real)
        
        # Compute nearest neighbors for generative coverage (recall)
        nn_real_to_gen = NearestNeighbors(n_neighbors=1).fit(gen_features)
        distances_real_to_gen, _ = nn_real_to_gen.kneighbors(real_features)
        recall = np.mean(distances_real_to_gen)
        
        # Compute density ratio (a measure of mode coverage)
        density_ratio = precision / (recall + 1e-10)
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'density_ratio': float(density_ratio)
        }
    
    def compute_wasserstein_distance(self):
        """
        Compute Wasserstein distance between real and generated distributions
        """
        # Extract marginal distributions along color channels
        real_channels = self.real_images.detach().cpu().numpy()
        gen_channels = self.generated_images.detach().cpu().numpy()
        
        channel_distances = []
        for c in range(real_channels.shape[1]):
            # Flatten spatial dimensions
            real_dist = real_channels[:, c].flatten()
            gen_dist = gen_channels[:, c].flatten()
            
            # Compute Wasserstein distance
            w_dist = wasserstein_distance(real_dist, gen_dist)
            channel_distances.append(w_dist)
        
        # Average across channels
        avg_distance = np.mean(channel_distances)
        
        return avg_distance
    
    def compute_perceptual_metrics(self):
        """
        Compute perceptual quality metrics between real and generated images
        """
        # Compute Structural Similarity Index (SSIM)
        real_np = self.real_images.detach().cpu().numpy().transpose(0, 2, 3, 1)
        gen_np = self.generated_images.detach().cpu().numpy().transpose(0, 2, 3, 1)
        
        # Compute SSIM for each real-generated pair and take the average
        ssim_values = []
        for i in range(min(len(real_np), len(gen_np))):
            ssim_val = ssim(real_np[i], gen_np[i], multichannel=True, data_range=1.0)
            ssim_values.append(ssim_val)
        
        avg_ssim = np.mean(ssim_values)
        
        return {'ssim': float(avg_ssim)}
    
    def compute_spectral_statistics(self):
        """
        Compute spectral statistics to analyze frequency domain characteristics
        """
        real_np = self.real_images.detach().cpu().numpy()
        gen_np = self.generated_images.detach().cpu().numpy()
        
        # Compute frequency spectra
        real_spectra = np.abs(np.fft.fft2(real_np, axes=(2, 3)))
        gen_spectra = np.abs(np.fft.fft2(gen_np, axes=(2, 3)))
        
        # Average over batch and channels
        real_avg_spectrum = np.mean(real_spectra, axis=(0, 1))
        gen_avg_spectrum = np.mean(gen_spectra, axis=(0, 1))
        
        # Compute statistics on spectrum difference
        spectrum_diff = np.abs(real_avg_spectrum - gen_avg_spectrum)
        spectrum_mae = np.mean(spectrum_diff)
        spectrum_rmse = np.sqrt(np.mean(spectrum_diff**2))
        
        # Radial frequency analysis
        h, w = real_avg_spectrum.shape
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r = r.astype(int)
        
        # Compute radial power spectrum
        real_radial = np.bincount(r.flatten(), weights=real_avg_spectrum.flatten()) / np.bincount(r.flatten())
        gen_radial = np.bincount(r.flatten(), weights=gen_avg_spectrum.flatten()) / np.bincount(r.flatten())
        
        # Compute KL divergence between radial distributions
        radial_kl = np.sum(real_radial * np.log((real_radial + 1e-10) / (gen_radial + 1e-10)))
        
        return {
            'spectrum_mae': float(spectrum_mae),
            'spectrum_rmse': float(spectrum_rmse),
            'radial_kl': float(radial_kl)
        }
    
    def visualize_feature_space(self, output_path="feature_space_visualization.png"):
        """
        Visualize real and generated samples in a reduced feature space
        """
        # Extract features
        real_features = self.extract_features(self.real_images).cpu().numpy()
        gen_features = self.extract_features(self.generated_images).cpu().numpy()
        
        # Combine features for TSNE
        combined_features = np.vstack([real_features, gen_features])
        
        # Create labels for visualization
        labels = np.concatenate([
            np.zeros(len(real_features)),
            np.ones(len(gen_features))
        ])
        
        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        embedded_features = tsne.fit_transform(combined_features)
        
        # Separate points back to real and generated
        real_points = embedded_features[labels == 0]
        gen_points = embedded_features[labels == 1]
        
        # Plot the visualization
        plt.figure(figsize=(10, 8))
        plt.scatter(real_points[:, 0], real_points[:, 1], c='blue', alpha=0.5, label='Real Images')
        plt.scatter(gen_points[:, 0], gen_points[:, 1], c='red', alpha=0.5, label='Generated Images')
        plt.title('t-SNE Visualization of Feature Space')
        plt.legend()
        plt.savefig(output_path)
        plt.close()
        
        # Calculate and return overlap metrics
        real_hull = scipy.spatial.ConvexHull(real_points)
        gen_hull = scipy.spatial.ConvexHull(gen_points)
        
        # Estimate overlap using distance metrics
        knn_real = NearestNeighbors(n_neighbors=1).fit(real_points)
        knn_gen = NearestNeighbors(n_neighbors=1).fit(gen_points)
        
        avg_dist_real_to_gen = np.mean(knn_gen.kneighbors(real_points)[0])
        avg_dist_gen_to_real = np.mean(knn_real.kneighbors(gen_points)[0])
        
        return {
            'visualization_path': output_path,
            'avg_dist_real_to_gen': float(avg_dist_real_to_gen),
            'avg_dist_gen_to_real': float(avg_dist_gen_to_real)
        }
    
    def run_all_evaluations(self):
        """Run all evaluation metrics and return comprehensive results"""
        results = {
            'fid': self.compute_fid(),
            'inception_score': self.compute_inception_score(),
            'wasserstein_distance': self.compute_wasserstein_distance(),
            'perceptual_metrics': self.compute_perceptual_metrics(),
            'density_stats': self.compute_density_stats(),
            'spectral_statistics': self.compute_spectral_statistics(),
            'feature_space': self.visualize_feature_space()
        }
        
        return results

# Function to load and preprocess images for evaluation
def prepare_images_for_evaluation(real_data_path, generated_data_path, n_samples=1000, img_size=32):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    # Load real images
    real_dataset = ImageDataset(real_data_path, transform=transform)
    real_loader = DataLoader(real_dataset, batch_size=n_samples, shuffle=True)
    real_images = next(iter(real_loader))
    
    # Load generated images
    generated_dataset = ImageDataset(generated_data_path, transform=transform)
    gen_loader = DataLoader(generated_dataset, batch_size=n_samples, shuffle=True)
    generated_images = next(iter(gen_loader))
    
    return real_images, generated_images

# Main evaluation function
def evaluate_diffusion_model(diffusion_model, config, n_eval_samples=1000):
    """
    Evaluate a trained diffusion model against the original dataset
    
    Args:
        diffusion_model: Trained DiffusionModel instance
        config: Configuration object
        n_eval_samples: Number of samples to use for evaluation
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Generate samples
    print(f"Generating {n_eval_samples} samples for evaluation...")
    generated_samples = diffusion_model.sample(
        n_eval_samples, 
        config.image_size, 
        config.num_channels
    )
    generated_samples = (generated_samples + 1) / 2  # Denormalize to [0, 1]
    
    # Save generated samples
    os.makedirs("evaluation_samples", exist_ok=True)
    for i in range(min(n_eval_samples, 100)):  # Save first 100 samples
        sample = generated_samples[i].cpu().numpy().transpose(1, 2, 0)
        plt.imsave(f"evaluation_samples/sample_{i}.png", sample)
    
    # Load real dataset samples
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])
    
    dataset = ImageDataset(os.path.join(config.data_dir, "images"), transform=transform)
    dataloader = DataLoader(dataset, batch_size=n_eval_samples, shuffle=True)
    real_samples = next(iter(dataloader))
    
    # Initialize evaluator
    evaluator = DiffusionEvaluator(real_samples, generated_samples, device=config.device)
    
    # Run evaluations
    print("Running statistical evaluations...")
    evaluation_results = evaluator.run_all_evaluations()
    
    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)
    
    # Print summary
    print("\nEvaluation Results Summary:")
    print(f"FID Score: {evaluation_results['fid']:.4f}")
    print(f"Inception Score: {evaluation_results['inception_score']:.4f}")
    print(f"Wasserstein Distance: {evaluation_results['wasserstein_distance']:.4f}")
    print(f"SSIM: {evaluation_results['perceptual_metrics']['ssim']:.4f}")
    print(f"Density Ratio: {evaluation_results['density_stats']['density_ratio']:.4f}")
    
    return evaluation_results
