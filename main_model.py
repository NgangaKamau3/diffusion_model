import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import wget
import tarfile

# Configuration parameters
class Config:
    def __init__(self):
        self.data_dir = "./datasets"  # Base data directory 
        self.images_dir = os.path.join(self.data_dir, "images")  # Images subdirectory
        self.batch_size = 64
        self.image_size = 32
        self.num_channels = 3
        self.time_steps = 1000
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = 30
        self.lr = 2e-4
        self.sample_interval = 5

        # Create necessary directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs("samples", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

# Download and extract dataset
import os
import wget
import tarfile

def download_dataset(url_base="https://portal.nersc.gov/cfs/m4392/G25/", dataset_name="Dataset_Specific_Unlabelled.h5"):
    """
    Downloads the specified dataset file if not already downloaded
    """
    config = Config()
    
    # Create main dataset directory and images subdirectory
    os.makedirs(config.data_dir, exist_ok=True)
    images_dir = os.path.join(config.data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Check if images exist first
    if os.path.exists(images_dir) and len([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]) > 0:
        print(f"Images already exist in {images_dir}. Using existing dataset.")
        return

    dataset_path = os.path.join(config.data_dir, dataset_name)
    
    # Dataset file exists but hasn't been extracted
    if os.path.exists(dataset_path):
        print(f"Dataset file {dataset_name} exists. Proceeding to extract...")
        try:
            if (dataset_path.endswith('.tar.gz')):
                with tarfile.open(dataset_path, "r:gz") as tar:
                    tar.extractall(path=images_dir)
                print("Dataset extracted successfully.")
            else:
                print("Creating dummy dataset as existing file is not in the expected format...")
                create_dummy_dataset(images_dir)
        except Exception as e:
            print(f"Error extracting dataset: {e}")
            print("Creating dummy dataset instead...")
            create_dummy_dataset(images_dir)
        return

    # If we get here, need to download the dataset
    print(f"Downloading dataset from {url_base}{dataset_name}...")
    try:
        wget.download(f"{url_base}{dataset_name}", dataset_path)
        print("\nDataset downloaded successfully.")
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("Creating dummy dataset instead...")
        create_dummy_dataset(images_dir)

def create_dummy_dataset(images_dir, num_images=1000):
    """Create dummy images for testing"""
    print(f"Creating {num_images} dummy images in {images_dir}")
    for i in range(num_images):
        # Create a random RGB image
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(images_dir, f'dummy_image_{i:04d}.png'))
    print("Dummy dataset creation complete!")

# Run the function
download_dataset()

# Custom dataset class
class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and
                           f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

# U-Net model for noise prediction
class UNet(nn.Module):
    def __init__(self, in_channels=3, time_emb_dim=256):
        super(UNet, self).__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial conv
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        
        # Downsample blocks
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Middle block
        self.middle = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Upsample blocks
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Time embeddings to be added
        self.time_mlp_down1 = nn.Linear(time_emb_dim, 128)
        self.time_mlp_down2 = nn.Linear(time_emb_dim, 256)
        self.time_mlp_middle = nn.Linear(time_emb_dim, 256)
        
        # Output layer
        self.conv_out = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t):
        # Time embedding
        t = t.unsqueeze(-1).float()
        t_emb = self.time_mlp(t)
        
        # Initial conv
        x1 = self.conv_in(x)
        
        # Down blocks with time embeddings
        x2 = self.down1(x1)
        x2 = x2 + self.time_mlp_down1(t_emb).unsqueeze(-1).unsqueeze(-1)
        
        x3 = self.down2(x2)
        x3 = x3 + self.time_mlp_down2(t_emb).unsqueeze(-1).unsqueeze(-1)
        
        # Middle
        x_mid = self.middle(x3)
        x_mid = x_mid + self.time_mlp_middle(t_emb).unsqueeze(-1).unsqueeze(-1)
        
        # Up blocks with skip connections
        x = self.up1(torch.cat([x_mid, x3], dim=1))
        x = self.up2(torch.cat([x, x2], dim=1))
        
        # Output
        return self.conv_out(x)

# Diffusion model
class DiffusionModel:
    def __init__(self, config):
        self.device = config.device
        self.model = UNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.mse = nn.MSELoss()
        self.time_steps = config.time_steps
        
        # Define beta schedule and derived quantities
        self.betas = torch.linspace(config.beta_start, config.beta_end, config.time_steps).to(self.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
    
    def forward_diffusion_sample(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        # Mean + variance
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise
    
    def train_step(self, x_0):
        self.optimizer.zero_grad()
        
        # Sample t uniformly
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.time_steps, (batch_size,), device=self.device).long()
        
        # Forward diffusion
        x_t, noise = self.forward_diffusion_sample(x_0, t)
        
        # Predict noise
        noise_pred = self.model(x_t, t)
        
        # Loss
        loss = self.mse(noise_pred, noise)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def sample(self, n_samples, img_size, channels):
        self.model.eval()
        with torch.no_grad():
            # Start from pure noise
            x = torch.randn(n_samples, channels, img_size, img_size).to(self.device)
            
            # Reverse diffusion sampling
            for t in tqdm(reversed(range(self.time_steps)), desc="Sampling"):
                t_batch = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
                
                # Predict noise
                noise_pred = self.model(x, t_batch)
                
                # Extract needed parameters
                alpha = self.alphas[t]
                alpha_cumprod = self.alphas_cumprod[t]
                alpha_cumprod_prev = self.alphas_cumprod_prev[t]
                beta = self.betas[t]
                
                # Sample from p(x_{t-1} | x_t)
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                variance = beta * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)
                mean_coef1 = torch.sqrt(alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)
                mean_coef2 = torch.sqrt(alpha_cumprod_prev) * beta / (1. - alpha_cumprod)
                
                # Update x
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * noise_pred) + torch.sqrt(variance) * noise
        
        self.model.train()
        return x
    
    def save_model(self, path="diffusion_model.pth"):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path="diffusion_model.pth"):
        self.model.load_state_dict(torch.load(path))

# Training function
def train_diffusion_model(config):
    # Prepare dataset
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Create dummy dataset if images directory is empty
    if not os.path.exists(config.images_dir) or len(os.listdir(config.images_dir)) == 0:
        print("No images found. Creating dummy dataset...")
        create_dummy_dataset(config.images_dir)
    
    print(f"Loading dataset from {config.images_dir}")
    dataset = ImageDataset(config.images_dir, transform=transform)
    
    # Verify dataset size
    if len(dataset) == 0:
        raise ValueError(f"No images found in {config.images_dir}")
    
    print(f"Dataset size: {len(dataset)} images")
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Initialize model
    diffusion = DiffusionModel(config)
    
    # Training loop
    losses = []
    for epoch in range(config.epochs):
        epoch_losses = []
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch+1}/{config.epochs}")
        
        for i, images in progress_bar:
            images = images.to(config.device)
            loss = diffusion.train_step(images)
            epoch_losses.append(loss)
            
            progress_bar.set_postfix(loss=loss)
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{config.epochs}, Average Loss: {avg_loss:.4f}")
        
        # Sample and save images periodically
        if (epoch + 1) % config.sample_interval == 0:
            samples = diffusion.sample(16, config.image_size, config.num_channels)
            samples = (samples.cpu().numpy() + 1) / 2  # Denormalize
            samples = np.clip(samples, 0, 1)
            
            plt.figure(figsize=(8, 8))
            for i in range(16):
                plt.subplot(4, 4, i+1)
                plt.imshow(np.transpose(samples[i], (1, 2, 0)))
                plt.axis('off')
            
            # Save the figure
            os.makedirs("samples", exist_ok=True)
            plt.savefig(f"samples/epoch_{epoch+1}.png")
            plt.close()
            
            # Save model checkpoint
            diffusion.save_model(f"checkpoints/diffusion_epoch_{epoch+1}.pth")
    
    # Save final model
    os.makedirs("checkpoints", exist_ok=True)
    diffusion.save_model("checkpoints/diffusion_final.pth")
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, config.epochs + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss')
    plt.savefig('loss_curve.png')
    
    return diffusion

# Run training if executed directly
if __name__ == "__main__":
    # Create necessary directories
    config = Config()
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(os.path.join(config.data_dir, "images"), exist_ok=True)
    os.makedirs("samples", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Download and prepare dataset
    download_dataset()
    
    # Train the model
    diffusion = train_diffusion_model(config)   