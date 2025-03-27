import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision
import os

class DiffusionTrainer:
    def __init__(self, model, n_steps, device):
        self.model = model
        self.n_steps = n_steps
        self.device = device

    def train(self, dataloader, epochs, optimizer):
        """Train the diffusion model"""
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                # Handle tuple of (data, labels) from dataset
                x = batch[0].to(self.device)  # Get just the image data
                loss = self.train_step(optimizer, x)
                epoch_losses.append(loss)
                progress_bar.set_postfix({"loss": loss})
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # Sample and save images periodically
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                os.makedirs("results", exist_ok=True)
                samples = self.sample(batch_size=16)
                grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True)
                torchvision.utils.save_image(grid, f"results/diffusion_samples_epoch_{epoch+1}.png")
        
        return losses

    def train_step(self, optimizer, x_0):
        """Single training step for diffusion model"""
        optimizer.zero_grad()
        
        # Sample timesteps uniformly
        t = torch.randint(0, self.n_steps, (x_0.shape[0],), device=self.device)
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Forward diffusion process
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict noise
        noise_pred = self.model(x_t, t.float() / self.n_steps)
        
        # Calculate loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def q_sample(self, x_0, t, noise):
        """Forward diffusion process"""
        # Calculate alpha coefficients
        alphas = self.get_alphas()
        alphas_t = alphas[t][:, None, None, None]
        
        # Add noise according to diffusion schedule
        return torch.sqrt(alphas_t) * x_0 + torch.sqrt(1 - alphas_t) * noise

    def get_alphas(self):
        """Calculate alpha coefficients for diffusion process"""
        betas = torch.linspace(1e-4, 0.02, self.n_steps, device=self.device)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return alphas_cumprod