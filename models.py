import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=64, latent_dim=100):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim*2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim*2, hidden_dim*4, 4, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim*4 * 4 * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dim*4 * 4 * 4, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, hidden_dim*4 * 4 * 4)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim*2, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, input_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def encode(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 256, 4, 4)
        x = self.decoder(x)
        return x
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

class Generator(nn.Module):
    def __init__(self, latent_dim=100, hidden_dim=64, output_channels=3):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # Initial dense layer
            nn.Linear(latent_dim, hidden_dim * 8 * 4 * 4),
            nn.ReLU(True),
            
            # Reshape to start convolutions - using proper Module now
            Reshape((-1, hidden_dim * 8, 4, 4)),
            
            # Transposed convolutions
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(hidden_dim, output_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=64):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input layer
            nn.Conv2d(input_channels, hidden_dim, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Hidden layers
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output layer
            nn.Conv2d(hidden_dim * 8, 1, 4, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x).view(-1, 1)