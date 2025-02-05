import torch
from utils import load_c_en_de, load_en_de

# Latent_Diffusion class
class Latent_Diffusion:
    def __init__(self, 
                 network,
                 device,
                 n_steps,  
                 min_beta, 
                 max_beta,
                 image_chw,
                 denoiser,
                 in_channels,
                 hidden_channels):
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.hidden_channels = hidden_channels
        self.in_channels_de = hidden_channels * 4
        self.network = network
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(
            device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)
        
        encoder_decoder = load_en_de()     
        self.encoder = encoder_decoder.encoder
        self.decoder = encoder_decoder.decoder
        
        conditional_encoder_decoder = load_c_en_de()     
        self.conditional_encoder = conditional_encoder_decoder.encoder
        self.conditional_decoder = conditional_encoder_decoder.decoder
    
    def encoder(self, ins):
        return self.encoder(ins)
    
    def decoder(self, ins):
        return self.decoder(ins)
    
    def conditional_encoder(self, ins, lbs):
        return self.conditional_encoder(ins, lbs)
    
    def conditional_decoder(self, ins, lbs):
        return self.conditional_decoder(ins, lbs)
        
    def noising_images(self, x0, t):
        # Make input image more noisy (we can directly skip to the desired step)
        # n: number of images or batch size, c: number of channels i.e. 1 in this case bcoz mnist
        # h: height of the image, # w: width of the image
        x_e = self.encoder(x0)
        n, c, h, w = x_e.shape
        a_bar = self.alpha_bars[t]

        eta_latent = torch.randn(n, c, h, w).to(self.device)

        x = a_bar.sqrt().reshape(n, 1, 1, 1) * x_e + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta_latent
        return x, eta_latent, x_e

    def noise_prediction(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        out_latent = self.network(x, t)
        out_de = self.decoder(out_latent)
        return out_de, out_latent
    
    def noising_images_labels(self, x0, t, c):
        # Make input image more noisy (we can directly skip to the desired step)
        # n: number of images or batch size, c: number of channels i.e. 1 in this case bcoz mnist
        # h: height of the image, # w: width of the image
        x_e = self.conditional_encoder(x0, c)
        n, c, h, w = x_e.shape
        a_bar = self.alpha_bars[t]

        eta_latent = torch.randn(n, c, h, w).to(self.device)

        x = a_bar.sqrt().reshape(n, 1, 1, 1) * x_e + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta_latent
        return x, eta_latent, x_e
    
    def noise_prediction_labels(self, x, t, c):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        out_latent = self.network(x, t, c)
        out_de = self.conditional_decoder(out_latent, c)
        return out_de, out_latent
