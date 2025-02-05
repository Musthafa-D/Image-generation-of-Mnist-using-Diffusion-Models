import torch
from ccbdl.utils.logging import get_logger

# DDPM class i.e. Denoise Diffusion Probabilistic Model
class DDPM:
    def __init__(self, 
                 network,
                 device,
                 n_steps,  
                 min_beta, 
                 max_beta,
                 image_chw):
        
        self.logger = get_logger()
        self.logger.info("ddpm setup.")
        
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(
            device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)
    
    def noising_images(self, x0, t, eta=None):
        # Make input image more noisy (we can directly skip to the desired step)
        # n: number of images or batch size, c: number of channels i.e. 1 in this case bcoz mnist
        # h: height of the image, # w: width of the image
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        x = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return x

    def noise_prediction(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.network(x, t)
    
    def noise_prediction_labels(self, x, t, c):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.network(x, t, c)
