import torch
from utils import sinusoidal_embedding
from ccbdl.network.base import BaseNetwork
from ccbdl.utils.logging import get_logger


class ConvBlock(torch.nn.Module):
    def __init__(self, shape, in_c, out_c, activation=None):
        super(ConvBlock, self).__init__()
        self.sequence = torch.nn.Sequential(torch.nn.LayerNorm(shape),
                                            torch.nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
                                            torch.nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
                                            torch.nn.SiLU() if activation is None else activation())
    
    def forward(self, ins):
        return self.sequence(ins)


class Encoder_Decoder_Latent(BaseNetwork):
    def __init__(self, n_steps, name, time_emb_dim, in_channels, hidden_channels, 
                 activation_function, num_layers, block_layers):
        super().__init__(name)

        self.logger = get_logger()
        self.logger.info("Encoder network.")

        # Sinusoidal embedding
        self.time_embed = torch.nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)
        
        act = getattr(torch.nn, activation_function)

        # Initialize lists to store down-sampling and up-sampling blocks
        self.encoder_blocks = torch.nn.Sequential()
        self.decoder_blocks = torch.nn.Sequential()
        
        self.downs = torch.nn.Sequential()
        self.ups = torch.nn.Sequential()

        # Initialize list to store time embedding layers
        self.te_down_list = torch.nn.Sequential()
        self.te_up_list = torch.nn.Sequential()

        # Add down-sampling blocks and corresponding time embedding layers
        for i in range(num_layers):
            if i == 0:
                out_channels = hidden_channels
                te_layer = self._make_te(time_emb_dim, 1)
            else:
                te_layer = self._make_te(time_emb_dim, in_channels)
                out_channels = out_channels*2
                
            encoder_block = []
            for j in range(block_layers):
                if j == 0:
                    encoder_block.append(ConvBlock((in_channels, 8, 8), in_channels, out_channels, act))
                else:
                    encoder_block.append(ConvBlock((out_channels, 8, 8), out_channels, out_channels, act))
            encoder_block = torch.nn.Sequential(*encoder_block)
            
            down = torch.nn.Sequential(torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                                       act())
                                       
            self.encoder_blocks.append(encoder_block)
            self.downs.append(down)
            in_channels = out_channels
            
            self.te_down_list.append(te_layer)

        # Add middle block and corresponding time embedding layer
        self.te_mid = self._make_te(time_emb_dim, in_channels)
        b_mid = []
        for _ in range(block_layers):
            b_mid.append(ConvBlock((out_channels, 8, 8), out_channels, out_channels, act))
        self.b_mid = torch.nn.Sequential(*b_mid)
        
        # Add up-sampling blocks and corresponding time embedding layers
        for i in reversed(range(num_layers)):
            if i == (num_layers - 1):
                in_channels = out_channels
                
            te_layer = self._make_te(time_emb_dim, in_channels)    
            up = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels, in_channels, 3, 1, 1),
                                     act())
                        
            if i != 0:
                decoder_block = []
                for j in range(block_layers):
                    if j == 0:
                        decoder_block.append(ConvBlock((in_channels, 8, 8), in_channels, out_channels, act))
                    elif j == 1:
                        decoder_block.append(ConvBlock((out_channels, 8, 8), out_channels, out_channels//2, act))
                    else:
                        decoder_block.append(ConvBlock((out_channels//2, 8, 8), out_channels//2, out_channels//2, act))
                decoder_block = torch.nn.Sequential(*decoder_block)
                
                in_channels = out_channels//2
                out_channels = out_channels//2
            else:
                decoder_block = []
                for j in range(block_layers):
                    if j == 0:
                        decoder_block.append(ConvBlock((in_channels, 8, 8), in_channels, out_channels, act))
                    else:
                        decoder_block.append(ConvBlock((out_channels, 8, 8), out_channels, out_channels, act))
                decoder_block = torch.nn.Sequential(*decoder_block)
                
            self.decoder_blocks.append(decoder_block)
            self.ups.append(up)

            self.te_up_list.append(te_layer)

        # Add output convolution
        self.conv_out = torch.nn.Conv2d(out_channels, 1, 3, 1, 1)
    
    def _make_te(self, dim_in, dim_out):
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out),
            torch.nn.SiLU(),
            torch.nn.Linear(dim_out, dim_out)
        )

    def noise_prediction(self, ins, t):
        t = self.time_embed(t)
        n = len(ins)

        # Down-sampling path
        out = ins
        down_outputs = []
        for i, (encoder_block, te_layer) in enumerate(zip(self.encoder_blocks, self.te_down_list)):
            if i == 0:
                out = encoder_block(out + te_layer(t).reshape(n, -1, 1, 1))
            else:
                out = encoder_block(self.downs[i-1](out) + te_layer(t).reshape(n, -1, 1, 1))
            down_outputs.append(out)

        # Middle block
        out = self.b_mid(self.downs[i](out) + self.te_mid(t).reshape(n, -1, 1, 1))

        # Up-sampling path
        for i, (decoder_block, te_layer) in enumerate(zip(self.decoder_blocks, self.te_up_list)):
            out = decoder_block(self.ups[i](out) + te_layer(t).reshape(n, -1, 1, 1))

        # Output convolution
        out = self.conv_out(out)
        return out
    
    def forward(self, noisy_images, time_steps):
        return self.noise_prediction(noisy_images, time_steps)


class Conditional_Encoder_Decoder_Latent(BaseNetwork):
    def __init__(self, n_steps, name, time_emb_dim, label_emb_dim, num_classes, 
                 in_channels, hidden_channels, activation_function, num_layers, 
                 block_layers):
        super().__init__(name)

        self.logger = get_logger()
        self.logger.info("Conditional Encoder network.")

        # Time embedding
        self.time_embed = torch.nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)
        
        # Label embedding
        self.label_embed = torch.nn.Embedding(num_classes, label_emb_dim)
        
        act = getattr(torch.nn, activation_function)

        # Initialize lists to store down-sampling and up-sampling blocks
        self.encoder_blocks = torch.nn.Sequential()
        self.decoder_blocks = torch.nn.Sequential()
        
        self.downs = torch.nn.Sequential()
        self.ups = torch.nn.Sequential()

        # Initialize list to store time embedding layers
        self.te_down_list = torch.nn.Sequential()
        self.te_up_list = torch.nn.Sequential()
        
        # Initialize list to store label embedding layers
        self.lb_down_list = torch.nn.Sequential()
        self.lb_up_list = torch.nn.Sequential()

        # Add down-sampling blocks and corresponding time embedding layers
        for i in range(num_layers):
            if i == 0:
                out_channels = hidden_channels
                te_layer = self._make_te(time_emb_dim, 1)
                lb_layer = self._make_lb(label_emb_dim, 1)
            else:
                te_layer = self._make_te(time_emb_dim, in_channels)
                lb_layer = self._make_lb(label_emb_dim, in_channels)
                out_channels = out_channels*2
                
            encoder_block = []
            for j in range(block_layers):
                if j == 0:
                    encoder_block.append(ConvBlock((in_channels, 8, 8), in_channels, out_channels, act))
                else:
                    encoder_block.append(ConvBlock((out_channels, 8, 8), out_channels, out_channels, act))
            encoder_block = torch.nn.Sequential(*encoder_block)
            
            down = torch.nn.Sequential(torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                                       act())
                                       
            self.encoder_blocks.append(encoder_block)
            self.downs.append(down)
            in_channels = out_channels
            
            self.te_down_list.append(te_layer)
            self.lb_down_list.append(lb_layer)

        # Add middle block and corresponding time embedding layer
        self.te_mid = self._make_te(time_emb_dim, in_channels)
        self.lb_mid = self._make_lb(label_emb_dim, in_channels)
        b_mid = []
        for _ in range(block_layers):
            b_mid.append(ConvBlock((out_channels, 8, 8), out_channels, out_channels, act))
        self.b_mid = torch.nn.Sequential(*b_mid)
        
        # Add up-sampling blocks and corresponding time embedding layers
        for i in reversed(range(num_layers)):
            if i == (num_layers - 1):
                in_channels = out_channels
                
            te_layer = self._make_te(time_emb_dim, in_channels)
            lb_layer = self._make_lb(label_emb_dim, in_channels)
            up = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels, in_channels, 3, 1, 1),
                                     act())
                        
            if i != 0:
                decoder_block = []
                for j in range(block_layers):
                    if j == 0:
                        decoder_block.append(ConvBlock((in_channels, 8, 8), in_channels, out_channels, act))
                    elif j == 1:
                        decoder_block.append(ConvBlock((out_channels, 8, 8), out_channels, out_channels//2, act))
                    else:
                        decoder_block.append(ConvBlock((out_channels//2, 8, 8), out_channels//2, out_channels//2, act))
                decoder_block = torch.nn.Sequential(*decoder_block)
                
                in_channels = out_channels//2
                out_channels = out_channels//2
            else:
                decoder_block = []
                for j in range(block_layers):
                    if j == 0:
                        decoder_block.append(ConvBlock((in_channels, 8, 8), in_channels, out_channels, act))
                    else:
                        decoder_block.append(ConvBlock((out_channels, 8, 8), out_channels, out_channels, act))
                decoder_block = torch.nn.Sequential(*decoder_block)
                
            self.decoder_blocks.append(decoder_block)
            self.ups.append(up)

            self.te_up_list.append(te_layer)
            self.lb_up_list.append(lb_layer)

        # Add output convolution
        self.conv_out = torch.nn.Conv2d(out_channels, 1, 3, 1, 1)
    
    def _make_te(self, dim_in, dim_out):
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out),
            torch.nn.SiLU(),
            torch.nn.Linear(dim_out, dim_out)
        )
    
    def _make_lb(self, dim_in, dim_out):
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out),
            torch.nn.SiLU(),
            torch.nn.Linear(dim_out, dim_out)
        )

    def noise_prediction(self, ins, t, lb):
        t = self.time_embed(t)
        lb = self.label_embed(lb)
        n = len(ins)

        # Down-sampling path
        out = ins
        down_outputs = []
        for i, (encoder_block, te_layer, lb_layer) in enumerate(zip(self.encoder_blocks, self.te_down_list, self.lb_down_list)):
            if i == 0:
                out = encoder_block(out + te_layer(t).reshape(n, -1, 1, 1) + lb_layer(lb).reshape(n, -1, 1, 1))
            else:
                out = encoder_block(self.downs[i-1](out) + te_layer(t).reshape(n, -1, 1, 1) + lb_layer(lb).reshape(n, -1, 1, 1))
            down_outputs.append(out)

        # Middle block
        out = self.b_mid(self.downs[i](out) + self.te_mid(t).reshape(n, -1, 1, 1) + self.lb_mid(lb).reshape(n, -1, 1, 1))

        # Up-sampling path
        for i, (decoder_block, te_layer, lb_layer) in enumerate(zip(self.decoder_blocks, self.te_up_list, self.lb_up_list)):
            out = decoder_block(self.ups[i](out) + te_layer(t).reshape(n, -1, 1, 1) + lb_layer(lb).reshape(n, -1, 1, 1))

        # Output convolution
        out = self.conv_out(out)
        return out
    
    def forward(self, noisy_images, time_steps, labels):
        return self.noise_prediction(noisy_images, time_steps, labels)
