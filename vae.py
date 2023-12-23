from typing import List, Optional, Union, Tuple
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

class VAE(nn.Module):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List = None,  desired_hw: Tuple[int,int] = (64,64), 
                 kld_beta: float = 1.0, log_var_clip_val: float = torch.inf):
        
        super().__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.desired_hw = desired_hw

        # VAE outputs 64x64 by default, and we can double size by appending an extra set of hidden layers to hidden_dims 
        largest_dim_size = max(desired_hw)
        n_extra_hidden_dims = math.ceil(math.log2(largest_dim_size/64))
        self.raw_hw = (64*(2**n_extra_hidden_dims), 64*(2**n_extra_hidden_dims))

        modules = []
        if hidden_dims is None:
            # append required number of extra hidden layers to make sure output/input sizes are at least as big as desired_shape
            hidden_dims = [2**(int(math.log2(32)-(i+1))) for i in range(n_extra_hidden_dims)] + [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims

        # Build Encoder
        curr_in_channels = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(curr_in_channels, out_channels=h_dim,
                              kernel_size = 3, stride = 2, padding = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            curr_in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()  # decoder is assumed to have same 'structure' as encoder but in reverse

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=self.in_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())
        
        self._kld_reg = self.latent_dim / (self.raw_hw[0]*self.raw_hw[1])
        self.kld_beta = kld_beta

        # clips the log_var to prevent NaN loss
        self.log_var_clip_val = log_var_clip_val

    def encode(self, input: Tensor, return_padded_input: bool = False) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        assert len(input.shape) == 4, "input must have four dimensions, ordered as [N, C, H, W], where N is the batch dimension and C is the channel dimension"
        assert input.shape[-2:] == self.desired_hw, f"shape of input tensor for encoding must match desired_shape (input shape: {input.shape[-2:]}, desired_shape: {self.desired_hw})"
        assert input.shape[1] == self.in_channels, f"number of input channels must match expected number of channels (input channels: {input.shape[1]}, expected channels: {self.in_channels})"

        # pad input to match expected input shape
        w_pad = (self.raw_hw[1] - input.shape[-1]) / 2
        h_pad = (self.raw_hw[0] - input.shape[-2]) / 2
        input = F.pad(
            input, 
            (math.ceil(w_pad), math.floor(w_pad), math.ceil(h_pad), math.ceil(h_pad)),  # if input shape is not even, we account for this by having one padding side one larger
            "constant", 
            0
        )

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = torch.clip(self.fc_var(result), -self.log_var_clip_val, self.log_var_clip_val)  # to avoid NaN loss

        if return_padded_input:
            return [mu, log_var], input
        else:
            return [mu, log_var]

    def decode(self, z: Tensor, return_raw_output: bool = False) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[0], 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)

        # slice decoded tensor to match desired shape
        des_h = self.desired_hw[0]
        des_w = self.desired_hw[1]
        mid = self.raw_hw[0]//2
        sliced_result = result[
            :,
            :,
            mid-math.ceil(des_h/2):mid+math.floor(des_h/2),
            mid-math.ceil(des_w/2):mid+math.floor(des_w/2),
        ]

        if return_raw_output:
            return sliced_result, result
        else:
            return sliced_result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self, reconstruction: Tensor, input: Tensor, mu: Tensor, log_var: Tensor) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param reconstruction: (Tensor) The reconstructed input produced by this model
        :param input: (Tensor) The input that produced the reconstruction
        :param mu: (Tensor) The mean vector of the latent dimension (should be Nx1 where N is number of latent dims)
        :param log_var: (Tensor) The log variance vector of the latent dimension (as above)
        :param kwargs:
        :return:
        """

        recons_loss =F.mse_loss(reconstruction, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + self.kld_beta * self._kld_reg * kld_loss
        
        return {'loss': loss, 'recons_loss': recons_loss.detach(), 'kld_loss': -kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: Union[torch.device, str], **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]