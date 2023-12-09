import torch
from torch		import nn
from torch.nn	import functional as F
from .base 		import *
from ...types_	import *

class VAE(Base):
	def	__init__(
		self,
		latent_dim: int = 128,
		hidden_dims: List = None,
		**kwargs
	) -> None:
		super(VAE, self).__init__()

		if hidden_dims is None:
			hidden_dims = [ 32, 64, 128, 256, 512 ]

		self.latent_dim = latent_dim
		self.hidden_dims = hidden_dims

		self.encoder = self.Encoder(latent_dim, hidden_dims)
		self.decoder = self.Decoder(latent_dim, hidden_dims[::-1])
	
	def encode(self, x: Tensor, **kwargs) -> List[Tensor]:
		return self.encoder(x, **kwargs)

	def decode(self, z: Tensor, **kwargs) -> Tensor:
		return self.decoder(z, **kwargs)
		
	def	reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
		# Current distribution
		# mu = 0
		# standard deviation = sqrt(e^(mu - log_var))
		std = torch.exp(log_var / 2)

		# Normal distribution
		# epsilon ~ N(0, 1)
		eps = torch.randn_like(std)
		z = mu + std * eps

		return z

	def	forward(self, input: Tensor, **kwargs) -> List[Tensor]:
		mu, log_var = self.encoder(input, **kwargs)
		z = self.reparameterize(mu, log_var, **kwargs)
		return [ self.decoder(z, **kwargs), input, mu, log_var ]

	def loss_function(self, *args, **kwargs) -> dict:
		recons = args[0]
		input = args[1]
		mu = args[2]
		log_var = args[3]

		# Reconstruction loss with mean square error
		recons_loss = F.mse_loss(recons, input)

		# Hyperparameter for KLD loss
		kld_weight = kwargs['M_N']

		# kld loss = var + mu^2 - 1 - log(var)
		kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

		# Total loss
		loss = recons_loss + kld_weight * kld_loss

		return {
			'Loss': loss,
			'Reconstruction_Loss': recons_loss.detach(),
			'KLD': kld_loss.detach()
		}

	def sample(self, num_samples:int, device: int, **kwargs) -> Tensor:
		samples = torch.randn(num_samples, self.latent_dim)
		samples = samples.to(device)

		return samples

	def generate(self, x: Tensor, **kwargs) -> Tensor:
		return self.forward(x, **kwargs)[0]

	class Encoder(Base.Encoder):
		def __init__(
			self,
			latent_dim: int,
			hidden_dims: List,
			**kwargs
		):
			super(VAE.Encoder, self).__init__()

			self.latent_dim = latent_dim
			self.hidden_dims = hidden_dims

			self.__init_encoder()

		def __init_encoder(self) -> None:
			modules = []
			in_channels = 3
			for hidden_dim in self.hidden_dims:
				modules.append(
					nn.Sequential(
						nn.Conv2d(
							in_channels,
							out_channels=hidden_dim,
							kernel_size=3,
							stride=2,
							padding=1
						),
						nn.BatchNorm2d(hidden_dim),
						nn.LeakyReLU(),
						nn.Dropout2d(0.1)
					)
				)
				in_channels = hidden_dim

			self.conv_layer = nn.Sequential(*modules)
			self.fc_mu = nn.Linear(self.hidden_dims[-1] * 2 * 2, self.latent_dim)
			self.fc_var = nn.Linear(self.hidden_dims[-1] * 2 * 2, self.latent_dim)

		def forward(self, x: Tensor, **kwargs) -> List[Tensor]:
			result = self.conv_layer(x)
			result = torch.flatten(result, start_dim=1)
			mu = self.fc_mu(result)
			log_var = self.fc_var(result)

			return [ mu, log_var ]

	class Decoder(Base.Decoder):
		def __init__(
			self,
			latent_dim: int,
			hidden_dims: List,
			**kwargs
		):
			super(VAE.Decoder, self).__init__()

			self.latent_dim = latent_dim
			self.hidden_dims = hidden_dims

			self.__init_decoder()

		def __init_decoder(self) -> None:
			self.input = nn.Linear(self.latent_dim, self.hidden_dims[0] * 2 * 2)

			modules = []
			for i in range(len(self.hidden_dims) - 1):
				modules.append(
					nn.Sequential(
						nn.ConvTranspose2d(
							self.hidden_dims[i],
							self.hidden_dims[i+1],
							kernel_size=3,
							stride=2,
							padding=1,
							output_padding=1
						),
						nn.BatchNorm2d(self.hidden_dims[i+1]),
						nn.LeakyReLU(),
						nn.Dropout2d(0.1)
					)
				)

			self.conv_layer = nn.Sequential(*modules)

			self.output = nn.Sequential(
				nn.ConvTranspose2d(
					self.hidden_dims[-1],
					out_channels=3,
					kernel_size=3,
					stride=2,
					padding=1,
					output_padding=1
				),
				nn.BatchNorm2d(3),
				nn.Tanh()
			)

		def forward(self, z: Tensor, **kwargs) -> Tensor:
			result = self.input(z)
			result = result.view(-1, self.hidden_dims[0], 2, 2)
			result = self.conv_layer(result)
			result = self.output(result)

			return result