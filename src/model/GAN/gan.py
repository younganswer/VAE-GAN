# Import user-defined packages
from .base 		import Base
from ...types_	import *

import torch
from torch		import nn
from torch.nn	import functional as F

class GAN(Base):
	def __init__(
		self,
		latent_dim: int = 128,
		hidden_dims: List = None,
		**kwargs
	):
		super(GAN, self).__init__()

		if hidden_dims is None:
			hidden_dims = [ 512, 256, 128, 64, 32 ]
		
		self.latent_dim = latent_dim
		self.hidden_dims = hidden_dims

		self.generator = self.Generator(latent_dim, hidden_dims)
		self.discriminator = self.Discriminator(latent_dim, hidden_dims[::-1])

	def sample(self, num_samples: int, device: int, **kwargs) -> Tensor:
		z = torch.randn(num_samples, self.latent_dim)
		z = z.to(device)

		return z

	class Generator(Base.Generator):
		def __init__(
			self,
			latent_dim: int,
			hidden_dims: List,
			**kwargs
		):
			super(GAN.Generator, self).__init__()

			self.latent_dim = latent_dim
			self.hidden_dims = hidden_dims

			self.__init_decoder()

		def __init_decoder(self) -> None:
			self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dims[0] * 2 * 2)

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
						nn.LeakyReLU()
					)
				)

			self.decoder = nn.Sequential(*modules)

			self.output = nn.Sequential(
				nn.ConvTranspose2d(
					self.hidden_dims[-1],
					self.hidden_dims[-1],
					kernel_size=3,
					stride=2,
					padding=1,
					output_padding=1
				),
				nn.BatchNorm2d(self.hidden_dims[-1]),
				nn.LeakyReLU(),
				nn.Conv2d(
					self.hidden_dims[-1],
					out_channels=3,
					kernel_size=3,
					padding=1
				),
				nn.Tanh()
			)
		
		def forward(self, z: Tensor, **kwargs) -> Tensor:
			result = self.decoder_input(z)
			result = result.view(-1, self.hidden_dims[0], 2, 2)
			result = self.decoder(result)
			result = self.output(result)

			return result
		
		def loss_function(self, *args, **kwargs) -> Tensor:
			pred_fake = args[0]
			loss = F.binary_cross_entropy(pred_fake, torch.ones_like(pred_fake))

			return loss

	class Discriminator(Base.Discriminator):
		def __init__(
			self,
			latent_dim: int,
			hidden_dims: List,
			**kwargs
		):
			super(GAN.Discriminator, self).__init__()

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
						nn.LeakyReLU()
					)
				)
				in_channels = hidden_dim

			self.encoder = nn.Sequential(*modules)
			self.fc = nn.Sequential(
				nn.Linear(self.hidden_dims[-1] * 2 * 2, self.latent_dim),
				nn.LeakyReLU()
			)
			self.out = nn.Sequential(
				nn.Linear(self.latent_dim, 1),
				nn.Sigmoid()
			)

		def forward(self, x: Tensor, **kwargs) -> Tensor:
			result = self.encoder(x)
			result = torch.flatten(result, start_dim=1)
			result = self.fc(result)
			result = self.out(result)

			return result

		def loss_function(self, *args, **kwargs) -> Tensor:
			pred_fake, pred_real = args[0], args[1]
			fake_loss = F.binary_cross_entropy(pred_fake, torch.zeros_like(pred_fake))
			real_loss = F.binary_cross_entropy(pred_real, torch.ones_like(pred_real))
			loss = (fake_loss + real_loss) / 2

			return loss