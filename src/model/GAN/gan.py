from .base 		import Base
from ...types_	import *

class GAN(Base):
	def __init__(
		self,
		latent_dim: int = 10,
		hidden_dims: List = None,
		**kwargs
	):
		super(GAN, self).__init__()

		if hidden_dims is None:
			hidden_dims = [512, 256, 128, 64, 32]
		
		self.latent_dim = latent_dim
		self.hidden_dims = hidden_dims
		self.generator = self.Generator(latent_dim, hidden_dims)
		self.discriminator = self.Discriminator(latent_dim, hidden_dims.reverse())

	def forward(self, x: Tensor, **kwargs) -> Tensor:
		pass

	def loss_function(self, x):
		pass

	class Generator(Base.Generator):
		def __init__(
			self,
			latent_dim: int,
			hidden_dims: List,
			**kwargs
		):
			super(GAN.Generator, self).__init__()

			self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 7 * 7)

			modules = []
			for i in range(len(hidden_dims)):
				modules.append(
					nn.Sequential(
						nn.ConvTranspose2d(
							hidden_dims[i],
							hidden_dims[i+1],
							kernel_size=3,
							stride=2,
							padding=1,
							output_padding=1
						),
						nn.BatchNorm2d(hidden_dims[i+1]),
						nn.LeakyReLU()
					)
				)

			self.decoder = nn.Sequential(*modules)

			self.output = nn.Sequential(
				nn.ConvTranspose2d(
					hidden_dims[-1],
					out_channels=3,
					kernel_size=3,
					stride=2,
					padding=1,
					output_padding=1
				),
				nn.BatchNorm2d(3),
				nn.Tanh()
			)
		
		def sample(self, num_samples: int, device: int, **kwargs) -> Tensor:
			z = torch.randn(num_samples, self.latent_dim)
			z = z.to(device)

			return self.decode(z)

		def forward(self, z: Tensor) -> Tensor:
			result = self.decoder_input(z)
			result = result.view(-1, self.hidden_dims[0], 7, 7)
			result = self.decoder(result)
			result = self.output(result)

			return result

	class Discriminator(Base.Discriminator):
		def __init__(
			self,
			latent_dim: int,
			hidden_dims: List,
			**kwargs
		):
			super(GAN.Discriminator, self).__init__()

			modules = []
			in_channels = 3
			for hidden_dim in hidden_dims:
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
				nn.Linear(hidden_dims[-1] * 7 * 7, self.latent_dim),
				nn.LeakyReLU()
			)
			self.out = nn.Sequential(
				nn.Linear(self.latent_dim, 1),
				nn.Tanh()
			)

		def forward(self, x: Tensor, **kwargs) -> Tensor:
			result = self.encoder(x)
			result = torch.flatten(result, start_dim=1)
			result = self.fc(result)
			result = self.out(result)

			return result