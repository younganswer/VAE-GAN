import torch
from torch		import nn
from torch.nn	import functional as F
from .base 		import *
from ...types_	import *

class VAE(Base):
	def	__init__(
		self,
		latent_dim: int = 16,
		hidden_dims: List = None,
		**kwargs
	) -> None:
		super(VAE, self).__init__()

		if hidden_dims is None:
			hidden_dims = [ 32, 64, 128, 256, 512 ]

		self.latent_dim = latent_dim
		self.hidden_dims = hidden_dims

		self.__init_encoder()
		self.__init_decoder()
		
	def	__init_encoder(self) -> None:
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
		self.fc_mu = nn.Linear(self.hidden_dims[-1] * 2 * 2, self.latent_dim)
		self.fc_var = nn.Linear(self.hidden_dims[-1] * 2 * 2, self.latent_dim)

	def	__init_decoder(self) -> None:
		self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dims[-1] * 2 * 2)

		modules = []

		for i in range(len(self.hidden_dims) - 1, 0, -1):
			modules.append(
				nn.Sequential(
					nn.ConvTranspose2d(
						self.hidden_dims[i],
						self.hidden_dims[i-1],
						kernel_size=3,
						stride=2,
						padding=1,
						output_padding=1
					),
					nn.BatchNorm2d(self.hidden_dims[i-1]),
					nn.LeakyReLU()
				)
			)

		self.decoder = nn.Sequential(*modules)

		self.output = nn.Sequential(
			nn.ConvTranspose2d(
				self.hidden_dims[0],
				self.hidden_dims[0],
				kernel_size=3,
				stride=2,
				padding=1,
				output_padding=1
			),
			nn.BatchNorm2d(self.hidden_dims[0]),
			nn.LeakyReLU(),
			nn.Conv2d(
				self.hidden_dims[0],
				out_channels=3,
				kernel_size=3,
				padding=1
			),
			nn.Tanh()
		)

	def	encode(self, input: Tensor) -> List[Tensor]:
		result = self.encoder(input)
		result = torch.flatten(result, start_dim=1)
		mu = self.fc_mu(result)
		log_var = self.fc_var(result)

		return [mu, log_var]

	def	decode(self, z: Tensor) -> Tensor:
		result = self.decoder_input(z)
		result = result.view(-1, self.hidden_dims[-1], 2, 2)
		result = self.decoder(result)
		result = self.output(result)

		return result

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
		mu, log_var = self.encode(input)
		z = self.reparameterize(mu, log_var)
		return [ self.decode(z), input, mu, log_var ]

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
			'loss': loss,
			'Reconstruction_Loss': recons_loss.detach(),
			'KLD': kld_loss.detach()
		}

	def sample(self, num_samples:int, device: int, **kwargs) -> Tensor:
		z = torch.randn(num_samples, self.latent_dim)
		z = z.to(device)
		samples = self.decode(z)

		return samples

	def generate(self, x: Tensor, **kwargs) -> Tensor:
		return self.forward(x)[0]