import torch
from torch import nn
from abc import ABC, abstractmethod

class Base(ABC, nn.Module):
	def	__init__(self):
		super(Base, self).__init__()

		self.generator = self.Generator()
		self.discriminator = self.Discriminator()

	def generate(self, input: torch.Tensor) -> torch.Tensor:
		return self.generator(input)

	@abstractmethod
	def forward(self, x):
		pass

	@abstractmethod
	def loss_function(self, x):
		pass

	class Generator(nn.Module):
		def __init__(self):
			super(Base.Generator, self).__init__()

		@abstractmethod
		def forward(self, x):
			pass

	class Discriminator(nn.Module):
		def __init__(self):
			super(Base.Discriminator, self).__init__()

		@abstractmethod
		def forward(self, x):
			pass