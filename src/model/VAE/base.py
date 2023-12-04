# Import user-defined packages
from ...types_	import *

from torch		import nn
from abc		import ABC, abstractmethod

class Base(ABC, nn.Module):
	def __init__(self):
		super(Base, self).__init__()

	@abstractmethod
	def sample(self, num_samples: int, device: int, **kwargs) -> Tensor:
		pass

	@abstractmethod
	def forward(self, x):
		pass

	@abstractmethod
	def loss_function(self, x):
		pass

	def generate(self, input: Tensor) -> Tensor:
		raise NotImplementedError

	class Encoder(nn.Module):
		def __init__(self):
			super(Base.Encoder, self).__init__()

		@abstractmethod
		def forward(self, x):
			pass

	class Decoder(nn.Module):
		def __init__(self):
			super(Base.Decoder, self).__init__()

		@abstractmethod
		def forward(self, x):
			pass