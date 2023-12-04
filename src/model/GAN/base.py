# Import user-defined packages
from ...types_ import *

import torch
from torch	import nn
from abc	import ABC, abstractmethod

class Base(ABC, nn.Module):
	def	__init__(self):
		super(Base, self).__init__()

	@abstractmethod
	def sample(self, num_samples: int, device: int, **kwargs) -> Tensor:
		pass
	
	class Generator(nn.Module):
		def __init__(self):
			super(Base.Generator, self).__init__()


		@abstractmethod
		def forward(self, z: Tensor) -> Tensor:
			pass

	class Discriminator(nn.Module):
		def __init__(self):
			super(Base.Discriminator, self).__init__()

		@abstractmethod
		def forward(self, x):
			pass