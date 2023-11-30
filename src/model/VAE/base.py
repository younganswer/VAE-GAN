from torch		import nn
from abc		import abstractmethod
from ...types_	import *

class Base(nn.Module):
	def __init__(self):
		super(Base, self).__init__()

	def	encode(self, input: Tensor) -> Tensor:
		raise NotImplementedError

	def decode(self, input: Tensor) -> Tensor:
		raise NotImplementedError

	def sample(self, num_samples: int) -> Tensor:
		raise NotImplementedError

	def generate(self, input: Tensor) -> Tensor:
		raise NotImplementedError

	@abstractmethod
	def forward(self, x):
		pass

	@abstractmethod
	def loss_function(self, x):
		pass