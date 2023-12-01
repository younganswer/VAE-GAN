import torch
import unittest
from torchsummary import summary
from .vae import VAE

class TestVAE(unittest.TestCase):
	def setUp(self) -> None:
		self.model = VAE()

	def test_summary(self):
		print(summary(self.model, (3, 64, 64), device='cpu'))

	def test_forward(self):
		x = torch.randn(1, 3, 64, 64)
		y = self.model(x)

	def test_loss_function(self):
		x = torch.randn(1, 3, 64, 64)
		y = self.model(x)
		loss = self.model.loss_function(*y, M_N=0.005)

if __name__ == '__main__':
	unittest.main()