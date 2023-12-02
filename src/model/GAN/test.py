# Import user-defined packages
from .gan import GAN

import torch
import unittest
from torchsummary import summary

class TestGAN(unittest.TestCase):
	def setUp(self) -> None:
		self.device = torch.device(0 if torch.cuda.is_available() else 'cpu')
		self.model = GAN().to(self.device)

	def test_summary(self):
		samples = self.model.sample(1, self.device)
		print(summary(self.model, samples[0], (3, 224, 224), device=self.device))

	def test_forward(self):
		z = torch.randn(1, 10)
		x = torch.randn(1, 3, 224, 224)
		y = self.model(z, x)

	def test_loss_function(self):
		z = torch.randn(1, 10)
		x = torch.randn(1, 3, 224, 224)
		y = self.model(z, x)
		loss = self.model.loss_function(*y)

if __name__ == '__main__':
	unittest.main()