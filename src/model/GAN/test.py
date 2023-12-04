# Import user-defined packages
from .gan import GAN

import torch
import unittest
from torchsummary import summary

class TestGAN(unittest.TestCase):
	def setUp(self) -> None:
		self.device = torch.device(0 if torch.cuda.is_available() else 'cpu')
		self.model = GAN().to(self.device)
		self.generator = self.model.generator
		self.discriminator = self.model.discriminator

	def test_summary(self):
		print("Generator")
		print(summary(self.generator, (128,)))
		print("Discriminator")
		print(summary(self.discriminator, (3, 64, 64)))

	def test_generator_forward(self):
		z = torch.randn(1, 128)
		x = self.generator(z)
		self.assertEqual(x.shape, (1, 3, 64, 64))

	def test_generator_loss_function(self):
		z = torch.randn(1, 128)
		x = self.generator(z)
		pred_fake = self.discriminator(x)
		loss = self.generator.loss_function(pred_fake)
		self.assertEqual(loss.shape, ())

	def test_discriminator_forward(self):
		x = torch.randn(1, 3, 64, 64)
		y = self.discriminator(x)
		self.assertEqual(y.shape, (1, 1))

	def test_discriminator_loss_function(self):
		x = torch.randn(1, 3, 64, 64)
		y = self.discriminator(x)
		pred_fake = self.discriminator(x)
		pred_real = self.discriminator(x)
		loss = self.discriminator.loss_function(pred_fake, pred_real)
		self.assertEqual(loss.shape, ())

if __name__ == '__main__':
	unittest.main()