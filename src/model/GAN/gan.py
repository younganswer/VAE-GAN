from .base import Base

class GAN(Base):
	def __init__(self):
		super(GAN, self).__init__()

	def forward(self, x):
		pass

	def loss_function(self, x):
		pass

	class Generator(Base.Generator):
		def __init__(self):
			super(GAN.Generator, self).__init__()

		def forward(self, x):
			pass

	class Discriminator(Base.Discriminator):
		def __init__(self):
			super(GAN.Discriminator, self).__init__()

		def forward(self, x):
			pass