# Import user-defined packages
from .gan		import GAN
from ...dataset	import CustomDataset

import torch
from torch.utils.data	import DataLoader
from torchvision		import transforms
from torch.nn			import functional as F

def train(train_loader, learning_rate=0.005, epochs=5):
	device = torch.device(0 if torch.cuda.is_available() else 'cpu')
	print("Using {} device".format(device))
	model = GAN().to(device)
	generator = model.generator
	discriminator = model.discriminator
	generator_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
	discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

	for epoch in range(epochs):
		for i, data in enumerate(train_loader):
			data = data.to(device)

			# Normalize data to [-1, 1]
			min_val = data.min()
			max_val = data.max()
			data = (data - min_val) / (max_val - min_val)
			data = (data - 0.5) / 0.5

			# Add noise to label			
			#noise_factor = 0.01
			noise_factor = 0
			noise = torch.randn(data.shape[0], 1, device=device) * noise_factor
			real_label = torch.ones(data.shape[0], 1, device=device) + noise
			fake_label = torch.zeros(data.shape[0], 1, device=device) + noise

			# Train Discriminator --------------------------------------------------------------
			# Train real data
			discriminator.zero_grad()
			pred_real = discriminator(data)
			real_loss = F.mse_loss(pred_real, real_label)
			if (i + 1) % 200 == 0: # Flip label per 200 steps
				real_loss = F.mse_loss(pred_real, fake_label)
			real_loss.backward()

			# Train fake data
			fake = generator(model.sample(data.shape[0], device))
			pred_fake = discriminator(fake.detach())
			fake_loss = F.mse_loss(pred_fake, fake_label)
			if (i + 1) % 200 == 0: # Flip label per 200 steps
				fake_loss = F.mse_loss(pred_fake, real_label)
			fake_loss.backward()

			discriminator_loss = real_loss + fake_loss / 2
			discriminator_optimizer.step()
			# ----------------------------------------------------------------------------------

			# Train Generator ------------------------------------------------------------------
			generator.zero_grad()
			pred_fake = discriminator(fake)
			generator_loss = F.mse_loss(pred_fake, real_label)
			generator_loss.backward()
			generator_optimizer.step()
			# ----------------------------------------------------------------------------------

			if (i + 1) % 100 == 0:
				print("Epoch [{}/{}], Step [{:4d}/{}], Generator Loss: {:.4f}, Discriminator Loss: {:.4f}".format(
					epoch + 1,
					epochs,
					i + 1,
					len(train_loader),
					generator_loss.item(),
					discriminator_loss.item()
				))

	return model

def main():
	transform = transforms.Compose([
		transforms.Resize((64, 64)),
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,))
	])
	train_data = CustomDataset(
		root='./data/CelebA/',
		split='train',
		transform=transform
	)
	train_loader = DataLoader(
		dataset=train_data,
		batch_size=128,
		shuffle=True,
		drop_last=True,
	)

	model = train(train_loader, learning_rate=0.005, epochs=3)

	torch.save(model.state_dict(), './src/model/GAN/CelebA_64_square.pth')

	return 0

if __name__ == '__main__':
	main()