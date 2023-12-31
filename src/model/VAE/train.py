# Import user-defined packages
from .vae		import VAE
from ...dataset	import CustomDataset

import torch
from torch.utils.data	import DataLoader
from torchvision		import transforms

def train(train_loader, learning_rate=0.005, epochs=10):
	device = torch.device(0 if torch.cuda.is_available() else 'cpu')
	print("Using {} device".format(device))
	model = VAE().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	criterion = model.loss_function

	for epoch in range(epochs):
		for i, data in enumerate(train_loader):
			data = data.to(device)
			optimizer.zero_grad()
			outputs = model(data)
			result = criterion(*outputs, M_N=0.00025)
			loss = result['Loss']
			recon_loss = result['Reconstruction_Loss']
			kl_loss = result['KLD']
			loss.backward()
			optimizer.step()
			if (i + 1) % 100 == 0:
				print('Epoch [{}/{}], Step [{:4d}/{}], Loss: {:.4f}, Recon Loss: {:.4f}, KL Div: {:.4f}'.format(
					epoch + 1, epochs, i + 1, len(train_loader), loss.item(), recon_loss.item(), kl_loss.item()
				))
	
	return model

def main():
	train_transform = transforms.Compose([
		transforms.Resize((64, 64)),
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,))
	])
	train_data = CustomDataset(
		root='./data/CelebA/',
		split='train',
		transform=train_transform
	)
	train_loader = DataLoader(
		dataset=train_data,
		batch_size=128,
		shuffle=True,
		drop_last=True,
	)

	model = train(train_loader, learning_rate=0.005, epochs=100)

	torch.save(model.state_dict(), './src/model/VAE/CelebA_64_square.pth')

	return 0

if __name__ == '__main__':
	main()