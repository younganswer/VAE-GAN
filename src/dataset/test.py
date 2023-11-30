import sys
import unittest
import numpy as np
from torch.utils.data	import DataLoader
from torchvision		import transforms
from .CustomDataset		import CustomDataset

class TestCustomDataset(unittest.TestCase):
	def setUp(self) -> None:
		train_transform = transforms.Compose([
			transforms.Resize((256, 256)),
			transforms.RandomCrop(224),
			transforms.RandomHorizontalFlip(0.5),
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
		])
		valid_transform = transforms.Compose([
			transforms.Resize((256, 256)),
			transforms.ToTensor(),
		])
		test_transform = transforms.Compose([
			transforms.Resize((256, 256)),
			transforms.ToTensor(),
		])
		self.train_dataset = CustomDataset(sys.argv[1], split="train", transform=train_transform)
		self.valid_dataset = CustomDataset(sys.argv[1], split="valid", transform=valid_transform)
		self.test_dataset = CustomDataset(sys.argv[1], split="test", transform=test_transform)

	def test_train_dataset(self):
		print("Train dataset size: ", len(self.train_dataset))

	def test_valid_dataset(self):
		print("Valid dataset size: ", len(self.valid_dataset))

	def test_test_dataset(self):
		print("Test dataset size: ", len(self.test_dataset))

if __name__ == '__main__':
	if (len(sys.argv) != 2):
		print("Usage: python3 -m dataset.test <dataset_root>")
		sys.exit(1)
	unittest.main(argv=['first-arg-is-ignored'], exit=False)