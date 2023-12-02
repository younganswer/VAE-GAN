import os
import glob
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from ..types_ import *

class CustomDataset(Dataset):
	__dataset: Dict = {}
	def	__init__(self, root: str, split: str = "train", transform: transforms = None) -> None:
		super(CustomDataset, self).__init__()
		self.root = root
		self.split = split.lower()
		self.transform = transform

		valid_split = [ "train", "valid", "test", "all" ]
		if self.split not in valid_split:
			raise ValueError(f'Invalid split name {self.split}\nValid split names: {valid_split}')
		if self.root not in CustomDataset.__dataset:
			self.__init_dataset()			

	def	__init_dataset(self) -> None:
		CustomDataset.__dataset[self.root] = {}

		train_data = sorted(glob.glob(os.path.join(self.root, 'train', '*.jpg')))
		test_data = sorted(glob.glob(os.path.join(self.root, 'test', '*.jpg')))

		train_size = int(len(train_data) * 0.8)
		valid_size = len(train_data) - train_size
		
		CustomDataset.__dataset[self.root]['train'] = train_data[:train_size]
		CustomDataset.__dataset[self.root]['valid'] = train_data[train_size:]
		CustomDataset.__dataset[self.root]['test'] = test_data
		CustomDataset.__dataset[self.root]['all'] = train_data + test_data
		
		random.seed(42)
		random.shuffle(CustomDataset.__dataset[self.root]['train'])
		random.shuffle(CustomDataset.__dataset[self.root]['test'])
		random.shuffle(CustomDataset.__dataset[self.root]['valid'])
		random.shuffle(CustomDataset.__dataset[self.root]['all'])
	
	def	__getitem__(self, index: int) -> Tensor:
		if len(CustomDataset.__dataset[self.root][self.split]) <= index:
			raise IndexError(f'Index {index} out of bounds for {self.split} dataset with size {len(CustomDataset.__dataset[self.root]["train"])}')
		path = CustomDataset.__dataset[self.root][self.split][index]
		image = Image.open(path).convert('RGB')
		if self.transform:
			image = self.transform(image)
		return image

	def	__len__(self) -> int:
		return len(CustomDataset.__dataset[self.root][self.split])