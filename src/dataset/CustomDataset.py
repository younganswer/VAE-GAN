import os
import glob
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from ..types_ import *

class CustomDataset(Dataset):
	__dataset: Dict = {}
	__train_size = 0.6
	__valid_size = 0.2
	__test_size = 0.2

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

		sorted_data = sorted(glob.glob(os.path.join(self.root, '*')))
		random.seed(4242)
		random.shuffle(sorted_data)

		data_len = len(sorted_data)
		train_size = int(CustomDataset.__train_size * data_len)
		valid_size = int(CustomDataset.__valid_size * data_len)
		test_size = int(CustomDataset.__test_size * data_len)

		CustomDataset.__dataset[self.root]['train'] = sorted_data[:train_size]
		CustomDataset.__dataset[self.root]['valid'] = sorted_data[train_size:train_size + valid_size]
		CustomDataset.__dataset[self.root]['test'] = sorted_data[train_size + valid_size:]
		CustomDataset.__dataset[self.root]['all'] = sorted_data
		
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