import os
import sys
import random

target_dataset = input("Enter the name of the dataset: ")

cur_dir = os.getcwd()
src_dir = os.path.join(cur_dir, 'data/{}'.format(target_dataset))
train_dir = os.path.join(src_dir, 'train')
test_dir = os.path.join(src_dir, 'test')

print("Current directory: ", cur_dir)
print("Source directory: ", src_dir)
print("Train directory: ", train_dir)
print("Test directory: ", test_dir)

if not os.path.exists(train_dir):
	os.makedirs(train_dir)
	print("Created train directory.")
if not os.path.exists(test_dir):
	os.makedirs(test_dir)
	print("Created test directory.")

# get the list of all images
img_list = os.listdir(src_dir)
img_list = [os.path.join(src_dir, img) for img in img_list]
img_list = [img for img in img_list if os.path.isfile(img)]
img_list = [img for img in img_list if img.endswith('.jpg')]
print("Total images: ", len(img_list))

# shuffle the list
random.shuffle(img_list)
print("Shuffle the list.")

train_ratio = 0.8
test_ratio = 0.2

train_list = img_list[:int(len(img_list)*train_ratio)]
test_list = img_list[int(len(img_list)*train_ratio):]
print("Train images: ", len(train_list))
print("Test images: ", len(test_list))

for img in train_list:
	os.rename(img, os.path.join(train_dir, os.path.basename(img)))

for img in test_list:
	os.rename(img, os.path.join(test_dir, os.path.basename(img)))

print('Done!')