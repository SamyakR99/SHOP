import sys
sys.path.insert(0, './')
sys.path.insert(0, '../')
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
import os
import torchvision.transforms as transforms
from randaugment import RandAugment

from PIL import ImageDraw
import random



class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x

class foodseg103(data.Dataset):
    def __init__(self, root, data_split, img_size=224):
        # data_split = train / val
        self.root = root
        self.classnames = ["candy", "egg tart", "french fries", "chocolate", "biscuit", "popcorn", "pudding", "ice cream", "cheese butter", "cake", "wine", "milkshake", "coffee", "juice", "milk", "tea", "almond", "red beans", "cashew", "dried cranberries", "soy", "walnut", "peanut", "egg", "apple", "date", "apricot", "avocado", "banana", "strawberry", "cherry", "blueberry", "raspberry", "mango", "olives", "peach", "lemon", "pear", "fig", "pineapple", "grape", "kiwi", "melon", "orange", "watermelon", "steak", "pork", "chicken duck", "sausage", "fried meat", "lamb", "sauce", "crab", "fish", "shellfish", "shrimp", "soup", "bread", "corn", "hamburg", "pizza", "hanamaki baozi", "wonton dumplings", "pasta", "noodles", "rice", "pie", "tofu", "eggplant", "potato", "garlic", "cauliflower", "tomato", "kelp", "seaweed", "spring onion", "rape", "ginger", "okra", "lettuce", "pumpkin", "cucumber", "white radish", "carrot", "asparagus", "bamboo shoots", "broccoli", "celery stick", "cilantro mint", "snow peas", "cabbage", "bean sprouts", "onion", "pepper", "green beans", "French beans", "king oyster mushroom", "shiitake", "enoki mushroom", "oyster mushroom", "white button mushroom", "salad", "other ingredients"]
        
        
        self.data_split = data_split
        if data_split == 'train_seg':
            self.labels_lab = np.load('/home/samyakr2/food/FoodSeg103/Images/final_img_set/train_labels.npy', allow_pickle=True).item()
        
        if data_split == 'test_seg':
            self.labels_lab = np.load('/home/samyakr2/food/FoodSeg103/Images/final_img_set/test_labels.npy', allow_pickle=True).item()
            
        
        image_list_file = os.path.join('/home/samyakr2/food/FoodSeg103/Images/final_img_set/%s.txt' % data_split)
        
        with open(image_list_file) as f:
            image_list = f.readlines()
        self.image_list = [a.strip() for a in image_list]

        
        self.image_list = self.image_list

        
        train_transform = transforms.Compose([
            # transforms.RandomResizedCrop(img_size)
            # transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            CutoutPIL(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        test_transform = transforms.Compose([
            # transforms.CenterCrop(img_size),
            # transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
        # self.masks_img_transform = mask_img_transform
        if self.data_split == 'train_seg':
            self.transform = train_transform
        elif self.data_split == 'test_seg':
            self.transform = test_transform
        else:
            raise ValueError('data split = %s is not supported in Nus Wide' % self.data_split)

        
    def __len__(self):
        return len(self.image_list)
    

    def __getitem__(self, index):
        

        img_path = os.path.join('/home/samyakr2/food/FoodSeg103/Images/final_img_set/',self.data_split[:-4], self.image_list[index])
        img = Image.open(img_path).convert('RGB')
        
        label_vector = torch.tensor(self.labels_lab[self.image_list[index][:-4]])      
        targets = label_vector.float()
        target = targets[None, ]
        
        if self.transform is not None:
            img = self.transform(img)

        return (img, self.image_list[index]), target
    def name(self):
        return 'foodseg103'
