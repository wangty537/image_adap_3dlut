from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

class fiveKData(Dataset):

    def __init__(self, root, transform, mode):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.image_dir = os.path.join(root, 'input\\JPG\\480p')
        self.label_dir = os.path.join(root, 'expertC\\JPG\\480p')
        # train set1
        file = open(os.path.join(root, 'train_input.txt'), 'r')
        file_set1 = sorted(file.readlines())
        self.image_list1 = []
        self.label_list1 = []
        for i in range(len(file_set1)):
            self.image_list1.append(os.path.join(self.image_dir, file_set1[i][:-1]+'.jpg'))
            self.label_list1.append(os.path.join(self.label_dir, file_set1[i][:-1]+'.jpg'))
        # train set2
        file = open(os.path.join(root, 'train_label.txt'), 'r')
        file_set2 = sorted(file.readlines())
        self.image_list2 = []
        self.label_list2 = []
        for i in range(len(file_set2)):
            self.image_list2.append(os.path.join(self.image_dir, file_set2[i][:-1]+'.jpg'))
            self.label_list2.append(os.path.join(self.label_dir, file_set2[i][:-1]+'.jpg'))
        # test set
        file = open(os.path.join(root, 'test.txt'), 'r')
        file_set3 = sorted(file.readlines())
        self.image_test_list = []
        self.label_test_list = []
        for i in range(len(file_set3)):
            self.image_test_list.append(os.path.join(self.image_dir, file_set3[i][:-1]+'.jpg'))
            self.label_test_list.append(os.path.join(self.label_dir, file_set3[i][:-1]+'.jpg'))

        self.image_list = self.image_list1 + self.image_list2
        self.label_list = self.label_list1 + self.label_list2

    def __getitem__(self, idx):
        if self.mode == "train":
            img_name = self.image_list[idx]
            label_name = self.label_list[idx]
            img = Image.open(img_name)
            label = Image.open(label_name)

        elif self.mode == "test":
            img_name = self.image_test_list[idx]
            label_name = self.label_test_list[idx]
            img = Image.open(img_name)
            label = Image.open(label_name)
        else:
            print("error mode")

        if self.mode == "train":
            if np.random.random() > 0.1:
                img = TF.hflip(img)
                label = TF.hflip(label)

            a = np.random.uniform(0.8, 1.2)
            img = TF.adjust_brightness(img, a)

        img = self.transform(img)
        label = self.transform(label)

        return img, label, img_name

    def __len__(self):
        if self.mode == "train":
            return len(self.image_list)
        elif self.mode == "test":
            return len(self.image_test_list)


transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
root_dir = "G:\\BaiduNetdiskDownload\\data\\fiveK\\fiveK"
mode = "train"

data_set = fiveKData(root_dir, transform, mode)
data_loader_train = DataLoader(data_set, batch_size=1)
print(len(data_loader_train))

mode = "test"
data_set_test = fiveKData(root_dir, transform, mode)
data_loader_test = DataLoader(data_set_test, batch_size=1)
print(len(data_loader_test))

#
# if __name__ == '__main__':
#     transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
#     root_dir = "G:\\BaiduNetdiskDownload\\data\\fiveK\\fiveK"
#     mode = "train"
#
#     data_set = fiveKData(root_dir, transform, mode)
#     data_loader = DataLoader(data_set, batch_size=4)
#     print(len(data_set))
#     # get some random training images
#
#     def show_im(img_t):
#         plt.figure()
#         img = img_t.numpy()
#         img = np.transpose(img, (1, 2, 0))
#         plt.imshow(img)
#
#     i = 0
#     for data in data_loader:
#
#         imgs, labels, img_names = data
#         print(imgs.shape, imgs.dtype, imgs.max())
#         print(imgs.shape)
#         show_im(make_grid(imgs))
#         show_im(make_grid(labels))
#         print(img_names)
#         plt.show()
#         i = i + 1
#         if i > 16:
#             break




