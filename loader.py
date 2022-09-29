import h5py
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class PetDataset(Dataset):

    def __init__(self, path):
        super(PetDataset, self).__init__()
        self.images = h5py.File(path + 'images.h5', 'r')['images']
        self.labels = h5py.File(path + 'binary.h5', 'r')['binary']
        self.bboxes = h5py.File(path + 'bboxes.h5', 'r')['bboxes']
        self.masks = h5py.File(path + 'masks.h5', 'r')['masks']

    def transpose(self, imgs):
        '''
        Swaps the dimensions of the image, resulting in the shape [N, C, H, W].

        parameters:
        imgs (torch.Tensor): Images to transpose

        returns:
        img_transpose (torch.Tensor): Images with correct dimensions, usable with our models
        '''
        return torch.transpose(torch.transpose(imgs, 3, 1), 2, 3)

    def __getitem__(self, index):
        image = self.transpose(torch.from_numpy(self.images[index]))
        label = torch.from_numpy(self.labels[index])
        bbox = torch.from_numpy(self.bboxes[index])
        mask = self.transpose(torch.from_numpy(self.masks[index]))
        return image, mask, label, bbox
    
    def __len__(self):
        return self.images.shape[0]


def PetDatasetGenerator(dataset, batch_size, shuffle=False):
    idxs = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(idxs)
    start, stop = 0, len(dataset)
    while start < stop:
        end = min(start + batch_size, stop)
        images, masks, labels, bboxes = dataset[np.sort(idxs[start:end]) if shuffle else idxs[start:end]]
        start += batch_size
        yield images, masks, labels, bboxes


if __name__ == '__main__':
    # just some experiment code :)
    img_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    trainset = PetDataset('datasets-oxpet/train/')
    traingenerator = PetDatasetGenerator(trainset, 100, shuffle=True)
    start = time.time()
    for i, data in enumerate(traingenerator):
        images, masks, labels, bboxes = data
        images = img_transform(images)
    print(time.time() - start)
