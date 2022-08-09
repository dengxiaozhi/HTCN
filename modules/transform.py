import cv2
import numpy as np
import torchvision.transforms as transforms


class GaussianBlur:
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()
        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample


class Transforms:
    def __init__(self, size, s=1.0, mean=None, std=None, blur=False):
        self.train_transform = [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        if blur:
            self.train_transform.append(GaussianBlur(kernel_size=23))
        self.train_transform.append(transforms.ToTensor())
        self.test_transform = [
            transforms.Resize(size=(size, size)),
            transforms.ToTensor(),
        ]
        if mean and std:
            self.train_transform.append(transforms.Normalize(mean=mean, std=std))
            self.test_transform.append(transforms.Normalize(mean=mean, std=std))
        self.train_transform = transforms.Compose(self.train_transform)
        self.test_transform = transforms.Compose(self.test_transform)

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x), self.train_transform(x)
