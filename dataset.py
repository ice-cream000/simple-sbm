import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class MyTransforms():
    def __init__(self):
        pass
    def __call__(self, x):
        return 2.0 * x - 1.0

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),  # PIL Image → Tensor (C, H, W), [0, 255] → [0, 1]
        MyTransforms() # [0, 1] → [-1, 1]
    ])

class DatasetCifar10(Dataset):
    def __init__(self, path='./data', transform=None, train=True, target_digits=None):
        self.transform = transform
        self.train = train
        self.data = torchvision.datasets.CIFAR10(root=path, train=self.train, download=True)
        self.target_digits = target_digits

        if self.target_digits is not None:
            self.indices = [i for i, (_, label) in enumerate(self.data) if label in self.target_digits]
        else:
            self.indices = list(range(len(self.data)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        out_data = self.data[actual_idx][0]
        out_label = self.data[actual_idx][1]
        if self.transform is not None:
            out_data = self.transform(out_data)
        return out_data, out_label