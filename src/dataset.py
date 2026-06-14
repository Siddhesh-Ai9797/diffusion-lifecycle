import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloader(batch_size=64, data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,1] range
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1, 1]
    ])

    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


if __name__ == "__main__":
    loader = get_dataloader()
    images, labels = next(iter(loader))
    print("Batch shape:", images.shape)
    print("Min pixel value:", images.min().item())
    print("Max pixel value:", images.max().item())