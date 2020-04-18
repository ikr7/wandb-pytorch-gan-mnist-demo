from torchvision.datasets import MNIST
import torchvision.transforms as transforms


def get_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    return MNIST(root='~/toy-data', train=True, transform=transform)