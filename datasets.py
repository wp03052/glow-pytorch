import torch
import torchvision.datasets as vdsets


class Dots(torch.utils.data.TensorDataset):

    def __init__(self, *tensors, noisy=False, img_size=64):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.noisy = noisy
        self.img_size = img_size

    def __getitem__(self, index):
        x, y = tuple(tensor[index] for tensor in self.tensors)
        if self.img_size != 64:
            x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(self.img_size, self.img_size)).squeeze(0)
        if self.noisy:
            x = x + 0.03 * torch.randn(size=x.shape, dtype=x.dtype, device=x.device)
            x = 1.0 - torch.abs(1.0 - x)
            x = torch.abs(x)
        return (x, y)
