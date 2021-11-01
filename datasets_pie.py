import torch
import math
import numpy as np


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


def gen_pie(params):
    count = int(params[0])

    # 0.35 + k/20: 0.4-0.8, range 1-9
    size = int(params[1])
    if size == 0:
        size = np.random.randint(1, 10)
    size = size / 20.0 + 0.35

    # (k-5)/20: -0.2 - 0.2
    locx = int(params[2])
    if locx == 0:
        locx = np.random.randint(1, 10)
    locx = (locx - 5) / 20.0

    locy = int(params[3])
    if locy == 0:
        locy = np.random.randint(1, 10)
    locy = (locy - 5) / 20.0

    # k/10, 0.0-1.0
    color = int(params[4])
    if color == 0:
        color = np.random.randint(1, 10)
    color = color / 10.0

    resolution = 1000
    lutx = []
    luty = []
    lut2 = []
    for i in range(64):
        for j in range(64):
            x = (i - 32.0) / 32.0 - locx
            y = (j - 32.0) / 32.0 - locy
            if x ** 2 + y ** 2 <= size ** 2:
                lutx.append(i)
                luty.append(j)
                lut2.append(int((math.atan2(y, x) / 2.0 / math.pi + 0.5) * (resolution - 1)))

    random_color = np.random.uniform(0, 0.9, size=(count, 3))
    random_color[0, 0] = np.random.uniform(0.8, 0.9)
    random_color[0, 1:] = 0.0
    random_color[1:, 0] = 0.0
    random_weights = np.random.uniform(0.01, 1, size=count - 1)
    random_weights = np.concatenate([[color], (1 - color) * random_weights / np.sum(random_weights)])

    color_band = np.zeros(shape=(resolution, 3), dtype=np.float32)
    color_weights = np.cumsum(random_weights)
    color_weights = np.insert(color_weights, 0, 0.0)
    for c in range(len(random_color)):
        color_band[int(resolution * color_weights[c]):int(resolution * color_weights[c + 1])] = random_color[c]

    for i in range(3):
        swap_start, swap_end = 0, 0
        while swap_end - swap_start < 10:
            swap_start = np.random.randint(0, resolution // 2 - 10)
            swap_end = np.random.randint(10, resolution // 2)
        recv_start = np.random.randint(resolution // 2, resolution - swap_end + swap_start)
        recv_end = recv_start + swap_end - swap_start
        buffer = color_band[swap_start:swap_end].copy()
        color_band[swap_start:swap_end] = color_band[recv_start:recv_end]
        color_band[recv_start:recv_end] = buffer

    canvas = np.ones(shape=(64, 64, 3), dtype=np.float32)
    canvas[lutx, luty] = color_band[lut2]
    canvas= canvas.transpose(2,0,1)
    return canvas, color


class PieDataset():
    def __init__(self, params=('40000',)):
        """ params is a tuple of strings, each string contain five integers,
        representing [num-of-color, size, x-location, y-location, proportion-of-red];
        each value is from 1-9, if value is 0, than this dimension is randomly selected.
        For data point selects a random param"""
        self.data_dims = [64, 64, 3]
        self.name = "pie"
        self.batch_size = 100
        self.params = params

        self.train_ptr = 0
        self.train_cache = []
        self.max_size = 200000

        self.range = [0.0, 1.0]

    def next_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        prev_ptr = self.train_ptr
        self.train_ptr += batch_size
        if self.train_ptr > self.max_size:
            prev_ptr = 0
            self.train_ptr = batch_size
        while self.train_ptr > len(self.train_cache):
            self.train_cache.append(gen_pie(np.random.choice(self.params)))
        out = np.stack(self.train_cache[prev_ptr:self.train_ptr], axis=0)
        x = torch.Tensor(np.array(list(out[:, 0]), dtype=np.float32))
        y = torch.Tensor(np.array(list(out[:, 1]), dtype=np.float32))
        return (x, y)

    def reset(self):
        self.train_ptr = 0

    @staticmethod
    def eval_size(arr):
        return np.array([compute_radius(img) for img in arr])

    @staticmethod
    def eval_color_proportion(arr):
        return np.array([compute_proportion(img) for img in arr])

    @staticmethod
    def eval_location(arr):
        return np.stack([compute_location(img) for img in arr], axis=0)
    
