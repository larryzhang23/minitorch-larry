import os
import math
import numpy as np
import multiprocessing as mp
from datasets import load_dataset

import minitorch


mnist = load_dataset("ylecun/mnist")


BACKEND = minitorch.TensorBackend(minitorch.FastOps)
BATCH = 64

# Number of classes (10 digits)
C = 10

# Size of images (height and width)
H, W = 28, 28


def RParam(*shape):
    r = 0.1 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(r)


def URParam(low, high, *shape):
    r = minitorch.rand(shape, backend=BACKEND) * (high - low) + low 
    return minitorch.Parameter(r)


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        initial_weight_lb = -1 / math.sqrt(in_size)
        initial_weight_ub = 1 / math.sqrt(in_size)
        self.weights = URParam(initial_weight_lb, initial_weight_ub, in_size, out_size)
        self.bias = URParam(initial_weight_lb, initial_weight_ub, out_size)
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        return x.view(batch, in_size) @ self.weights.value + self.bias.value


class Conv2d(minitorch.Module):
    def __init__(self, in_channels, out_channels, kh, kw):
        super().__init__()
        initial_weight_lb = -1 / math.sqrt(in_channels * kh * kw)
        intial_weight_ub = 1 / math.sqrt(in_channels * kh * kw)
        self.weights = URParam(initial_weight_lb, intial_weight_ub, out_channels, in_channels, kh, kw)
        self.bias = URParam(initial_weight_lb, intial_weight_ub, 1, out_channels, 1, 1)

    def forward(self, input):
        # TODO: Implement for Task 4.5.
        # import pdb; pdb.set_trace()
        return minitorch.conv2d(input, self.weights.value) + self.bias.value


class Network(minitorch.Module):
    """
    Implement a CNN for MNist classification based on LeNet.

    This model should implement the following procedure:

    1. Apply a convolution with 4 output channels and a 3x3 kernel followed by a ReLU (save to self.mid)
    2. Apply a convolution with 8 output channels and a 3x3 kernel followed by a ReLU (save to self.out)
    3. Apply 2D pooling (either Avg or Max) with 4x4 kernel.
    4. Flatten channels, height, and width. (Should be size BATCHx392)
    5. Apply a Linear to size 64 followed by a ReLU and Dropout with rate 25%
    6. Apply a Linear to size C (number of classes).
    7. Apply a logsoftmax over the class dimension.
    """

    def __init__(self):
        super().__init__()

        # For vis
        self.mid = None
        self.out = None

        # TODO: Implement for Task 4.5.
        self.conv1 = Conv2d(1, 4, 3, 3)
        self.conv2 = Conv2d(4, 8, 3, 3)
        self.linear1 = Linear(392, 64)
        self.linear2 = Linear(64, 10)

    def forward(self, x):
        # TODO: Implement for Task 4.5.
        B, c, h, w = x.shape
        x = self.conv1(x).relu()
        self.mid = x 
        x = minitorch.avgpool2d(x, (2, 2))
        x = self.conv2(x).relu()
        self.out = x 
        x = minitorch.avgpool2d(x, (2, 2))
        x = x.view(B, 392)
        x = self.linear1(x).relu()
        # if x.requires_grad():
        #     ignore = False
        # else:
        #     ignore = True
        # x = minitorch.dropout(x, 0.25, ignore)
        x = self.linear2(x)
        return minitorch.logsoftmax(x, dim=-1)
        


def make_mnist_helper(dataset, idx):
    y = dataset['label'][idx]
    vals = [0.0] * 10
    vals[y] = 1.0
    image = np.array(dataset['image'][idx], dtype=np.float64)
    image = ((image / 255) - 0.5) / 0.5
    print(f"finish idx: {idx}")
    return image, vals

def make_mnist(start, stop, split):
    folder = os.path.join(os.path.dirname(__file__), "data")
    X_path = os.path.join(folder, f"{split}_X.npy")
    y_path = os.path.join(folder, f"{split}_y.npy")
    if os.path.exists(X_path) and os.path.exists(y_path):
        with open(X_path, "rb") as f:
            X = np.load(f)
        with open(y_path, "rb") as f:
            ys = np.load(f)
    else:
        dataset = mnist[split]
        with mp.Pool(8) as p:
            X_y = p.starmap(make_mnist_helper, zip([dataset] * (stop - start), range(start, stop)))
        X, ys = zip(*X_y)
        # ys = []
        # X = []
        # dataset = mnist[split]
        # for i in range(start, stop):
        #     y = dataset['label'][i]
        #     vals = [0.0] * 10
        #     vals[y] = 1.0
        #     ys.append(vals)
        #     image = np.array(dataset['image'][i], dtype=np.float64)
        #     image = ((image / 255) - 0.5) / 0.5
        #     X.append(image)
        #     print(f"pic {i} in storage")
        with open(X_path, "wb") as f:
            np.save(f, np.stack(X, axis=0))
        with open(y_path, "wb") as f:
            np.save(f, np.array(ys))
    return X.tolist(), ys.tolist()


def default_log_fn(epoch, total_loss, correct, total, losses, model):
    print(f"Epoch {epoch} loss {total_loss} valid acc {correct}/{total}")


class ImageTrain:
    def __init__(self):
        self.model = Network()

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x], backend=BACKEND))

    def train(
        self, data_train, data_val, learning_rate, max_epochs=500, log_fn=default_log_fn
    ):
        (X_train, y_train) = data_train
        (X_val, y_val) = data_val
        self.model = Network()
        model = self.model
        n_training_samples = len(X_train)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        print({k: v.value.unique_id for k, v in self.model.named_parameters()})
        losses = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0

            model.train()
            for batch_num, example_num in enumerate(
                range(0, n_training_samples, BATCH)
            ):

                if n_training_samples - example_num <= BATCH:
                    continue
                y = minitorch.tensor(
                    y_train[example_num : example_num + BATCH], backend=BACKEND
                )
                x = minitorch.tensor(
                    X_train[example_num : example_num + BATCH], backend=BACKEND
                )
                x.requires_grad_(True)
                y.requires_grad_(True)
                # Forward
                out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                prob = (out * y).sum(1)
                loss = -(prob / y.shape[0]).sum()

                assert loss.backend == BACKEND
                loss.view(1).backward()

                total_loss += loss[0]
                losses.append(total_loss)

                # Update
                optim.step()

                if batch_num % 5 == 0:
                    model.eval()
                    # Evaluate on 5 held-out batches

                    correct = 0
                    for val_example_num in range(0, 1 * BATCH, BATCH):
                        y = minitorch.tensor(
                            y_val[val_example_num : val_example_num + BATCH],
                            backend=BACKEND,
                        )
                        x = minitorch.tensor(
                            X_val[val_example_num : val_example_num + BATCH],
                            backend=BACKEND,
                        )
                        out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                        for i in range(BATCH):
                            m = -1000
                            ind = -1
                            for j in range(C):
                                if out[i, j] > m:
                                    ind = j
                                    m = out[i, j]
                            if y[i, ind] == 1.0:
                                correct += 1
                    log_fn(epoch, total_loss, correct, BATCH, losses, model)

                    total_loss = 0.0
                    model.train()


if __name__ == "__main__":
    data_train, data_val = (make_mnist(0, 5000, 'train'), make_mnist(0, 500, 'test'))
    print("Dataset preparation done.")
    ImageTrain().train(data_train, data_val, learning_rate=0.001)
