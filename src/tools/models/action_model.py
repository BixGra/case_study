import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

# Settings

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

ACTIONS = {
    "no action": torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0]),
    "run": torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0]),
    "pass": torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0]),
    "rest": torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0]),
    "walk": torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0]),
    "dribble": torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0]),
    "shot": torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0]),
    "tackle": torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0]),
    "cross": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1]),
}

# Model


class ExtractLSTM(nn.Module):
    def forward(self, x):
        tensor, _ = x
        return tensor[:, -1, :]


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        output = self.fc(x)
        return output

    def save(self, path: str = "./models") -> str:
        name = f"{self._get_name()}_{int(time.time())}"
        torch.save(self.fc.state_dict(), f"{path.rstrip('/')}/{name}")
        return name

    def load(self, model_path: str):
        self.fc.load_state_dict(torch.load(model_path))
        self.eval()
        return self


class ActionModel(Model):
    def __init__(self, sequence_length: int = 3):
        super(ActionModel, self).__init__()
        self.sequence_length = sequence_length
        self.fc = nn.Sequential(
            nn.LSTM(9, 5, self.sequence_length),
            ExtractLSTM(),
            nn.Linear(5, 9),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        output = self.fc(x)
        return output

    def predict(self, x: Tensor) -> str:
        output = self(x)
        output = (
            random.choices(
                list(ACTIONS.items()),
                weights=torch.squeeze(torch.squeeze(output, dim=0), dim=0),
            )[0]
        )
        return output


def train(
    loader: DataLoader,
    model: Model,
    criterion: nn.Module,
    optimizer: optim,
    num_epochs: int = 10,
) -> list[np.ndarray]:
    """
    Train function
    """
    start_time = time.time()
    losses = []
    for epoch in range(num_epochs):
        if len(losses) > 3 and round(max(losses[-3:]), 3) == round(min(losses[-3:]), 3):
            print("Stopped by stable loss")
            break
        losses_ = []
        for data in loader:
            # Data loading
            x = data[0]
            y = data[1]
            if cuda:
                x = data[0].cuda()
                y = data[1].cuda()
            x.requires_grad = True
            y.requires_grad = True
            output = model(x)
            # Backpropagation
            optimizer.zero_grad()
            loss = criterion(y, output)
            loss.backward()
            optimizer.step()
            # Display & metrics
            losses_.append(float(loss))
            print(
                "Epoch {:>4} of {}, Train Loss: {:5.2f}, Elapsed time: {:.2f}".format(
                    epoch + 1, num_epochs, losses_[-1], time.time() - start_time
                ),
                end="\r",
                flush=True,
            )
        losses.append(np.mean(losses_))
        print(
            "Epoch {:>4} of {}, Train Loss: {:5.2f}, Elapsed time: {:.2f}".format(
                epoch + 1, num_epochs, losses[-1], time.time() - start_time
            ),
            end="\n",
            flush=True,
        )
    return losses
