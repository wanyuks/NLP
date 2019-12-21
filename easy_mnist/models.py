import torch
import torch.nn as nn
import torch.nn.functional as F


# 二分类logistic
class LogisticRegressionBinary(nn.Module):
    def __init__(self, input_size, out_put_size):
        super(LogisticRegressionBinary, self).__init__()
        self.input_size = input_size
        self.out_put_size = out_put_size
        self.LR = nn.Linear(in_features=self.input_size,
                            out_features=self.output_size)

    def forward(self, x):
        out = self.LR(x)
        out = torch.sigmoid(out)
        return out


# 多分类的logistic
class LogisticRegressionMulti(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegressionMulti, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.LR = nn.Linear(in_features=self.input_size,
                            out_features=self.output_size)

    def forward(self, x):
        x = x.reshape(-1, self.input_size)
        return self.LR


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

