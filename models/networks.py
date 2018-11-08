import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_network import BaseNet



class CustomNet(BaseNet):
    def __init__(self, args):
        super(CustomNet, self).__init__()
        kw = args.kernel_width
        pad = kw // 2
        N_ch_input, N_class_output = 1, 10

        self.conv1 = nn.Conv2d(N_ch_input, 32, kernel_size=kw, padding=pad)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=kw, padding=pad)
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=kw, padding=pad)
        self.conv3_1 = nn.Conv2d(64, 256, kernel_size=kw, padding=pad)
        self.conv3_2 = nn.Conv2d(64, 256, kernel_size=kw, padding=pad)
        self.fc1 = nn.Linear(7*7*512, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.output = nn.Linear(500, N_class_output)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x1 = F.relu(F.max_pool2d(self.conv2_1(x), kernel_size=2))
        x2 = F.relu(F.max_pool2d(self.conv2_2(x), kernel_size=2))

        x1 = F.relu(F.max_pool2d(self.conv3_1(x1), kernel_size=2))
        x2 = F.relu(F.max_pool2d(self.conv3_2(x2), kernel_size=2))

        x = torch.cat([x1,x2], dim=1)
        x = x.view(-1, 7*7*512)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.output(x)
        return F.log_softmax(x, dim=1)
