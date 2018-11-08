import os
import torch
import torch.nn as nn


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    def save_network(self, save_dir, name, epoch):
        filename = os.path.join(save_dir, name, '%d_net.pth' % (epoch))
        torch.save(self.state_dict(), filename)

    def load_network(self, save_dir, name, epoch):
        filename = os.path.join(save_dir, name, '%d_net.pth' % (epoch))
        self.load_state_dict(torch.load(filename))
