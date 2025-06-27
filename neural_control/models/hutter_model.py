import torch.nn as nn
import torch
import torch.nn.functional as F


class Net(nn.Module):
    """
    Simple MLP with three hidden layers, based on RL work of Marco Hutter's
    group
    """

    def __init__(
        self, state_dim, horizon, ref_dim, nr_actions_predict, conv=True
    ):
        """
        in_size: number of input neurons (features)
        out_size: number of output neurons
        """
        super(Net, self).__init__()
        self.states_in = nn.Linear(state_dim, 64)
        self.conv_ref = nn.Conv1d(ref_dim, 20, kernel_size=3)
        # the size will be nr_channels * (1dlength - kernel_size + 1)
        self.horizon = horizon
        self.conv = conv
        self.reshape_len = 20 * (horizon - 2) if conv else 64
        self.ref_in = nn.Linear(horizon * ref_dim, 64)
        self.fc1 = nn.Linear(64 + self.reshape_len, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, nr_actions_predict)

    def forward(self, state, ref):
        # process state and reference differently
        state = torch.tanh(self.states_in(state))
        if self.conv:
            # ref = torch.reshape(ref, (-1, self.ref_dim, 3))
            ref = torch.transpose(ref, 1, 2)
            ref = torch.relu(self.conv_ref(ref))
            ref = torch.reshape(ref, (-1, self.reshape_len))
        else:
            ref = torch.tanh(self.ref_in(ref))
        # concatenate
        x = torch.hstack((state, ref))
        # normal feed-forward
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc_out(x)
        return x

class Net_limit(nn.Module):
    """
    Simple MLP with three hidden layers, based on RL work of Marco Hutter's
    group
    """

    def __init__(
        self, state_dim, horizon, ref_dim, nr_actions_predict, low_bound, up_bound, conv=True
    ):
        """
        in_size: number of input neurons (features)
        out_size: number of output neurons
        """
        super(Net_limit, self).__init__()
        self.states_in = nn.Linear(state_dim, 64)
        self.conv_ref = nn.Conv1d(ref_dim, 20, kernel_size=3)
        # the size will be nr_channels * (1dlength - kernel_size + 1)
        self.horizon = horizon
        self.conv = conv
        self.reshape_len = 20 * (horizon - 2) if conv else 64
        self.ref_in = nn.Linear(horizon * ref_dim, 64)
        self.fc1 = nn.Linear(64 + self.reshape_len, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, nr_actions_predict)

        self.low_bound = low_bound
        self.up_bound = up_bound

    def forward(self, state, ref):
        # process state and reference differently
        state = torch.tanh(self.states_in(state))
        if self.conv:
            # ref = torch.reshape(ref, (-1, self.ref_dim, 3))
            ref = torch.transpose(ref, 1, 2)
            ref = torch.relu(self.conv_ref(ref))
            ref = torch.reshape(ref, (-1, self.reshape_len))
        else:
            ref = torch.tanh(self.ref_in(ref))
        # concatenate
        x = torch.hstack((state, ref))
        # normal feed-forward
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc_out(x)
        # apply limits
        x = torch.sigmoid(x) * (self.up_bound - self.low_bound) + self.low_bound
        return x