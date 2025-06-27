import torch
import numpy as np
from neural_control.plotting import print_state_ref_div
device = "cpu"
torch.autograd.set_detect_anomaly(True)
zero_tensor = torch.zeros(3).to(device)

rates_prior = torch.tensor([.5, .5, .5])


def quad_mpc_loss(states, ref_states, action_seq, printout=0):
    # MATCH TO MPC
    # self._Q_u = np.diag([50, 1, 1, 1])
    # self._Q_pen = np.diag([100, 100, 100, 10, 10, 10, 10, 10, 10, 1, 1, 1])
    pos_factor = 10
    u_thrust_factor = 5
    u_rates_factor = 0.1
    av_factor = 0.1
    vel_factor = 1

    position_loss = torch.sum((states[:, :, :3] - ref_states[:, :, :3])**2)
    velocity_loss = torch.sum((states[:, :, 6:9] - ref_states[:, :, 6:9])**2)

    av_loss = torch.sum(states[:, :, 9:12]**2)
    u_thrust_loss = torch.sum((action_seq[:, :, 0] - .5)**2)
    u_rates_loss = torch.sum((action_seq[:, :, 1:] - rates_prior)**2)

    loss = (
        pos_factor * position_loss + vel_factor * velocity_loss +
        av_factor * av_loss + u_rates_factor * u_rates_loss +
        u_thrust_factor * u_thrust_loss
    )

    if printout:
        print_state_ref_div(
            states[0].detach().numpy(), ref_states[0].detach().numpy()
        )
    return loss


def quad_loss_last(states, last_ref_state, action_seq, printout=0):
    angvel_factor = 2e-2
    vel_factor = 0.1
    pos_factor = 10
    yaw_factor = 10
    action_factor = .1

    action_loss = torch.sum((action_seq[:, :, 0] - .5)**2)

    position_loss = torch.sum((states[:, -1, :3] - last_ref_state[:, :3])**2)
    velocity_loss = torch.sum((states[:, -1, 6:9] - last_ref_state[:, 6:9])**2)

    ang_vel_error = torch.sum(states[:, :, 9:11]**2
                              ) + yaw_factor * torch.sum(states[:, :, 11]**2)
    # TODO: do on all intermediate states again?

    loss = (
        angvel_factor * ang_vel_error + pos_factor * position_loss +
        vel_factor * velocity_loss + action_factor * action_loss
    )
    if printout:
        print_state_ref_div(
            states[0].detach().numpy(), last_ref_state[0].detach().numpy()
        )
    return loss

