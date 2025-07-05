import torch
import numpy as np
from neural_control.dynamics.quad_dynamics_base import Dynamics
from neural_control.dynamics.quad_dynamics_flightmare import FlightmareDynamics
from neural_control.dynamics.box_dynamics import BoxDynamics
from neural_control.dynamics.mj_dyn import SimpleRobot, mj_forward_wrapper
import casadi as ca
import os
import mujoco
import mujoco.viewer
import time
from torch.autograd import gradcheck

from neural_control.drone_loss import quad_mpc_loss

def compare_gradient_difference(box_dynamics, test_dyn, horizon=10, dt=0.1, device="cpu"):
    B = 1
    state_dim = 12
    act_dim = 4

    # 初始状态
    init_state = torch.zeros(B, state_dim, dtype=torch.float32, device=device)
    init_state[:, 2] = 0.3  # 初始 z 高度

    # 给一个合理参考轨迹（比如保持 hover）
    ref_states = torch.zeros(horizon, B, state_dim, device=device)
    ref_states[..., 2] = 0.3  # 保持高度不变

    # action_seq 初始化，requires_grad
    # action_seq_box = torch.full((horizon, B, act_dim), 0.5, requires_grad=True, device=device)
    t = torch.arange(horizon, device=device)
    sin_wave = 0.5 + 0.4 * torch.sin(2 * 3.1415 * t / horizon)
    action_seq_box = sin_wave.view(horizon, 1, 1).repeat(1, B, act_dim)
    action_seq_box = action_seq_box + 0.05 * torch.randn(horizon, B, act_dim, device=device)
    action_seq_box = action_seq_box.clamp(0, 1).requires_grad_()
    action_seq_flight = action_seq_box.clone().detach().requires_grad_()

    # simulate box_dynamics
    states_box = [init_state]
    state = init_state.clone()
    for t in range(horizon):
        action = action_seq_box[t]
        state = box_dynamics(state.double(), action.double(), dt).float()
        states_box.append(state)
    states_box = torch.stack(states_box[1:], dim=0)

    # simulate flightmare
    states_flight = [init_state]
    state = init_state.clone()
    for t in range(horizon):
        action = action_seq_flight[t]
        state = test_dyn(state, action, dt)
        states_flight.append(state)
    states_flight = torch.stack(states_flight[1:], dim=0)

    # compute losses
    loss_box = quad_mpc_loss(states_box, ref_states, action_seq_box)
    loss_flight = quad_mpc_loss(states_flight, ref_states, action_seq_flight)

    # backward
    loss_box.backward()
    grad_box = action_seq_box.grad.clone()

    loss_flight.backward()
    grad_flight = action_seq_flight.grad.clone()

    # compare
    grad_diff = torch.norm(grad_box - grad_flight).item()
    state_diff = torch.norm(states_box - states_flight).item()
    print(f"Gradient L2 difference: {grad_diff:.6f}")
    print(f"State L2 difference: {state_diff:.6f}")
    return grad_box, grad_flight, states_box, states_flight

if __name__ == "__main__":
    box_dynamics = BoxDynamics()
    test_dyn = FlightmareDynamics()

    # 设置设备
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")  # For simplicity, using CPU here

    # 比较梯度差异
    grad_box, grad_flight, states_box, states_flight = compare_gradient_difference(box_dynamics, test_dyn, horizon=10, dt=0.1, device=device)
    
    print("Box Dynamics Gradient: \n", grad_box)
    print("Flightmare Dynamics Gradient: \n", grad_flight)
    print("Box Dynamics States: \n", states_box)
    print("Flightmare Dynamics States: \n", states_flight)

    def check_dynamics_gradients(state_tensor, action_tensor):
    # dt is fixed for the gradient check
        dt = 0.02 
        return box_dynamics(state_tensor, action_tensor, dt)
    
    batch_size = 1
    dummy_state  = torch.randn(batch_size, 12, dtype=torch.double, requires_grad=True)
    dummy_action = torch.randn(batch_size,  4, dtype=torch.double, requires_grad=True)
    print()
    print("Performing gradient check for BoxDynamics...")
    try:
        # `eps` is the step size for numerical approximation
        # `atol` is the absolute tolerance
        # `rtol` is the relative tolerance
        test = gradcheck(check_dynamics_gradients, (dummy_state, dummy_action), eps=2e-2, atol=1e-1, rtol=1e-1)
        print(f"Gradient check passed: {test}")
    except Exception as e:
        print(f"Gradient check failed: {e}")
        print("This might indicate an issue in the backward pass of MockMjForwardWrapper or other differentiable operations.")
