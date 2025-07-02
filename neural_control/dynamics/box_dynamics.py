import torch
import numpy as np
from neural_control.dynamics.quad_dynamics_base import Dynamics
from neural_control.dynamics.quad_dynamics_flightmare import FlightmareDynamics
from neural_control.dynamics.mj_dyn import SimpleRobot, mj_forward_euler_wrapper
import casadi as ca
import os
import mujoco
import mujoco.viewer
import time

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
    return grad_box, grad_flight

def pd_omega_control_per_axis_batch(omega, omega_cmd, J, Kp, Kd):
    """
    Batched PD angular velocity controller.

    :param omega:      (B, 3) current angular velocity
    :param omega_cmd:  (B, 3) desired angular velocity
    :param J:          (3,) or (B, 3) inertia (diagonal)
    :param Kp:         (3,) or (B, 3) proportional gain
    :param Kd:         (3,) or (B, 3) derivative gain
    :return:           (B, 3) torque for each sample
    """
    # Ensure shape
    # omega = torch.as_tensor(omega, dtype=torch.float32)
    # omega_cmd = torch.as_tensor(omega_cmd, dtype=torch.float32)

    # Broadcast J, Kp, Kd to (B, 3) if necessary
    B = omega.shape[0]
    J = torch.as_tensor(J, dtype=torch.float32)
    if J.ndim == 1:
        J = J.unsqueeze(0).repeat(B, 1)
    Kp = torch.as_tensor(Kp, dtype=torch.float32)
    if Kp.ndim == 1:
        Kp = Kp.unsqueeze(0).repeat(B, 1)
    Kd = torch.as_tensor(Kd, dtype=torch.float32)
    if Kd.ndim == 1:
        Kd = Kd.unsqueeze(0).repeat(B, 1)

    # PD control
    omega_error = omega_cmd - omega
    omega_dot_cmd = Kp * omega_error - Kd * omega

    J_omega = J * omega
    cross = torch.cross(omega, J_omega, dim=1)

    torque = J * omega_dot_cmd + cross
    return torque


class BoxDynamics(Dynamics):

    def __init__(self, modified_params={}, simulate_rotors=False):
        super().__init__(modified_params=modified_params)

        self.simulate_rotors = simulate_rotors
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        BOX_XML_PATH = os.path.join(CURRENT_DIR, '.', 'source', 'box', 'box_6d.xml')
        BOX_XML_PATH = os.path.normpath(BOX_XML_PATH)
        self.model = mujoco.MjModel.from_xml_path(BOX_XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.robot = SimpleRobot(self.model, self.data)
        
    def __call__(self, state_tensor, action_tensor, dt):
        """
        Performs multiple simulation steps using a forward Euler integration.

        Args:
            state_tensor (torch.Tensor): The initial state tensor.
            action_tensor (torch.Tensor): The action tensor to apply.
            dt (float): The total time duration for the simulation.

        Returns:
            tuple: A tuple containing the final state_tensor and sensor_tensor.
        """
        # Ensure input tensors are not modified in place if they are used elsewhere
        # .clone() is already a good practice.
        state_tensor_current = state_tensor.clone() 
        action_tensor_current = action_tensor.clone() # accelerate, omega_cmd

        # action is normalized between 0 and 1 --> rescale
        box_id = self.model.body("box").id
        box_mass = self.model.body_mass[box_id]
        action_tensor_current[:,0] = action_tensor_current[:,0] * 15 - 7.5 + 9.81  # Scale thrust to be around 9.81 m/s^2
        box_inertia_diag = self.model.body_inertia[box_id]
        action_tensor_current[:,1:] = action_tensor_current[:,1:] - 0.5  # It is omega_cmd

        # Calculate number of steps. Ensure it's at least 1 if dt > 0.
        # Use a small epsilon to avoid floating point issues when dt is a multiple of timestep.
        if self.model.opt.timestep <= 0:
            raise ValueError("Model timestep must be positive.")
        
        step_num = max(1, int(dt / self.model.opt.timestep + 1e-9)) # Add epsilon for robustness

        # Perform the simulation steps
        sensor_tensor = None  # Initialize sensor_tensor

        for _ in range(step_num):
            force = torch.zeros_like(action_tensor_current)
            force[:,0] = action_tensor_current[:,0] * box_mass
            force[:,1:] = pd_omega_control_per_axis_batch(
                state_tensor_current[:, 9:12],  # omega
                action_tensor_current[:, 1:],  # omega_cmd
                torch.tensor(box_inertia_diag, device=action_tensor_current.device),  # J
                torch.tensor([16.6, 16.6, 5.0], device=action_tensor_current.device),  # Kp
                torch.tensor([0.0, 0.0, 0.0], device=action_tensor_current.device)   # Kd
            ).to(action_tensor_current.device)
            state_tensor_current, sensor_tensor = mj_forward_euler_wrapper(
                state_tensor_current, force, self.robot
            )
            # Potentially update action_tensor_current if actions change over time
            # For a constant action over dt, this is fine.

        return state_tensor_current





if __name__ == "__main__":
    box_dynamics = BoxDynamics()
    simulation_duration=60.0
    control_decimation=1
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")  # For simplicity, using CPU here

    state_tensor = torch.zeros((1, 2 * box_dynamics.model.nv), dtype=torch.double, device=device)
    state_tensor[:,:box_dynamics.model.nv] = torch.tensor([0, 0, .3, 0, 0, 0], dtype=torch.double, device=device)
    action_tensor = torch.zeros(box_dynamics.model.nu, dtype=torch.double, device=device)

    test_dyn = FlightmareDynamics()

    grad_box, grad_flight = compare_gradient_difference(box_dynamics, test_dyn)
    print("Box Dynamics Gradient: \n", grad_box)
    print("Flightmare Dynamics Gradient: \n", grad_flight)

    with mujoco.viewer.launch_passive(box_dynamics.model, box_dynamics.data) as viewer:
        # viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1

        start = time.time()
        counter = 0
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            if counter % control_decimation == 0:
                    # action_tensor = torch.zeros(action_dim, dtype=torch.double, device=device)
                    action_tensor = torch.tensor([[0.5, 0.5, 0.5, 0.7]], dtype=torch.double, device=device)
            counter += 1
            
            start_time = time.time()
            next_state_tensor = box_dynamics(state_tensor, action_tensor, 0.1)
            # next_state_tensor_test = test_dyn(state_tensor.float(), action_tensor.float(), 0.1)
            # print("error: ", torch.norm(next_state_tensor - next_state_tensor_test))
            # print("mujoco state: \n", next_state_tensor)
            # print("flightmare state: \n", next_state_tensor_test)
            # print("action: ", action_tensor)
            # print()

            state_tensor = next_state_tensor
            print("forward euler time: ", time.time() - start_time)
            
            viewer.sync()
            
            # 以 model.opt.timestep 作为周期控制步长（若不足则休眠补足）
            elapsed = time.time() - step_start
            time_until_next_step = box_dynamics.model.opt.timestep - elapsed
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    print("Finished simulation.")

