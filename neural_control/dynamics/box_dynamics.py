import torch
import numpy as np
from neural_control.dynamics.quad_dynamics_base import Dynamics
from neural_control.dynamics.quad_dynamics_flightmare import FlightmareDynamics
from neural_control.dynamics.mj_dyn import SimpleRobot, mj_forward_wrapper, rotvec_to_euler_tensor, euler_to_rotvec_tensor
import casadi as ca
import os
import mujoco
import mujoco.viewer
import time

from neural_control.drone_loss import quad_mpc_loss


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
        if self.model.opt.timestep <= 0:
            raise ValueError("Model timestep must be positive.")

        self.kp = torch.tensor([16.6, 16.6, 5.0])
        self.kd = torch.tensor([0.0, 0.0, 0.0])
        box_id = self.model.body("box").id
        self.box_mass = self.model.body_mass[box_id]
        self.box_inertia_diag = self.model.body_inertia[box_id]

        self.ROTVEC = True  # Set to True if the state IN DYNAMICS is in rotation vector format, False if in euler format
        
    def __call__(self, state_tensor, action_tensor, dt):
        """
        Performs multiple simulation steps using a forward Euler integration.

        Args:
            state_tensor (torch.Tensor): The initial state tensor, input and output is in euler format.
            action_tensor (torch.Tensor): It is accelerate(0) and omega_cmd(1,2,3).
            dt (float): The total time duration for the simulation.

        Returns:
            tuple: A tuple containing the final state_tensor and sensor_tensor.
        """

        state_tensor_cpy = state_tensor.clone()
        action_tensor_cpy = action_tensor.clone()
        # action is normalized between 0 and 1 --> rescale
        action_tensor_cpy[:,0] = action_tensor_cpy[:,0] * 15 - 7.5 + 9.81  # Scale thrust to be around 9.81 m/s^2
        action_tensor_cpy[:,1:] = action_tensor_cpy[:,1:] - 0.5  # It is omega_cmd
        
        step_num = max(1, int(dt / self.model.opt.timestep + 1e-9)) # Add epsilon for robustness

        # Perform the simulation steps
        sensor_tensor = None  # Initialize sensor_tensor

        force = torch.zeros_like(action_tensor_cpy)
        force[:,0] = action_tensor_cpy[:,0] * self.box_mass
        force[:,1:] = self.pd_omega_control_per_axis_batch(
            state_tensor_cpy[:, 9:12],  # omega
            action_tensor_cpy[:, 1:],  # omega_cmd
            torch.tensor(self.box_inertia_diag, device=action_tensor_cpy.device),  # J
            self.kp.to(action_tensor_cpy.device),  # Kp
            self.kd.to(action_tensor_cpy.device)   # Kd
        ).to(action_tensor_cpy.device)

        # Convert euler angles to rotation vector
        if self.ROTVEC:
            # Convert euler angles to rotation vector
            state_tensor_cpy = torch.cat([
                state_tensor_cpy[:, :3],
                euler_to_rotvec_tensor(state_tensor_cpy[:, 3:6], eps=1e-6),
                state_tensor_cpy[:, 6:]
            ], dim=1)

        for _ in range(step_num):
            state_tensor_cpy, sensor_tensor = mj_forward_wrapper(
                state_tensor_cpy, force, self.robot, self.ROTVEC
            )
            # Potentially update action_tensor_cpy if actions change over time
            # For a constant action over dt, this is fine.

        state_tensor_next = state_tensor_cpy

        if self.ROTVEC:
            # Convert rotation vector back to euler angles
            state_tensor_next = torch.cat([
                state_tensor_next[:, :3],
                rotvec_to_euler_tensor(state_tensor_cpy[:, 3:6], eps=1e-6),
                state_tensor_next[:, 6:]
            ], dim=1)

        return state_tensor_next
    
    def pd_omega_control_per_axis_batch(self, omega, omega_cmd, J, Kp, Kd):
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
            next_state_tensor_test = test_dyn(state_tensor.float(), action_tensor.float(), 0.1)
            print("error: ", torch.norm(next_state_tensor - next_state_tensor_test))
            print("mujoco state: \n", next_state_tensor)
            print("flightmare state: \n", next_state_tensor_test)
            print()

            state_tensor = next_state_tensor
            # print("forward euler time: ", time.time() - start_time)
            
            viewer.sync()
            
            # 以 model.opt.timestep 作为周期控制步长（若不足则休眠补足）
            elapsed = time.time() - step_start
            time_until_next_step = box_dynamics.model.opt.timestep - elapsed
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    print("Finished simulation.")

