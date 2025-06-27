import torch
import numpy as np
from neural_control.dynamics.quad_dynamics_base import Dynamics
from mj_dyn import SimpleRobot, mj_forward_euler_wrapper
import casadi as ca
import os
import mujoco
import mujoco.viewer
import time


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
        action_tensor_current = action_tensor.clone() # Cloned for consistency, though not strictly necessary if not modified inside loop

        # Calculate number of steps. Ensure it's at least 1 if dt > 0.
        # Use a small epsilon to avoid floating point issues when dt is a multiple of timestep.
        if self.model.opt.timestep <= 0:
            raise ValueError("Model timestep must be positive.")
        
        step_num = max(1, int(dt / self.model.opt.timestep + 1e-9)) # Add epsilon for robustness

        # Perform the simulation steps
        sensor_tensor = None  # Initialize sensor_tensor

        for _ in range(step_num):
            state_tensor_current, sensor_tensor = mj_forward_euler_wrapper(
                state_tensor_current, action_tensor_current, self.robot
            )
            # Potentially update action_tensor_current if actions change over time
            # For a constant action over dt, this is fine.

        return state_tensor_current, sensor_tensor





if __name__ == "__main__":
    box_dynamics = BoxDynamics()
    simulation_duration=60.0
    control_decimation=20
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")  # For simplicity, using CPU here

    state_tensor = torch.zeros(2 * box_dynamics.model.nv, dtype=torch.double, device=device)
    state_tensor[:box_dynamics.model.nv] = torch.tensor([0, 0, .3, 0, 0, 0], dtype=torch.double, device=device)
    action_tensor = torch.zeros(box_dynamics.model.nu, dtype=torch.double, device=device)

    with mujoco.viewer.launch_passive(box_dynamics.model, box_dynamics.data) as viewer:
        # viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1

        start = time.time()
        counter = 0
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            if counter % control_decimation == 0:
                    # action_tensor = torch.zeros(action_dim, dtype=torch.double, device=device)
                    action_tensor = torch.tensor([0.723 * 9.81, 0., 0., 0.1], dtype=torch.double, device=device)
            counter += 1
            
            start_time = time.time()
            state_tensor, sensor_tensor = box_dynamics(state_tensor, action_tensor, 0.1)
            # print("forward euler time: ", time.time() - start_time)
            
            viewer.sync()
            
            # 以 model.opt.timestep 作为周期控制步长（若不足则休眠补足）
            elapsed = time.time() - step_start
            time_until_next_step = box_dynamics.model.opt.timestep - elapsed
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    print("Finished simulation.")
    
