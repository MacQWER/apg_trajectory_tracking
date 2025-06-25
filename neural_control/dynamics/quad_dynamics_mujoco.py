import os
import torch
import numpy as np
import mujoco
from wrapper import SimpleRobot, mj_forward_euler_batched
from neural_control.dynamics.quad_dynamics_base import Dynamics

class FlightmareDynamicsMujoco(Dynamics):

    def __init__(self, modified_params={}):
        super().__init__(modified_params=modified_params)

        path = "source/drone/scene.xml"
        self.model = mujoco.MjModel.from_xml_path(path)
        self.robot = SimpleRobot(self.model)
        self.u_low = 0.0
        self.u_high = 13.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

    def __call__(self, state, action, dt):
        step_num = int(dt / self.model.opt.timestep)
        if step_num < 1:
            raise ValueError("dt must be greater than or equal to the model timestep.")
        else:
            action = torch.sigmoid(action) * (self.u_high - self.u_low) + self.u_low
            action.to(self.device)
            state_next, sensordata_next = mj_forward_euler_batched(
                state, action, self.model
            )
            for _ in range(step_num - 1):
                state_next, sensordata_next = mj_forward_euler_batched(
                    state_next, action, self.model
                )
            return state_next
        
        


    
