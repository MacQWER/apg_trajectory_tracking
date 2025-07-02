import os
import torch
import numpy as np
import mujoco
import mujoco.viewer
import time
import multiprocessing
from scipy.spatial.transform import Rotation as R

from neural_control.dynamics.mj_dyn.quad_math import euler_xyz_to_quaternion, quaternion_to_euler_xyz_safe_tensor, rotvec_to_quaternion, quaternion_to_rotvec_tensor

# Set to True if the state is in rotation vector format, False if in quaternion format


class MjStepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state_tensor, action_tensor, robot, ROTVEC=True):
        # print("type of state_tensor:", type(state_tensor))
        device = action_tensor.device
        origin_shape = state_tensor.shape
        state = state_tensor.cpu().squeeze(0).numpy()
        action = action_tensor.cpu().squeeze(0).numpy()

        state_quat = np.zeros(robot.model.nq + robot.model.nv, dtype=np.float64)
        state_quat[:3] = state[:3]
        if ROTVEC:
            # Convert rotation vector to quaternion
            state_quat[3:robot.model.nq] = rotvec_to_quaternion(state[3:6])
        else:
            # Convert Euler angles to quaternion
            state_quat[3:robot.model.nq] = euler_xyz_to_quaternion(state[3:6])
        state_quat[robot.model.nq:] = state[6:]

        obs_tensor, rewards, done, info = robot.step(state_quat, action)
        
        state_next_tensor      = torch.tensor(info["state"], device=device, dtype=torch.double)
        sensordata_next_tensor = torch.tensor(info["sensordata"], device=device, dtype=torch.double)

        state_jacobian_x_tensor  = torch.tensor(info["state_jacobian_x"], device=device, dtype=torch.double)
        state_jacobian_u_tensor  = torch.tensor(info["state_jacobian_u"], device=device, dtype=torch.double)
        sensor_jacobian_x_tensor = torch.tensor(info["sensor_jacobian_x"], device=device, dtype=torch.double)
        sensor_jacobian_u_tensor = torch.tensor(info["sensor_jacobian_u"], device=device, dtype=torch.double)
        
        ctx.save_for_backward(state_jacobian_x_tensor, state_jacobian_u_tensor,
                              sensor_jacobian_x_tensor, sensor_jacobian_u_tensor
                              )
        
        # Convert state_next_tensor back to rotvec format
        state_next_euler_tensor = torch.zeros(robot.model.nq + robot.model.nv - 1, dtype=torch.double, device=device)
        state_next_euler_tensor[:3] = state_next_tensor[:3]  # Position
        if ROTVEC:
            # Convert quaternion back to rotation vector
            # Ensure the quaternion is in [w, x, y, z] format
            state_next_euler_tensor[3:6] = quaternion_to_rotvec_tensor(state_next_tensor[3:robot.model.nq])
        else:
            # Convert quaternion back to Euler angles
            # Ensure the quaternion is in [w, x, y, z] format
            state_next_euler_tensor[3:6] = quaternion_to_euler_xyz_safe_tensor(state_next_tensor[3:robot.model.nq], eps=1e-6)
        state_next_euler_tensor[6:] = state_next_tensor[robot.model.nq:]

        return state_next_euler_tensor.reshape(origin_shape), sensordata_next_tensor

    @staticmethod
    def backward(ctx, grad_state_next, grad_sensordata_next):
        state_jacobian_x_tensor  = ctx.saved_tensors[0]
        state_jacobian_u_tensor  = ctx.saved_tensors[1]
        sensor_jacobian_x_tensor = ctx.saved_tensors[2]
        sensor_jacobian_u_tensor = ctx.saved_tensors[3]
        
        grad_state  = torch.matmul(grad_state_next, state_jacobian_x_tensor) \
                    + torch.matmul(grad_sensordata_next, sensor_jacobian_x_tensor)      
        grad_action = torch.matmul(grad_state_next, state_jacobian_u_tensor) \
                    + torch.matmul(grad_sensordata_next, sensor_jacobian_u_tensor)

        return grad_state, grad_action, None, None

def mj_forward_wrapper(state, action, robot, ROTVEC):
    return MjStepFunction.apply(state, action, robot, ROTVEC)

class SimpleRobot:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def step(self, state, action):
        self.data.qpos[:] = state[:self.model.nq]
        self.data.qvel[:] = state[self.model.nq:]

        self.data.ctrl = action

        mujoco.mj_step(self.model, self.data)

        state_jacobian_x = np.ascontiguousarray(np.zeros((2*self.model.nv + self.model.na, 2*self.model.nv + self.model.na), dtype=np.float64))
        state_jacobian_u = np.ascontiguousarray(np.zeros((2*self.model.nv+self.model.na,self.model.nu), dtype=np.float64))
        sensor_jacobian_x = np.ascontiguousarray(np.zeros((self.model.nsensordata, 2*self.model.nv + self.model.na), dtype=np.float64))
        sensor_jacobian_u = np.ascontiguousarray(np.zeros((self.model.nsensordata, self.model.nu), dtype=np.float64))

        mujoco.mjd_transitionFD(self.model, self.data, float(0.00001), int(1), 
                                state_jacobian_x, state_jacobian_u, sensor_jacobian_x, sensor_jacobian_u)
            
        state_next = np.concatenate([self.data.qpos, self.data.qvel], axis=0)

        return state_next, 0.0, False, {
            "state": np.concatenate([self.data.qpos, self.data.qvel], axis=0),
            "sensordata": self.data.sensordata.copy(),
            "state_jacobian_x": state_jacobian_x,
            "state_jacobian_u": state_jacobian_u,
            "sensor_jacobian_x": sensor_jacobian_x,
            "sensor_jacobian_u": sensor_jacobian_u
        }
    

def run_simulation(simulation_duration=60.0, control_decimation=20):
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BOX_XML_PATH = os.path.join(CURRENT_DIR, '..', 'source', 'box', 'box_6d.xml')
    BOX_XML_PATH = os.path.normpath(BOX_XML_PATH)
    model = mujoco.MjModel.from_xml_path(BOX_XML_PATH)
    data = mujoco.MjData(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robot_state_dim = model.nq + model.nv  
    action_dim = model.nu
    hidden_dim = 64
    robot = SimpleRobot(model=model, data=data)

    print("Model is ready, starting simulation...")
    
    state_tensor = torch.zeros(2 * model.nv, dtype=torch.double, device=device)
    state_tensor[:model.nv] = torch.tensor([0, 0, .3, 0, 0, 0], dtype=torch.double, device=device)
    action_tensor = torch.zeros(action_dim, dtype=torch.double, device=device)
    print("initial state quat: ", state_tensor)
    print("nq: ", model.nq)
    print("nv: ", model.nv)
    print("na: ", model.na)
    print("nu: ", model.nu)
    print("nsensordata: ", model.nsensordata)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
        print("Starting viewing...")

        start = time.time()
        counter = 0
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            if counter % control_decimation == 0:
                    # action_tensor = torch.zeros(action_dim, dtype=torch.double, device=device)
                    action_tensor = torch.tensor([0.723 * 9.81, 0., 0., 0.], dtype=torch.double, device=device)
            counter += 1
            
            start_time = time.time()
            state_tensor, sensor_tensor = mj_forward_wrapper(state_tensor, action_tensor, robot)
            # print("forward euler time: ", time.time() - start_time)
            
            viewer.sync()
            
            # 以 model.opt.timestep 作为周期控制步长（若不足则休眠补足）
            elapsed = time.time() - step_start
            time_until_next_step = model.opt.timestep - elapsed
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == '__main__':
    run_simulation()
