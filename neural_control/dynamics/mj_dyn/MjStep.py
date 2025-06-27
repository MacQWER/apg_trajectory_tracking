import os
import torch
import numpy as np
import mujoco
import mujoco.viewer
import time
import multiprocessing
from scipy.spatial.transform import Rotation as R


def euler_xyz_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles (xyz) to quaternion (w, x, y, z)
    Input shape: (..., 3), Output shape: (..., 4)
    """
    original_shape = euler.shape
    euler = np.atleast_2d(euler)
    rot = R.from_euler('xyz', euler)
    quat_xyzw = rot.as_quat()  # (..., 4) in [x, y, z, w]
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    quat_wxyz = quat_wxyz.reshape(original_shape[:-1] + (4,))
    return quat_wxyz

def quaternion_to_euler_xyz_safe_tensor(quat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Converts quaternion [w,x,y,z] to Euler angles (roll, pitch, yaw),
    with special handling near the gimbal-lock singularity.
    Input shape: (...,4), output (...,3).
    """
    w, x, y, z = torch.unbind(quat, dim=-1)

    # 常规计算 sin,pitch
    sinp = 2 * (w*y - z*x)
    # clamp 防止 |sinp|>1 导致 asin NaN
    sinp_clamped = sinp.clamp(-1+eps, 1-eps)
    pitch = torch.asin(sinp_clamped)

    # 非奇异时的 roll,yaw
    sinr = 2 * (w*x + y*z)
    cosr = 1 - 2 * (x*x + y*y)
    roll_n = torch.atan2(sinr, cosr)

    siny = 2 * (w*z + x*y)
    cosy = 1 - 2 * (y*y + z*z)
    yaw_n  = torch.atan2(siny, cosy)

    # 探测真·奇异：|sinp| 接近 1
    singular = (sinp.abs() > 1 - eps)

    # 如果处于奇异，roll 和 yaw 公式退化：
    #  - roll_s = atan2(-2*(x*z - w*y), 1 - 2*(y*y+z*z))
    #  - yaw_s  = 0  （任意值都可，这里设为 0）
    roll_s = torch.atan2(-2 * (x*z - w*y),
                         1 - 2 * (y*y + z*z))
    yaw_s  = torch.zeros_like(roll_s)

    # 最终根据 mask 选值
    roll = torch.where(singular, roll_s, roll_n)
    yaw  = torch.where(singular, yaw_s,  yaw_n)

    return torch.stack((roll, pitch, yaw), dim=-1)


class MjStepFunction_euler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state_tensor, action_tensor, robot):
        # print("type of state_tensor:", type(state_tensor))
        device = action_tensor.device
        state = state_tensor.cpu().squeeze(0).numpy()
        action = action_tensor.cpu().squeeze(0).numpy()

        state_quat = np.zeros(robot.model.nq + robot.model.nv, dtype=np.float64)
        state_quat[:3] = state[:3]
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
        state_next_euler_tensor[3:6] = quaternion_to_euler_xyz_safe_tensor(state_next_tensor[3:robot.model.nq], eps=1e-6)
        state_next_euler_tensor[6:] = state_next_tensor[robot.model.nq:]

        return state_next_euler_tensor, sensordata_next_tensor

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

        return grad_state, grad_action, None

def mj_forward_euler_wrapper(state, action, robot):
    return MjStepFunction_euler.apply(state, action, robot)

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
            state_tensor, sensor_tensor = mj_forward_euler_wrapper(state_tensor, action_tensor, robot)
            # print("forward euler time: ", time.time() - start_time)
            
            viewer.sync()
            
            # 以 model.opt.timestep 作为周期控制步长（若不足则休眠补足）
            elapsed = time.time() - step_start
            time_until_next_step = model.opt.timestep - elapsed
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == '__main__':
    run_simulation()
