import os
import torch
import numpy as np
import mujoco
import multiprocessing
from Util import euler_xyz_to_quaternion, quaternion_to_euler_xyz_safe_tensor


class SimpleRobot:
    def __init__(self, model):
        # 每个进程独立构建自己的 data
        self.model = model
        self.data = mujoco.MjData(model)

    def step(self, state, action_array):
        # state: numpy array (nq+nv,)
        self.data.qpos[:] = state[:self.model.nq]
        self.data.qvel[:] = state[self.model.nq:]
        self.data.ctrl[:] = action_array
        mujoco.mj_step(self.model, self.data)

        # 构建雅可比矩阵缓冲区
        D = 2*self.model.nv + self.model.na
        state_jac_x = np.zeros((D, D), dtype=np.float64)
        state_jac_u = np.zeros((D, self.model.nu), dtype=np.float64)
        sens_jac_x = np.zeros((self.model.nsensordata, D), dtype=np.float64)
        sens_jac_u = np.zeros((self.model.nsensordata, self.model.nu), dtype=np.float64)
        mujoco.mjd_transitionFD(self.model, self.data, 1e-5, 1,
                                state_jac_x, state_jac_u,
                                sens_jac_x,  sens_jac_u)

        state_next = np.concatenate([self.data.qpos, self.data.qvel], axis=0)
        return state_next, self.data.sensordata.copy(), state_jac_x, state_jac_u, sens_jac_x, sens_jac_u

# 全局 worker，用于进程池
def _step_worker(args):
    model, state, action = args
    robot = SimpleRobot(model)
    return robot.step(state, action)

class MjStepFunctionEulerBatched(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state_tensor, action_tensor, model):
        # state_tensor: (B, state_dim), action_tensor: (B, action_dim)
        B = state_tensor.shape[0]
        device = action_tensor.device
        # 转 numpy
        states_np = state_tensor.detach().cpu().double().numpy()
        state_quats = np.zeros((B, model.nq + model.nv), dtype=np.float64)
        state_quats[:, :3] = states_np[:, :3]  # position
        state_quats[:, 3:model.nq] = euler_xyz_to_quaternion(states_np[:, 3:6])
        state_quats[:, model.nq:] = states_np[:, 6:]

        # action_tensor = torch.sigmoid(action_tensor) * 13.0  # 确保 action 在 [0, 13] 范围内
        actions_np = action_tensor.detach().cpu().double().numpy()

        # 多进程并行
        args_list = [(model, state_quats[i], actions_np[i].squeeze()) for i in range(B)]
        with multiprocessing.Pool(processes=min(B, multiprocessing.cpu_count())) as pool:
            results = pool.map(_step_worker, args_list)

        # 解包结果
        st_next_list, sens_list, jac_x_list, jac_u_list, s_jac_x_list, s_jac_u_list = zip(*results)
        # 转回 tensor
        # 下一状态和 sensordata
        st_next = torch.tensor(np.stack(st_next_list), device=device, dtype=torch.double)
        sens_next = torch.tensor(np.stack(sens_list), device=device, dtype=torch.double)
        # 雅可比
        jx = torch.tensor(np.stack(jac_x_list), device=device, dtype=torch.double)
        ju = torch.tensor(np.stack(jac_u_list), device=device, dtype=torch.double)
        sx = torch.tensor(np.stack(s_jac_x_list), device=device, dtype=torch.double)
        su = torch.tensor(np.stack(s_jac_u_list), device=device, dtype=torch.double)

        # 保存用于 backward
        ctx.save_for_backward(jx, ju, sx, su)

        # 转回 euler 形式
        out_states = torch.zeros((B, model.nq + model.nv - 1), device=device, dtype=torch.double)
        # position
        out_states[:, :3] = st_next[:, :3]
        # orientation
        out_states[:, 3:6] = quaternion_to_euler_xyz_safe_tensor(st_next[:, 3:model.nq], eps=1e-6)
        # velocity
        out_states[:, 6:] = st_next[:, model.nq:]

        return out_states, sens_next

    @staticmethod
    def backward(ctx, grad_out_states, grad_sens):
        jx, ju, sx, su = ctx.saved_tensors
        # batched matmul: (B,dim) x (B,dim,dim) -> (B,dim)
        grad_state = torch.einsum('bd, bde -> be', grad_out_states, jx) \
                   + torch.einsum('bd, bde -> be', grad_sens, sx)
        grad_action = torch.einsum('bd, bde -> be', grad_out_states, ju) \
                    + torch.einsum('bd, bde -> be', grad_sens, su)
        return grad_state, grad_action, None

# 使用接口
def mj_forward_euler_batched(state, action, model):
    return MjStepFunctionEulerBatched.apply(state, action, model)