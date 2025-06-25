import torch
import torch.nn.functional as F
import numpy as np

from .quat_math import quat_mul, rotvec_to_quat_tensor

hover_action = torch.tensor([3.2495625, 3.2495625, 3.2495625, 3.2495625], dtype=torch.float64)

def quad_loss(states, ref_states, action_seq) -> torch.Tensor:
    pos_factor = 10
    u_factor = 1
    av_factor = 0.1
    vel_factor = 1

    position_loss = torch.sum((states[:, :, :3] - ref_states[:, :, :3])**2)
    velocity_loss = torch.sum((states[:, :, 6:9] - ref_states[:, :, 6:9])**2)

    av_loss = torch.sum(states[:, :, 9:12]**2) # omega^2
    u_loss = torch.sum((action_seq[:, :, :] - hover_action.to(action_seq.device))**2)

    loss = (
        pos_factor * position_loss + vel_factor * velocity_loss +
        av_factor * av_loss + u_factor * u_loss
    )
    return loss

def compute_loss(state_cur: torch.Tensor,
                 state_tar:   torch.Tensor,
                 rot_weight: float    = 1.0,
                 vel_weight: float    = 0.0,
                 eps:        float    = 1e-8) -> torch.Tensor:
    """
    计算 pose loss，只对平移 + 姿态（rotvec）求 loss，忽略速度部分。

    参数：
      state_cur:        Tensor[..., D], 模型输出的 state 向量，最后一维 D>=6 (6 pose + 其它如速度)
      state_tar:          Tensor[..., D], 对应的 ground truth 向量，格式同 state_cur
      rot_weight:  float, 姿态 loss 的权重
      vel_weight:  float, 速度 loss 的权重（默认 0，表示忽略速度）
      eps:         float, 防止除零的小常数

    假设：
      - state_cur[..., :3] 和 state_tar[..., :3] 是平移 (x,y,z)
      - state_cur[..., 3:6] 和 state_tar[..., 3:6] 是旋转向量 (axis-angle)
      - state_cur[..., 6:] 和 state_tar[..., 6:] 是速度等附加量

    返回：
      标量 Tensor，即各样本（或单样本）loss 的均值
    """
    # 提取平移和旋转向量
    t_state_cur = state_cur[..., :3]
    t_state_tar   = state_tar[...,   :3]
    r_state_cur = state_cur[..., 3:6]
    r_state_tar   = state_tar[...,   3:6]

    # 平移 MSE (sum over xyz)
    # 结果 shape = (...)，之后在最后做 mean
    loss_t = F.mse_loss(t_state_cur, t_state_tar, reduction='none').sum(dim=-1)

    # 姿态：rotvec -> quat
    q_state_cur = rotvec_to_quat_tensor(r_state_cur, eps=eps)   # (...,4)
    q_state_tar   = rotvec_to_quat_tensor(r_state_tar,   eps=eps)   # (...,4)

    # 相对四元数 q_rel = q_state_cur^{-1} * q_state_tar
    # q_state_cur^{-1} = conj(q_state_cur) （假设单位四元数）
    q_conj = q_state_cur * torch.tensor([1.0, -1.0, -1.0, -1.0],
                                   device=q_state_cur.device)
    q_rel  = quat_mul(q_conj, q_state_tar)  # (...,4)

    # geodesic angle = 2 * atan2(||v||, w)
    w, v = q_rel.unbind(-1)[0], torch.stack(q_rel.unbind(-1)[1:], dim=-1)
    angle = 2.0 * torch.atan2(v.norm(dim=-1), w.clamp(-1 + 1e-7, 1 - 1e-7))
    loss_r = angle.pow(2)            # (...,)

    # 可选：速度 loss
    if vel_weight != 0.0 and state_cur.shape[-1] > 6:
        vel_state_cur = state_cur[..., 6:]
        vel_state_tar   = state_tar[...,   6:]
        loss_v   = F.mse_loss(vel_state_cur, vel_state_tar, reduction='none').sum(dim=-1)
    else:
        loss_v = 0.0

    # 加权求和并对所有样本求平均
    total = loss_t + rot_weight * loss_r + vel_weight * loss_v
    return total.mean()

if __name__ == "__main__":
    # 测试 compute_loss 函数
    # 单样本（no batch）
    state_cur = torch.randn(12, requires_grad=True)
    state_tar   = torch.zeros(12)
    loss = compute_loss(state_cur, state_tar, rot_weight=0.5)   # 只计算 pose 部分
    loss.backward()
    print(f"state_cur: {state_cur}")
    print(f"state_tar: {state_tar}")
    print(f"Single sample loss: {loss.item()}")
    print(f"Gradient: {state_cur.grad}")

    # 带 batch
    state_curs = torch.randn(32, 12, requires_grad=True)
    state_tars   = torch.zeros(32, 12)
    loss  = compute_loss(state_curs, state_tars, rot_weight=1.0)
    loss.backward()
    print(f"state_curs: {state_curs}")
    print(f"state_tars: {state_tars}")
    print(f"Batch samples loss: {loss.item()}")
    print(f"Gradient: {state_curs.grad}")