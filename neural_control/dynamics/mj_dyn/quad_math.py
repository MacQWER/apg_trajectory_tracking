import torch
import numpy as np
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

def rotvec_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """
    Converts a rotation vector (axis-angle) to a quaternion.
    Input shape: (..., 3)
    Output shape: (..., 4), quaternion format [w, x, y, z]
    """
    theta = np.linalg.norm(euler, axis=-1, keepdims=True)  # (..., 1)
    half_theta = theta / 2.0
    small_angle = theta < 1e-8  # (..., 1)

    axis = np.zeros_like(euler)  # (..., 3)

    # Only index and divide where not small angle
    if np.any(~small_angle):
        euler_non_small = euler[~small_angle.squeeze(-1)]        # shape: (?, 3)
        theta_non_small = theta[~small_angle]                    # shape: (?, 1)
        axis[~small_angle.squeeze(-1)] = euler_non_small / theta_non_small  # safe division

    sin_half_theta = np.sin(half_theta)  # (..., 1)
    w = np.cos(half_theta)[..., 0]       # (...,)
    xyz = axis * sin_half_theta          # (..., 3)
    quat = np.concatenate([w[..., np.newaxis], xyz], axis=-1)  # (..., 4)
    return quat

def quaternion_to_rotvec_tensor(quat: torch.Tensor) -> torch.Tensor:
    """
    Converts a quaternion [w, x, y, z] to a rotation vector (axis-angle).
    Input shape: (..., 4)
    Output shape: (..., 3)
    """
    q = quat / quat.norm(dim=-1, keepdim=True)  # normalize
    w, x, y, z = q.unbind(-1)
    angle = 2.0 * torch.acos(torch.clamp(w, -1.0, 1.0))  # in [0, π]
    sin_half_angle = torch.sqrt(1.0 - w * w)

    small_angle = sin_half_angle < 1e-8
    axis = torch.stack([x, y, z], dim=-1)
    axis = torch.where(small_angle.unsqueeze(-1), torch.zeros_like(axis), axis / sin_half_angle.unsqueeze(-1))
    rotvec = angle.unsqueeze(-1) * axis
    return rotvec

def euler_to_quaternion_tensor(euler: torch.Tensor) -> torch.Tensor:
    """
    Convert batched Euler angles (roll, pitch, yaw) in radians to quaternion [w, x, y, z]
    Input: (..., 3)
    Output: (..., 4)
    """
    roll, pitch, yaw = euler.unbind(-1)

    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, y, z], dim=-1)


def quaternion_to_euler_tensor(quat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert batched quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw)
    Input: (..., 4)
    Output: (..., 3)
    """
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=eps)
    w, x, y, z = quat.unbind(-1)

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0 + eps, 1.0 - eps))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)

def euler_to_rotvec_tensor(euler: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert batched Euler angles (..., 3) to rotation vectors (..., 3)
    Input: (..., 3)
    Output: (..., 3)
    """
    quat = euler_to_quaternion_tensor(euler)  # (..., 4)
    w, x, y, z = quat.unbind(-1)
    theta = 2.0 * torch.acos(torch.clamp(w, -1.0, 1.0))  # (...,)

    sin_half_theta = torch.sqrt(1.0 - w * w + eps)  # (...,)
    axis = torch.stack([x, y, z], dim=-1) / sin_half_theta.unsqueeze(-1)  # (..., 3)

    rotvec = theta.unsqueeze(-1) * axis  # (..., 3)

    # Replace NaNs or unstable values when theta is very small
    rotvec = torch.where(theta.unsqueeze(-1) < 1e-6, torch.zeros_like(rotvec), rotvec)
    return rotvec


def rotvec_to_euler_tensor(rotvec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert batched rotation vectors (..., 3) to Euler angles (..., 3)
    Input: (..., 3)
    Output: (..., 3)
    """
    theta = torch.norm(rotvec, dim=-1, keepdim=True)  # (..., 1)
    axis = rotvec / (theta + eps)  # (..., 3)

    half_theta = theta * 0.5
    sin_half_theta = torch.sin(half_theta)
    cos_half_theta = torch.cos(half_theta)

    # Quaternion: [w, x, y, z]
    quat = torch.cat([cos_half_theta, axis * sin_half_theta], dim=-1)  # (..., 4)

    return quaternion_to_euler_tensor(quat)