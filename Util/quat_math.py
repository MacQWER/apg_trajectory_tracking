import mujoco
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.transform import Rotation as R

def quat_to_rotvec(q, eps=1e-6):
    """
    Convert quaternion to 3D rotation vector (log map)
    """
    q = q / np.linalg.norm(q)
    w, xyz = q[0], q[1:]
    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))
    s = np.sqrt(1 - w**2)

    if s < eps:
        return xyz * 2  # 小角度近似
    else:
        axis = xyz / s
        return axis * angle
    
def rotvec_to_quat(rotvec, eps=1e-6):
    """
    Convert 3D rotation vector back to quaternion (exp map)
    """
    angle = np.linalg.norm(rotvec)
    if angle < eps:
        return np.array([1.0, 0.5 * rotvec[0], 0.5 * rotvec[1], 0.5 * rotvec[2]])
    axis = rotvec / angle
    half_angle = angle / 2
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    return np.concatenate(([w], xyz))

def euler_xyz_to_quaternion(euler_angles):
    """
    Converts Euler angles (roll, pitch, yaw) to quaternion [w, x, y, z].
    Input: numpy array of shape (..., 3)
    Output: numpy array of shape (..., 4)
    """
    # Ensure input is array
    angles = np.array(euler_angles, dtype=float)
    roll = angles[..., 0]
    pitch = angles[..., 1]
    yaw = angles[..., 2]

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.stack((w, x, y, z), axis=-1)


def quaternion_to_euler_xyz(quat):
    """
    Converts quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw).
    Input: numpy array of shape (..., 4)
    Output: numpy array of shape (..., 3)
    """
    q = np.array(quat, dtype=float)
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.where(
        np.abs(sinp) >= 1,
        np.sign(sinp) * (np.pi / 2),
        np.arcsin(sinp)
    )

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack((roll, pitch, yaw), axis=-1)


def quaternion_to_euler_xyz_safe(quat, eps=1e-6):
    """
    Converts quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw),
    with special handling near the gimbal-lock singularity.
    Input: numpy array of shape (..., 4)
    Output: numpy array of shape (..., 3)
    """
    q = np.array(quat, dtype=float)
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    # compute sin of pitch
    sinp = 2 * (w * y - z * x)
    sinp_clamped = np.clip(sinp, -1 + eps, 1 - eps)
    pitch = np.arcsin(sinp_clamped)

    # normal roll and yaw
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll_n = np.arctan2(sinr, cosr)

    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw_n = np.arctan2(siny, cosy)

    # detect singularity
    singular = np.abs(sinp) > (1 - eps)

    # singular formulas
    roll_s = np.arctan2(-2 * (x * z - w * y), 1 - 2 * (y * y + z * z))
    yaw_s = np.zeros_like(roll_s)

    # choose between normal and singular
    roll = np.where(singular, roll_s, roll_n)
    yaw  = np.where(singular, yaw_s,  yaw_n)

    return np.stack((roll, pitch, yaw), axis=-1)


#################################### torch.tensor version ####################################

def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

def rotvec_to_quat_tensor(rotvec: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert a 3D rotation vector to a unit quaternion.
    Args:
        rotvec: Tensor of shape (..., 3)
        eps: Small constant to avoid division by zero.
    Returns:
        Tensor of shape (..., 4) representing quaternion [w, x, y, z].
    """
    # Compute angle and axis
    angle = rotvec.norm(dim=-1, keepdim=True)
    axis = torch.where(angle < eps, torch.zeros_like(rotvec), rotvec / angle)
    half_angle = angle * 0.5

    # Quaternion components
    w = torch.cos(half_angle)
    xyz = axis * torch.sin(half_angle)
    return torch.cat([w, xyz], dim=-1)

def quat_to_rotvec_tensor(quat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert a unit quaternion to a 3D rotation vector (axis-angle).
    Args:
        quat: Tensor of shape (..., 4) representing [w, x, y, z]
        eps: Small constant to avoid division by zero.
    Returns:
        Tensor of shape (..., 3) representing axis * angle.
    """
    # Ensure quaternion is normalized
    quat = quat / quat.norm(dim=-1, keepdim=True)
    w = torch.clamp(quat[..., :1], -1.0, 1.0)
    xyz = quat[..., 1:]

    # Compute angle and axis
    angle = 2 * torch.acos(w)
    sin_half = torch.sqrt((1 - w * w).clamp(min=eps))
    axis = xyz / sin_half

    return axis * angle

def euler_xyz_to_quaternion_tensor(euler_angles):
    """
    Converts Euler angles (roll, pitch, yaw) to quaternion [w, x, y, z].
    Input shape: (..., 3)
    Output shape: (..., 4)
    """
    roll, pitch, yaw = torch.unbind(euler_angles, dim=-1)

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack((w, x, y, z), dim=-1)

def quaternion_to_euler_xyz_tensor(quat):
    """
    Converts quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw).
    Input shape: (..., 4)
    Output shape: (..., 3)
    """
    w, x, y, z = torch.unbind(quat, dim=-1)

    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * (torch.pi / 2),
        torch.asin(sinp)
    )

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)

def quaternion_to_euler_xyz_safe_tensor(quat, eps=1e-6):
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

def compute_pose_loss(rotvec_current: torch.Tensor, rotvec_target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the squared rotation loss between current and target poses.
    Loss = || log(q_target^{-1} * q_current) ||^2
    Args:
        rotvec_current: Tensor (..., 3), current pose in rotation-vector form.
        rotvec_target: Tensor (..., 3), target pose in rotation-vector form.
        eps: Small constant to stabilize near-zero rotations.
    Returns:
        Tensor (...), scalar loss per batch element.
    """
    # 1. Convert rotation vectors to quaternions
    q_current = rotvec_to_quat_tensor(rotvec_current, eps)
    q_target = rotvec_to_quat_tensor(rotvec_target, eps)

    # 2. Compute relative quaternion: q_rel = q_target^{-1} * q_current
    q_target_inv = torch.cat([q_target[..., :1], -q_target[..., 1:]], dim=-1)

    q_rel = quat_mul(q_target_inv, q_current)

    # 3. Map relative quaternion to rotation vector (log-map)
    r_rel = quat_to_rotvec_tensor(q_rel, eps)

    # 4. Return squared norm as loss
    return (r_rel**2).sum(dim=-1)

# Example usage:
if __name__ == "__main__":
    # Define current and target rotation-vectors
    rot_current = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    rot_target = torch.tensor([0.2, -0.1, 0.4])

    # Compute loss and backward
    loss = compute_pose_loss(rot_current, rot_target)
    loss.backward()

    print("Loss:", loss.item())
    print("Gradient wrt rot_current:", rot_current.grad)
