import mujoco
import torch
import torch.nn as nn
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

def random_quaternion(device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Generate a random unit quaternion uniformly distributed on SO(3).

    Returns:
    - q: Tensor of shape (4,), in [w, x, y, z] format
    """
    # Sample four independent normal variates
    q = torch.randn(4, device=device)
    # Normalize to unit length
    q = q / q.norm()
    # Reorder to [w, x, y, z]
    return torch.stack([q[3], q[0], q[1], q[2]], dim=0)

def random_rotvec(angle_range=(0, math.pi), device='cpu') -> torch.Tensor:
    """
    Randomly generate a rotation vector (axis-angle representation).
    
    Args:
        angle_range (tuple): Min and max rotation angle (in radians).
        device (str): 'cpu' or 'cuda'.

    Returns:
        torch.Tensor: A 3D rotation vector (shape: [3])
    """
    # Random axis: uniform on the unit sphere
    axis = torch.randn(3, device=device)
    axis = axis / axis.norm()

    # Random angle in specified range
    angle = torch.empty(1, device=device).uniform_(*angle_range)

    # Rotation vector = angle * axis
    rotvec = angle * axis
    return rotvec

def random_initial_pose_quat(
    pos_range: torch.Tensor,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Generate a random initial pose within the given position range and random orientation.

    Parameters:
    - pos_range: Tensor of shape (2, 3), where pos_range[0] is min [x, y, z] and pos_range[1] is max [x, y, z]
    - device: Torch device for the returned tensor

    Returns:
    - pose: Tensor of shape (7,), concatenation of [x, y, z, q_w, q_x, q_y, q_z]
    """
    # Sample position uniformly in range
    pos = torch.empty(3, device=device).uniform_(pos_range[0], pos_range[1])
    # Sample random orientation
    quat = random_quaternion(device=device)
    # Concatenate position and quaternion
    return torch.cat([pos, quat], dim=0)

def random_initial_pose_rotvec(
    pos_range: torch.Tensor,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    # Sample position uniformly in range
    min_pos = pos_range[0]
    max_pos = pos_range[1]
    pos = min_pos + (max_pos - min_pos) * torch.rand(3, device=device)
    # Sample random orientation
    rotvec = random_rotvec(device=device)
    # Concatenate position and quaternion
    return torch.cat([pos, rotvec], dim=0)


def generate_trajectory(
    start_pose: torch.Tensor,
    num_points: int = 100,
    step: float = 0.05,
    direction_smoothness: float = 0.1,
    pos_range: torch.Tensor = None,
    rotvec_range: torch.Tensor = None
) -> torch.Tensor:
    """
    生成一个平滑的轨迹，位置和旋转向量在指定范围内。
    Generate a smooth trajectory with positions and rotvecs in specified ranges.

    Parameters:
    - start_pose: Initial pose, tensor of shape (6,) [x, y, z, rx, ry, rz]
    - num_points: Number of points in the trajectory (default: 100)
    - step: Distance between consecutive points (default: 0.05)
    - direction_smoothness: Controls smoothness of direction changes (default: 0.1)
    - pos_range: Tensor (2, 3), min/max for xyz. If None, no clamping.
    - rotvec_range: Tensor (2, 3), min/max for rotvec. If None, no clamping.

    Returns:
    - traj: Tensor of shape (num_points, 6), each row is [x, y, z, rx, ry, rz]
    """
    device = start_pose.device
    dtype = start_pose.dtype

    positions = torch.zeros((num_points, 3), device=device, dtype=dtype)
    rotvecs = torch.zeros((num_points, 3), device=device, dtype=dtype)
    positions[0] = start_pose[:3]
    rotvecs[0] = start_pose[3:]

    direction = torch.randn(3, device=device, dtype=dtype)
    direction = direction / (direction.norm() + 1e-8)

    for i in range(1, num_points):
        noise = torch.randn(3, device=device, dtype=dtype) * direction_smoothness
        direction = direction + noise
        direction = direction / (direction.norm() + 1e-8)

        new_pos = positions[i - 1] + step * direction
        if pos_range is not None:
            min_pos, max_pos = pos_range.to(device=device, dtype=dtype)
            new_pos = torch.max(torch.min(new_pos, max_pos), min_pos)
        positions[i] = new_pos

        rot_noise = torch.randn(3, device=device, dtype=dtype) * direction_smoothness
        new_rot = rotvecs[i - 1] + rot_noise
        if rotvec_range is not None:
            min_rot, max_rot = rotvec_range.to(device=device, dtype=dtype)
            new_rot = torch.max(torch.min(new_rot, max_rot), min_rot)
        rotvecs[i] = new_rot

    traj = torch.cat([positions, rotvecs], dim=1)
    return traj

# example
if __name__ == "__main__":
    
    pos_rng = torch.tensor([[-1., -1., -1.], [1., 1., 1.]])
    rot_rng = torch.tensor([[-3.14, -3.14, -3.14], [3.14, 3.14, 3.14]])
    start = random_initial_pose_rotvec(pos_range=pos_rng)
    traj = generate_trajectory(start, num_points=10, pos_range=pos_rng, rotvec_range=rot_rng)
    print(traj)