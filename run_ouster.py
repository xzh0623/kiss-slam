from pathlib import Path

import numpy as np
from kiss_icp.datasets import dataset_factory
from kiss_slam.pipeline import SlamPipeline
from kiss_slam.tools.visualizer import RegistrationVisualizer
from pyquaternion import Quaternion

CYAN = np.array([0.0, 0.9, 1.0])  # GT trajectory
MAGENTA = np.array([1.0, 0.0, 1.0])  # current GT position
SPHERE_SIZE_GT = 0.5


class GTRegistrationVisualizer(RegistrationVisualizer):
    """KISS-SLAM visualizer that also draws the ground-truth trajectory.

    Colors (existing): yellow = local map, green = odometry, blue = key poses,
    red = loop closures.  Added: cyan line = full GT, magenta ball = GT pose of
    the scan currently being processed.
    """

    def __init__(self, gt_poses, first=0):
        super().__init__()
        self.gt_poses = np.asarray(gt_poses, dtype=np.float64)
        self.frame_idx = first  # index of the scan processed at the next update()

        gt_xyz = self.gt_poses[:, :3, 3]
        self.gt_line = self.o3d.geometry.LineSet()
        self.gt_line.points = self.o3d.utility.Vector3dVector(gt_xyz)
        self.gt_line.lines = self.o3d.utility.Vector2iVector(
            np.stack([np.arange(len(gt_xyz) - 1), np.arange(1, len(gt_xyz))], axis=1)
        )
        self.gt_line.colors = self.o3d.utility.Vector3dVector(
            np.tile(CYAN, (len(gt_xyz) - 1, 1))
        )
        self.vis.add_geometry(self.gt_line, reset_bounding_box=False)

        self.gt_marker = self._add_frame(self.gt_poses[0], SPHERE_SIZE_GT, MAGENTA)
        self._marker_pos = self.gt_poses[0, :3, 3].copy()
        self.vis.add_geometry(self.gt_marker, reset_bounding_box=False)

    def update(self, slam):
        # Must happen before super().update(), which blocks and renders.
        self._update_gt_marker()
        super().update(slam)
        self.frame_idx += 1

    def _update_gt_marker(self):
        i = min(self.frame_idx, len(self.gt_poses) - 1)
        pos = self.gt_poses[i, :3, 3]
        self.gt_marker.translate(pos - self._marker_pos)
        self._marker_pos = pos.copy()
        self.vis.update_geometry(self.gt_marker)


seq = Path('/media/jolin/1440-E804/Ouster Datasets/New College Dataset/2020-ouster-os1-64-realsense/02_long_experiment')
rosbag_dir = seq / 'rosbag'
dataset = dataset_factory(dataloader='rosbag', data_dir=rosbag_dir, topic='/os1_cloud_node/points')

file_path = seq / 'ground_truth' / 'registered_poses.csv'
gt = np.genfromtxt(str(file_path), delimiter=',', dtype=np.float64)

xyz = gt[:, 2:5]
rots = np.array([
    Quaternion(x=x, y=y, z=z, w=w).rotation_matrix
    for x, y, z, w in gt[:, 5:]
], dtype=np.float64)

poses = np.eye(4, dtype=np.float64).reshape(1, 4, 4).repeat(len(gt), axis=0)
poses[:, :3, :3] = rots
poses[:, :3, 3] = xyz

T_CL = np.eye(4, dtype=np.float32)
T_CL[:3, :3] = Quaternion(x=0, y=0, z=0.924, w=0.383).rotation_matrix
T_CL[:3, 3] = np.array([-0.084, -0.025, 0.050], dtype=np.float32)

poses = np.einsum('nij,jk->nik', poses, T_CL)
poses = np.einsum('ij,njk->nik', np.linalg.inv(poses[0]), poses)

dataset.gt_poses = poses
dataset.sequence_id = '02_long_experiment'

# visualize=False so the stock visualizer (and its window) is not created;
# we plug in our GT-aware one instead.
pipeline = SlamPipeline(dataset=dataset, n_scans=-1, visualize=False)
pipeline.visualizer = GTRegistrationVisualizer(poses, first=0)

res = pipeline.run()
print('RESULTS START')
res.print()
print('RESULTS END')


def path_length(poses):
    """Sum of distances between consecutive pose translations, in meters."""
    xyz = np.asarray(poses)[:, :3, 3]
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
 
 
# pipeline.poses holds the final (loop-closure optimized) trajectory of the
# frames that were actually processed; compare against GT over the same frames.
est_poses = pipeline.poses
n = len(est_poses)
est_len = path_length(est_poses)
gt_len = path_length(poses[:n])
 
print('TRAJECTORY LENGTH START')
print(f'Frames processed        : {n}')
print(f'Estimated trajectory    : {est_len:.2f} m')
print(f'Ground-truth trajectory : {gt_len:.2f} m (same {n} frames)')
print(f'Estimated / GT          : {est_len / gt_len * 100:.2f} %')
print('TRAJECTORY LENGTH END')