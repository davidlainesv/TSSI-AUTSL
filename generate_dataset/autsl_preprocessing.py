import tensorflow as tf
import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark

class Preprocessing:
    range_dict = {
        'pose': range(0, 17),
        'face': range(33, 33+468),
        'leftHand': range(33+468, 33+468+21),
        'rightHand': range(33+468+21, 33+468+21+21),
        'root': range(33+468+21+21, 33+468+21+21+1)
    }
    
    def __init__(self, tssi_order):
        joints_idxs = []
        for joint in tssi_order:
            joint_type = joint.split("_")[0]
            if joint_type == "root":
                landmark_id = 0
            else:
                landmark_id = int(joint.split("_")[1])
            idx = self.range_dict[joint_type][landmark_id]
            joints_idxs.append(idx)
        
        self.joints_idxs = joints_idxs
        self.left_wrist_idx = self.range_dict["pose"][PoseLandmark.LEFT_SHOULDER]
        self.right_wrist_idx = self.range_dict["pose"][PoseLandmark.RIGHT_SHOULDER]
        self.root_idx = self.range_dict["root"][0]
        
    def __call__(self, pose):
        # pose = tensor.numpy()
        pose = self.reshape(pose)
        pose = self.fill_z_with_zeros(pose)
        pose = self.add_root(pose)
        pose = self.sort_columns(pose)
        return pose
        
    def reshape(self, pose):
        pose = pose[:, 0, :, :3]
        return pose
    
    def fill_z_with_zeros(self, pose):
        x, y, _ = tf.unstack(pose, axis=-1)
        z = tf.zeros(tf.shape(x))
        return tf.stack([x, y, z], axis=-1)
        
    def add_root(self, pose):
        left = pose[:, self.left_wrist_idx, :]
        right = pose[:, self.right_wrist_idx, :]
        root = (left + right) / 2
        root = root[:, tf.newaxis, :]
        pose = tf.concat([pose, root], axis=1)
        return pose
    
    def sort_columns(self, pose):
        pose = tf.gather(pose, indices=self.joints_idxs, axis=1)
        return pose