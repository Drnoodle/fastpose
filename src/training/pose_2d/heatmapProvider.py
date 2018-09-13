import numpy as np

class HeatmapProvider:
    def __init__(self, width, height, sigma):
        self.width, self.height, self.sigma = width, height, sigma

    def build_heatmap(self, pose):

        joints = pose.get_joints()
        hm = np.zeros((self.height, self.width, joints.shape[0]), dtype=np.float32)

        for joint_id in range(joints.shape[0]):
            if pose.is_active_joint(joint_id):
                centerPos = (joints[joint_id, 0] * self.width, joints[joint_id, 1] * self.height)
                hm[:, :, joint_id] = self._make_gaussian(self.height, self.width, self.sigma, centerPos)

        return hm

    def _make_gaussian(self, height, width, sigma, center):
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        x0, y0 = center[0], center[1]
        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)
