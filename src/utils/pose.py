from src.utils.bbox import BBox
import numpy as np


"""
Pose configuration
"""
class PoseConfig():

    # The joint order defined by the system

    NAMES = ["head", "leftShoulder", "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist", "leftHip",
             "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"]

    HEAD, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 0, 1, 2, 3, 4, 5, 6
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 7, 8, 9, 10, 11, 12


    # The available bones

    BONES = [(1, 3), (3, 5), (2, 4), (4, 6), (7, 9), (9, 11), (8, 10), (10, 12), (7,8), (1,2), (1,7), (2,8)]

    """Return the total number of joints """
    @staticmethod
    def get_total_joints():
        return len(PoseConfig.NAMES)

    """Return the total number of bones """
    @staticmethod
    def get_total_bones():
        return len(PoseConfig.BONES)



"""
Wrap a 3D pose (numpy array of size <PoseConfig.get_total_joints(),3> )
"""
class Pose3D():


    FROM_HUMAN_36_PERMUTATION = [6, 7, 10, 8, 11, 9, 12, 3, 0, 4, 1, 5, 2]

    def __init__(self, npArray):

        if len(npArray.shape) != 2 or npArray.shape[0] != PoseConfig.get_total_joints() or npArray.shape[1] != 3:
            raise Exception("Pose 3D only accepts numpy array with shape : <total joints, 3 DIM>")

        self.joints = npArray

    """Build a 3D pose from a numpy human36M ordered content"""
    @staticmethod
    def build_from_human36(npArray):
        return Pose3D(npArray[Pose3D.FROM_HUMAN_36_PERMUTATION, :])

    """Return the 3D joints as numpy array"""
    def get_joints(self):
        return self.joints.copy()


    def __str__(self):
        return self.joints.__str__()



"""
Wrap a 2D pose (numpy array of size <PoseConfig.get_total_joints(),2> )
"""
class Pose2D:


    # The joints isn't in the same order in the differents datasets

    FROM_MPII_PERMUTATION = [9, 13, 12, 14, 11, 15, 10, 3, 2, 4, 1, 5, 0]
    FROM_COCO_PERMUTATION = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    FROM_COCO2_PERMUTATION = [0, 4, 1, 5, 2, 6, 3, 10, 7, 11, 8, 12, 9]
    TO_HUMAN_36_PERMUTATION = [8, 10, 12, 7, 9, 11, 0, 1, 3, 5, 2, 4, 6]
    #FROM_POSE2D_PERMUTATION = [0, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]


    def __init__(self, npArray):

        if len(npArray.shape) != 2 or npArray.shape[0] != PoseConfig.get_total_joints() or npArray.shape[1] != 2:
            raise Exception("Pose 2D only accepts numpy array with shape : <total joints, 2 DIM>")

        self.joints = npArray

        self.is_active_mask = []

        for joint_id in range(PoseConfig.get_total_joints()):
            self.is_active_mask.append(not np.array_equal(self.joints[joint_id, [0, 1]], [-1, -1]))

        self.is_active_mask = np.array(self.is_active_mask)


    """Build a 2D pose from a numpy mpii ordered content"""
    @staticmethod
    def build_from_mpii(npArray):
        return Pose2D(npArray[Pose2D.FROM_MPII_PERMUTATION, :])

    """Build a 2D pose from a numpy coco ordered content"""
    @staticmethod
    def build_from_coco(npArray):

        joints = npArray[Pose2D.FROM_COCO_PERMUTATION, :]

        return Pose2D(joints)

    """Build a 2D pose from a json with the format : {jointName : {'x','y'}, ...}
       with jointName in PoseConfig.NAMES"""
    @staticmethod
    def build_from_JSON(json):

        joints = np.zeros([PoseConfig.get_total_joints(), 2]) - 1.0

        for jointId, name in enumerate(PoseConfig.NAMES):
            joints[jointId, 0] = json[name]['x']
            joints[jointId, 1] = json[name]['y']

        return Pose2D(joints)

    """Return the 2D joints as numpy array"""
    def get_joints(self):
        return self.joints.copy()


    """Scale the x,y position by xScaler, yScaler"""
    def scale(self, xScaler=1.0, yScaler=1.0):
        joints = self.joints.copy()
        joints[self.is_active_mask, 0] = joints[self.is_active_mask, 0] * xScaler
        joints[self.is_active_mask, 1] = joints[self.is_active_mask, 1] * yScaler
        return Pose2D(joints)



    def to_pose_3d_features2(self):

        joints = self.joints.copy()

        # TODO : 2 next lines useless
        # normalize features
        center = self.get_gravity_center()

        joints[self.is_active_mask,:] = joints[self.is_active_mask,:] - center

        joints[self.is_active_mask, :] = joints[self.is_active_mask, :] - joints[self.is_active_mask, :].min(0)

        joints[self.is_active_mask, :] = joints[self.is_active_mask, :]/joints[self.is_active_mask, :].max(0)

        return joints.reshape(-1)



    """Convert the 2D pose to the numpy features required by the 2D=>3D model"""
    def to_pose_3d_features(self):

        joints = self.joints.copy()

        # normalize features
        center_hip = (joints[7, :] + joints[8, :]) / 2.0
        joints[:, 0] = joints[:, 0] - center_hip[0]
        joints[:, 1] = joints[:, 1] - center_hip[1]
        joints = joints / (np.absolute(joints).max() + 0.0000000000001)
        joints[:, 1] = joints[:, 1]

        # convert to human36 joints order
        joints = joints[Pose2D.TO_HUMAN_36_PERMUTATION, :]

        # build a batch of 1 record
        features = np.concatenate([joints[:, 0], joints[:, 1]])
        features = np.expand_dims(features, axis=0)

        return features



    """Return the total number of labeled joints (x and y position are != -1)"""
    def total_labeled_joints(self):
        return self.is_active_mask.sum()

    """Return the mask of labeled joints (x and y position are != -1)"""
    def get_active_joints(self):
        return self.is_active_mask.copy()

    """Return true if the given joint_id is labeled"""
    def is_active_joint(self, joint_id):
        return self.is_active_mask[joint_id]


    def distance_to(self, that):

        mask_1 = that.get_active_joints()
        mask_2 = self.get_active_joints()
        mask = mask_1 & mask_2

        j1 = self.get_joints()[mask,:]
        j2 = that.get_joints()[mask, :]

        return np.sqrt(((j1 -j2)**2).sum(1)).mean()



    def get_gravity_center(self):
        return self.joints[self.is_active_mask, :].mean(0)


    """Transform the pose in a bounding box or return the 100%, 100% box if impossible"""
    def to_bbox(self):

        if self.is_active_mask.sum() < 3:
            return BBox(0, 1, 0, 1)

        min_x, max_x = self.joints[self.is_active_mask, 0].min(), self.joints[self.is_active_mask, 0].max()
        min_y, max_y = self.joints[self.is_active_mask, 1].min(), self.joints[self.is_active_mask, 1].max()

        return BBox(min_x, max_x, min_y, max_y)


    """Return the pose in absolute coordinate if recorded from the given bbox"""
    def to_absolute_coordinate_from(self, bbox):

        joints = self.joints.copy()

        joints[self.is_active_mask, 0] = joints[self.is_active_mask, 0] * (bbox.get_max_x() - bbox.get_min_x()) + bbox.get_min_x()
        joints[self.is_active_mask, 1] = joints[self.is_active_mask, 1] * (bbox.get_max_y() - bbox.get_min_y()) + bbox.get_min_y()

        return Pose2D(joints)


    """Return the pose in the coordinate of the given bbox"""
    def to_relative_coordinate_into(self, bbox):

        joints = self.joints.copy()

        scale_x = bbox.get_max_x() - bbox.get_min_x()
        scale_y = bbox.get_max_y() - bbox.get_min_y()

        joints[self.is_active_mask, 0] = (joints[self.is_active_mask, 0] - bbox.get_min_x()) / scale_x
        joints[self.is_active_mask, 1] = (joints[self.is_active_mask, 1] - bbox.get_min_y()) / scale_y

        return Pose2D(joints)

    """Clamp the results in the selected range :min_value, max_value"""
    def clamp(self, min_value, max_value):

        new_joints = self.joints.copy()

        new_joints[self.is_active_mask, :] = np.clip(new_joints[self.is_active_mask, :], min_value, max_value)

        return Pose2D(new_joints)


    def __str__(self):
        return self.joints.__str__()