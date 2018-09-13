from math import sqrt

from src.utils.pose import PoseConfig

"""
Hide stranger people present in the hero's subject bbox
"""
class BodyCover:


    def __init__(self, padding):
        self.padding = padding


    def _clamp(self, v, min_v, max_v):
        return min(max(v, min_v), max_v)


    def _get_in_image_pos(self, bbox, img):
        minX = int(self._clamp(bbox.get_min_x() * img.shape[1], 0, img.shape[1] - 1))
        maxX = int(self._clamp(bbox.get_max_x() * img.shape[1], 0, img.shape[1] - 1))
        minY = int(self._clamp(bbox.get_min_y() * img.shape[0], 0, img.shape[0] - 1))
        maxY = int(self._clamp(bbox.get_max_y() * img.shape[0], 0, img.shape[0] - 1))
        return minX, maxX, minY, maxY


    def distance(self, v1, v2):
        return sqrt(((v1-v2)**2).sum())


    """
    Args :
      * image : the cropped image used for the inference with the subject to annotate
      * all bbox : all the bboxes in the cropped image coordinate (without padding)
      * id_to_preserve : the id of the hero's bbox to preserve
      * all poses : extra information that can be used to further hide foreigners

    """
    def hide_strangers(self, image, all_bboxes, id_to_preserve, all_poses=[]):


        covered_image = image.copy()

        # all persons bbox are zero valued

        for to_remove_bbox in all_bboxes:

            curr_bbox = to_remove_bbox.get_with_padding(self.padding)
            min_x, max_x, min_y, max_y = self._get_in_image_pos(curr_bbox, image)
            covered_image[min_y:max_y, min_x:max_x, :] = 0


        # hero is restored

        to_keep_bbox = all_bboxes[id_to_preserve].get_with_padding(self.padding)
        min_x, max_x, min_y, max_y = self._get_in_image_pos(to_keep_bbox, image)
        covered_image[min_y:max_y, min_x:max_x, :] = image[min_y:max_y, min_x:max_x, :]



        # further hide joints when people are too overlapped to hide the all bbox

        poses_to_remove = [all_poses[i] for i in range(len(all_poses)) if i != id_to_preserve and not isinstance(all_poses[i], type(None))]

        for to_hide_pose in poses_to_remove:

            if sum(to_hide_pose.get_active_joints()) > 3:
                curr_bbox = to_hide_pose.to_bbox()

                min_x, max_x, min_y, max_y = self._get_in_image_pos(curr_bbox, image)

                covered_image[min_y:max_y, min_x:max_x, :] = 0


        curr_bbox = all_bboxes[id_to_preserve].get_with_padding(self.padding)
        min_x, max_x, min_y, max_y = self._get_in_image_pos(curr_bbox, image)
        covered_image[min_y:max_y, min_x:max_x, :] = image[min_y:max_y, min_x:max_x, :]

        for pose in poses_to_remove:

            if sum(pose.get_active_joints()) <= 3:
                continue

            is_req_hide = pose.is_active_joint(PoseConfig.L_SHOULDER)
            is_req_hide = is_req_hide and pose.is_active_joint(PoseConfig.R_SHOULDER)
            is_req_hide = is_req_hide and pose.is_active_joint(PoseConfig.L_HIP)
            is_req_hide = is_req_hide and pose.is_active_joint(PoseConfig.R_HIP)

            if not is_req_hide:
                continue

            joints = pose.get_joints()

            joints[:, 0] *= image.shape[1]
            joints[:, 1] *= image.shape[0]
            joints = joints.astype(int)
            lshoulder, rshoulder = joints[PoseConfig.L_SHOULDER, :], joints[PoseConfig.R_SHOULDER, :]
            lhip, rhip = joints[PoseConfig.L_HIP, :], joints[PoseConfig.R_HIP, :]

            size = max(self.distance(lhip, rshoulder), self.distance(lshoulder, rhip))
            size = int(0.25 * size)

            for jointId in range(joints.shape[0]):

                center_x, center_y = joints[jointId, 0], joints[jointId, 1]

                is_stranger_joint_over_subject = all_bboxes[id_to_preserve].is_inside(
                    float(center_x) / image.shape[1],
                    float(center_y) / image.shape[0])

                has_to_be_hide = (center_x >= 0 and center_y >= 0)
                has_to_be_hide = has_to_be_hide and ((not is_stranger_joint_over_subject) or jointId == PoseConfig.HEAD)

                if has_to_be_hide:
                    min_x, max_x = center_x - size // 2, center_x + size // 2
                    min_y, max_y = center_y - size // 2, center_y + size // 2
                    covered_image[min_y:max_y, min_x:max_x, :] = 0


        return covered_image

