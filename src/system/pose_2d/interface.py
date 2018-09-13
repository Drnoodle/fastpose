import time

import tensorflow as tf
import os
import cv2
import numpy as np

from src.utils.body_cover import BodyCover
from src.utils.drawer import Drawer
from src.utils.pose import Pose2D, PoseConfig
import matplotlib.image as mpimg


class Pose2DInterface:



    def __init__(self, session, protograph, post_processing, input_size, subject_padding, input_node_name, output_node_name):

        os.environ['CUDA_VISIBLE_DEVICES'] = ''

        with tf.gfile.GFile(protograph, "rb") as f:
            restored_graph_def = tf.GraphDef()
            restored_graph_def.ParseFromString(f.read())

        tf.import_graph_def(
            restored_graph_def,
            input_map=None,
            return_elements=None,
            name=""
        )

        self.session = session

        self.graph = tf.get_default_graph()

        self.image = self.graph.get_tensor_by_name(input_node_name)

        self.output = self.graph.get_tensor_by_name(output_node_name)

        self.input_size = input_size

        self.post_processing = post_processing

        self.subject_padding = subject_padding

        self.body_cover = BodyCover(self.subject_padding)




    """
    In the case the model only output heatmaps, this postprocessing transform the
    resulting heatmaps in the pose 2D. (defined for the post_processing attribute in the init method)
    """
    @staticmethod
    def standard_heatmap_postprocessing(heatmaps, cropped_image_bbox, input_size):

        aligned_heatmaps = heatmaps.reshape(heatmaps.shape[0] * heatmaps.shape[1], -1)

        max_ids = np.argmax(aligned_heatmaps, axis=0)

        confidences = [aligned_heatmaps[max_ids[i], i] for i in range(len(max_ids))]

        xPos = np.remainder(max_ids, heatmaps.shape[1]).reshape((-1, 1))
        yPos = np.divide(max_ids, heatmaps.shape[1]).astype(np.uint8).reshape((-1, 1))

        res = np.hstack([xPos, yPos]).astype(np.float32)
        res[:, 0] /= heatmaps.shape[1]
        res[:, 1] /= heatmaps.shape[0]

        newPose = Pose2D(res).to_absolute_coordinate_from(cropped_image_bbox).clamp(0.0, 1.0)

        return newPose, confidences

    """
    In the case the model output heatmaps+offset vectors, this postprocessing transform the
    resulting output in the pose 2D. (defined for the post_processing attribute in the init method)
    """
    @staticmethod
    def our_approach_postprocessing(network_out, subject_bbox, input_size):


        total_joints = PoseConfig.get_total_joints()

        heatmap = network_out[:, :, :total_joints]
        xOff = network_out[:, :, total_joints:(total_joints * 2)]
        yOff = network_out[:, :, (total_joints * 2):]


        confidences = []
        joints = np.zeros((total_joints, 2)) - 1

        for jointId in range(total_joints):

            inlined_pix = heatmap[:, :, jointId].reshape(-1)
            pixId = np.argmax(inlined_pix)

            confidence = inlined_pix[pixId]

            # if max confidence below 0.1 => inactive joint
            if inlined_pix[pixId] < 0.01:
                confidences.append(confidence)
                continue

            outX = pixId % heatmap.shape[1]
            outY = pixId // heatmap.shape[1]

            x = outX / heatmap.shape[1] * input_size + xOff[outY, outX, jointId]
            y = outY / heatmap.shape[0] * input_size + yOff[outY, outX, jointId]

            x = x / input_size
            y = y / input_size

            joints[jointId, 0] = x
            joints[jointId, 1] = y
            confidences.append(confidence)


        return Pose2D(joints).to_absolute_coordinate_from(subject_bbox), confidences



    i = 0

    """
        Pose 2D inference

        Args:
            * img : the image to annotate
            * subject_bboxes : the bbox without padding in %
            * prev_poses : use by the body cover to hide stranger people when people are standing side by side

        Return:
            * a list of src.utils.pose.pose2D and a list of confidences (per person, per joint)

    """
    def predict(self, img, subject_bboxes, prev_poses=[]):

        Pose2DInterface.i += 1

        if len(subject_bboxes) == 0:
            return [], []


        cropped_images = []


        # filter bbox having no size, insuring the cropped image is not empty

        filtered_bbox,filtered_poses = [], []

        for subject_id in range(len(subject_bboxes)):

            subject_bbox = subject_bboxes[subject_id]
            subject_bbox_padded = subject_bbox.to_squared(img, self.subject_padding)

            width = int(subject_bbox_padded.get_width() * img.shape[1])
            height = int(subject_bbox_padded.get_height() * img.shape[0])

            if width > 0 and height > 0:
                filtered_bbox.append(subject_bboxes[subject_id])
                if subject_id < len(prev_poses):
                    filtered_poses.append(prev_poses[subject_id])

        subject_bboxes, prev_poses = filtered_bbox, filtered_poses



        # crop images and hide stranger bodies

        for subject_id in range(len(subject_bboxes)):


            subject_bbox = subject_bboxes[subject_id]

            subject_bbox_padded = subject_bbox.to_squared(img, self.subject_padding)


            ada_bboxes, adaPoses, subject_id_to_keep = [], [], subject_id


            for i in range(len(subject_bboxes)):

                curr_bbox = subject_bboxes[i]
                curr_bbox = curr_bbox.intersect(subject_bbox_padded)

                if curr_bbox is None: #intersection is empty
                    if i < subject_id:
                        subject_id_to_keep -= 1
                    continue

                curr_bbox = curr_bbox.translate(-subject_bbox_padded.get_min_x(), -subject_bbox_padded.get_min_y())
                curr_bbox = curr_bbox.scale(1.0 / subject_bbox_padded.get_width(), 1.0 / subject_bbox_padded.get_height())

                ada_bboxes.append(curr_bbox)

                if i < len(prev_poses) and prev_poses[i] is not None:
                    adaPoses.append(prev_poses[i].to_relative_coordinate_into(subject_bbox_padded))
                else:
                    adaPoses.append(None)


            cropped_img = subject_bbox_padded.crop(img)

            cropped_img = self.body_cover.hide_strangers(cropped_img, ada_bboxes, subject_id_to_keep, adaPoses)

            cropped_img = cv2.resize(cropped_img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)

            cropped_img = cropped_img.astype(np.float32) / (255.0 / 2.0) - 1.0

            cropped_images.append(cropped_img)


        # infer the cropped images

        out = np.zeros((0, PoseConfig.get_total_joints() * 3))

        if len(cropped_images) > 0:
            out = self.session.run(self.output, feed_dict={self.image: cropped_images})


        # decode outputs

        poses_2d, confidences = [], []

        for subject_id in range(out.shape[0]):

            # 1.- recover the pose inside the cropped image from the confidence heatmaps
            curr_heatmaps = out[subject_id, :, :, :]
            cropped_image_bbox = subject_bboxes[subject_id].to_squared(img, self.subject_padding)

            curr_pose_2d, curr_confidences = self.post_processing(curr_heatmaps, cropped_image_bbox, self.input_size)

            poses_2d.append(curr_pose_2d)
            confidences.append(curr_confidences)



        return poses_2d, confidences

