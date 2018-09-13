import tensorflow as tf
import numpy as np
import os

from src.utils.pose import Pose3D
from src.utils.pose import PoseConfig


class Pose3DInterface:



    def __init__(self, session, protograph):

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

        self.inp = self.graph.get_tensor_by_name("serverInputs/enc_in:0")

        self.out = self.graph.get_tensor_by_name("linear_model/add_1:0")

        self.dropout_keep_prob = self.graph.get_tensor_by_name("dropout_keep_prob:0")





    """
        Extend the pose2D in 3 dimension.

        Args:
            * a list of src.utils.pose.pose2D

        Return:
            * a list of src.utils.pose.pose3D

    """

    def predict(self, many_pose_2d):

        if len(many_pose_2d) == 0:
            return []

        features = []

        for pose in many_pose_2d:
            joints = pose.get_joints()
            size = max(pose.to_bbox().get_width(), pose.to_bbox().get_height())

            feat = ((joints - joints.min(0)) / size).reshape(-1)

            features.append(feat)

        features = np.array(features)



        # infer z dimension
        many_z_axis = self.session.run(self.out, feed_dict={self.inp: features,
                                                            self.dropout_keep_prob: 1.0
                                                            })

        many_pose3d = []

        for poseId in range(len(many_pose_2d)):
            # concat z-axis => <x,y,z>
            pose3d = np.zeros((PoseConfig.get_total_joints(), 3))
            pose3d[:, :2] = features[poseId, :].reshape(-1, 2)
            pose3d[:, 2] = many_z_axis[poseId, :]
            many_pose3d.append(Pose3D(pose3d))

        return many_pose3d




