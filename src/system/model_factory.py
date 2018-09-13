import tensorflow as tf
from src.system.pose_2d.interface import Pose2DInterface
from src.system.pose_3d.interface import Pose3DInterface
from src.system.object_detection.interface import YoloInterface

"""
Provide an instantiated model interface containing a method "predict" used for the inference.
"""

class ModelFactory:


    """
    Build the human detector model
    """
    @staticmethod
    def build_object_detection_interface():

        conf_thresh, nms_thresh = 0.25, 0.1

	    # default : yolo_tiny_single_class | also platinium-tiny (30% padding)
        config_file = "parameters/object_detection/tiny/yolo-voc.cfg"
        model_parameters = "parameters/object_detection/tiny/final.weights"

        return YoloInterface(config_file, model_parameters, conf_thresh, nms_thresh)


    """
    Build the 2 dimensional pose model
    """
    @staticmethod
    def build_pose_2d_interface():

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        graph_file = "parameters/pose_2d/tiny/pose2d.pb"

        input_node_name, output_node_name = "Image:0", "Output:0"

        input_size = 144

        #subject_padding = 0.3
        subject_padding = 0.4

        post_processing = Pose2DInterface.our_approach_postprocessing


        return Pose2DInterface(session, graph_file, post_processing, input_size, subject_padding, input_node_name, output_node_name)


    """
    Build the 3 dimensional pose model
    """
    @staticmethod
    def build_pose_3d_interface():

        #tf.reset_default_graph()
        #useGpu = True
        #device_count = {"GPU": 1} if useGpu else {"GPU": 0}
        #session = tf.Session(config=tf.ConfigProto(device_count=device_count))

        #savePath = "parameters/pose_3d/"

        #return Pose3DInterface(session, savePath)

        tf.reset_default_graph()
        use_gpu = True
        device_count = {"GPU": 1} if use_gpu else {"GPU": 0}
        session = tf.Session(config=tf.ConfigProto(device_count=device_count))

        protobuf = "parameters/pose_3d/pose3d.pb"

        return Pose3DInterface(session, protobuf)

