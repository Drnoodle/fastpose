import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from mpl_toolkits.mplot3d import axes3d, Axes3D

from src.utils.bbox import BBox
from src.utils.pose import Pose2D, PoseConfig

"""
Draw annotation over an image
"""
class Drawer:

    BONE_COLOR = (0,255,222)
    JOINT_COLOR =(5,5,5)

    PID_FOREGROUND = (0,255,222)
    PID_BACKGROUND = (35, 35, 35)
    PID_LETTERS = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]


    TXT_FOREGROUND = (225,225,225)
    TXT_BACKGROUND = (25,25,25)


    #BONE_COLOR = (110, 249, 227)
    #JOINT_COLOR = (60, 199, 157)

    @staticmethod
    def draw_text(img, position, txt, size=32,color=(0,0,0), fontpath = "ressources/fonts/Open_Sans/OpenSans-Bold.ttf"):

        font = ImageFont.truetype(fontpath, size)
        img_pil = Image.fromarray(img.copy())
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, txt, font=font, fill=(color[0], color[1], color[2],255))
        img = np.array(img_pil)

        return img


    """ Return a new image with the 2D pose depicted for a given src.utils.pose.pose2D"""
    @staticmethod
    def draw_2d_pose(img, pose_2d, thickness=2):

        img = img.copy()

        bones = PoseConfig.BONES

        joints = pose_2d.get_joints()

        joints[:,0] = (joints[:,0] * img.shape[1])
        joints[:,1] = (joints[:,1] * img.shape[0])
        joints = joints.astype(int)

        is_active_mask = pose_2d.get_active_joints()

        for bone_id in range(len(bones)):

            joint_ids = bones[bone_id]

            joint_1 = joints[joint_ids[0]]
            joint_2 = joints[joint_ids[1]]

            if is_active_mask[joint_ids[0]] and is_active_mask[joint_ids[1]]:
                cv2.line(img, tuple(joint_1), tuple(joint_2), Drawer.BONE_COLOR, thickness)


        for i in range(0,joints.shape[0]):
            color = Drawer.BONE_COLOR if i == 0 else Drawer.JOINT_COLOR
            cv2.circle(img, (joints[i,0], joints[i,1]), 3, color, -1)

        # draw the head as a point

        #tmp = pose_2d.get_gravity_center()
        #tmp_ids = [PoseConfig.HEAD, PoseConfig.L_HIP, PoseConfig.R_HIP, PoseConfig.L_SHOULDER, PoseConfig.R_SHOULDER]
        #avg_dist_from_center = np.sqrt(((pose_2d.get_joints()[tmp_ids,:] - tmp)**2).sum(1)).mean()

        #head_size = avg_dist_from_center
        #tmp = pose_2d.get_joints()[PoseConfig.HEAD, :]
        #bbox = BBox(tmp[0]-head_size/4.0, tmp[0]+head_size/4.0,tmp[1]-head_size/2.0, tmp[1]+head_size/2.0)

        #img = Drawer.draw_bbox(img, bbox, thickness=2, color=Drawer.PID_FOREGROUND)



        #cv2.circle(img,(joints[PoseConfig.HEAD, 0], joints[PoseConfig.HEAD, 1]), 6,  Drawer.JOINT_COLOR, -1)


        return img

    """Return a new image with the bbox drawn for a given src.utils.bbox.BBox"""
    @staticmethod
    def draw_bbox(img, bbox, thickness=3, color=(255, 255, 255)):

        img = img.copy()

        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        min_x, max_x = bbox.get_min_x(img), bbox.get_max_x(img)
        min_y, max_y = bbox.get_min_y(img), bbox.get_max_y(img)

        cv2.rectangle(img,(min_x,min_y),(max_x,max_y),color,thickness)

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img




    """ Return a new image with all 2D pose depicted for a given list of annotations"""
    @staticmethod
    def draw_scene(img, poses_2d, person_ids, fps=None, curr_frame=None):

        img = img.copy()

        img = cv2.rectangle(img, (0,0), (img.shape[1], 20), Drawer.TXT_BACKGROUND, cv2.FILLED)

        if not isinstance(fps, type(None)):
            img = Drawer.draw_text(img, (img.shape[1]-178,0), "running at "+str(fps)+" fps",size=13,color=Drawer.TXT_FOREGROUND)


        if not isinstance(curr_frame, type(None)):
            img = Drawer.draw_text(img, (40,0), "frame "+str(int(curr_frame)), color=Drawer.TXT_FOREGROUND, size=13)



        for pid in range(len(poses_2d)):

            # Draw the skeleton
            img = Drawer.draw_2d_pose(img, poses_2d[pid])

            # The person id is written on the gravity center

            tmp = poses_2d[pid].get_gravity_center()
            tmp[0] = (tmp[0]+poses_2d[pid].get_joints()[PoseConfig.HEAD, 0])/2.0
            tmp[1] = (tmp[1]+poses_2d[pid].get_joints()[PoseConfig.HEAD, 1])/2.0

            x, y = int(tmp[0]*img.shape[1]), int(tmp[1]*img.shape[0])

            img = cv2.rectangle(img, (x-13, y-23), (x + 17, y+7), Drawer.PID_FOREGROUND, cv2.FILLED)
            img = cv2.rectangle(img, (x-7, y-17), (x + 23, y+13), Drawer.PID_BACKGROUND, cv2.FILLED)
            img = Drawer.draw_text(img,  (x, y-20), Drawer.PID_LETTERS[person_ids[pid]],size=20, color=Drawer.PID_FOREGROUND, fontpath = "ressources/fonts/Open_Sans/OpenSans-Bold.ttf")

        return img


