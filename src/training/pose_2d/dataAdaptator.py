import cv2
import numpy as np
import random

class DataAdaptator:



    def __init__(self, coco, img_id, input_size, padding, jitter, mask=None, body_cover=None,
                 data_augment=None):

        self.coco, self.img_id = coco, img_id

        self.padding, self.jitter = padding, jitter

        self.input_size = input_size

        if not isinstance(mask, type(None)):
            self.active_ids = [i for i in range(self.coco.get_total_person_on(self.img_id)) if mask[i]]
        else:
            self.active_ids = [i for i in range(self.coco.get_total_person_on(self.img_id))]

        self.body_cover = body_cover
        self.data_augment = data_augment



    def size(self):
        return len(self.active_ids)


    def get_image(self):

        img = self.coco.get_image(self.img_id)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img

    def get_pose(self, entryId):
        return self.coco.get_poses(self.img_id)[self.active_ids[entryId]]



    def data(self, entryId):

        pose_id = self.active_ids[entryId]

        poses = self.coco.get_poses(self.img_id)
        image = self.coco.get_image(self.img_id)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if not isinstance(self.data_augment, type(None)):
            image, poses = self.data_augment.apply(image, poses)

        pose = poses[pose_id]

        # build the bounding box with 30% padding around the subject keypoints
        # adding some noise (jitter value) in the scale/position

        required_padding = self.padding + random.random() * self.jitter - self.jitter / 2.0

        bbox = pose.to_bbox().to_squared(image, required_padding)

        rnd_trans_x = (random.random() * self.jitter - self.jitter / 2.0) * (bbox.get_max_x() - bbox.get_min_x())
        rnd_trans_y = (random.random() * self.jitter - self.jitter / 2.0) * (bbox.get_max_y() - bbox.get_min_y())

        bbox = bbox.translate(rnd_trans_x, rnd_trans_y)

        # crop image and put the pose in relative coordinates

        cropped_image = bbox.crop(image)
        cropped_image = cv2.resize(cropped_image, self.input_size)

        if not isinstance(self.data_augment, type(None)):
            cropped_image = self.data_augment.random_subsample(cropped_image)


        pose = pose.to_relative_coordinate_into(bbox)

        # no need to hide stranger part => done
        if not isinstance(self.body_cover, type(None)):

            allPoses = [p.to_relative_coordinate_into(bbox) for p in poses]
            allBBoxes = [p.to_bbox() for p in allPoses]

            cropped_image = self.body_cover.hide_strangers(cropped_image, allBBoxes, pose_id, allPoses)


        cropped_image = cropped_image.astype(np.float32) / (255.0 / 2.0) - 1.0


        return cropped_image, pose




    def drawn(self):
        return self.data(int(random.random() * self.size()))