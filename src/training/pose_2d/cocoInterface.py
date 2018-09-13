import json
from src.utils.pose import Pose2D
import numpy as np
import os
import matplotlib.image as mpimg

class CocoInterface:


    def __init__(self, image_dir, images, annotations):
        self.images = images
        self.annotations = annotations
        self.img_dir = image_dir
        self.annotations_keys = [imgId for imgId in self.annotations.keys()]

    @staticmethod
    def build(annot_file, image_dir):

        # load annotations
        annot = json.load(open(annot_file, 'r'))

        # build imgId => image metadata
        images_res = {}
        for i in range(len(annot['images'])):
            images_res[annot['images'][i]['id']] = {
                'fileName': annot['images'][i]['file_name'],
                'width': annot['images'][i]['width'],
                'height': annot['images'][i]['height']
            }

        # build imgId => annotations
        annotations_res = {}

        for i in range(len(annot['annotations'])):

            entry = annot['annotations'][i]
            img_id = entry['image_id']

            kp = entry['keypoints']
            img_width, img_height = float(images_res[img_id]['width']), float(images_res[img_id]['height'])
            kp = [(x / img_width, y / img_height) if v >= 1 else (-1, -1) for x, y, v in
                  zip(kp[0::3], kp[1::3], kp[2::3])]
            kp = np.array(kp).astype(np.float32)
            pose = Pose2D.build_from_coco(kp)

            is_crowd = (entry['iscrowd'] == 1)

            if not img_id in annotations_res:
                annotations_res[img_id] = []

            annotations_res[img_id].append({
                'pose': pose,
                'isCrowd': is_crowd
            })

        return CocoInterface(image_dir, images_res, annotations_res)

    def size(self):
        return len(self.annotations_keys)

    def get_image(self, entry_id):
        img_path = os.path.join(self.img_dir, self.images[self.annotations_keys[entry_id]]["fileName"])
        return mpimg.imread(img_path)

    def get_image_shape(self, entry_id):
        width = self.images[self.annotations_keys[entry_id]]['width']
        height = self.images[self.annotations_keys[entry_id]]['height']
        return [height, width, 3]

    def get_total_person_on(self, entry_id):
        return len(self.annotations[self.annotations_keys[entry_id]])

    def get_pose(self, entry_id, person_id):
        return self.annotations[self.annotations_keys[entry_id]][person_id]['pose']

    def get_poses(self, entry_id):
        return [self.get_pose(entry_id, person_id) for person_id in range(self.get_total_person_on(entry_id))]
