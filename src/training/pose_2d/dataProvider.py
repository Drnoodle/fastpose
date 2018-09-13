from src.training.pose_2d.dataAdaptator import DataAdaptator
from src.utils.body_cover import BodyCover
from src.training.pose_2d.dataAugmentation import DataAugmentation
import random
import numpy as np

from src.training.pose_2d.cocoInterface import CocoInterface


class DataProvider:


    def __init__(self, coco, input_size, batch_size, padding, jitter, mask=None, body_cover=None, data_augment=None):

        self.adapted_datas = []

        self.batch_size = batch_size

        for img_id in range(coco.size()):
            curr_mask = mask[img_id] if not isinstance(mask, type(None)) else None
            ada_data = DataAdaptator(coco, img_id, input_size, padding, jitter, curr_mask, body_cover, data_augment)
            self.adapted_datas.append(ada_data)

        self.adapted_datas = [d for d in self.adapted_datas if d.size() > 0]

        total_examples = 0
        for i in range(len(self.adapted_datas)):
            total_examples += self.adapted_datas[i].size()
        print("TOTAL ANNOTATED EXAMPLES : " + str(total_examples))


    def size(self):
        return len(self.adapted_datas)

    def get_image(self, img_id):
        return self.adapted_datas[img_id].get_image()

    def total_poses_on(self, img_id):
        return self.adapted_datas[img_id].size()

    def get_pose(self, img_id, entry_id):
        return self.adapted_datas[img_id].get_pose(entry_id)


    def drawn(self):

        inputs, outputs = [], []

        for i in range(self.batch_size):
            rnd_img_id = int(random.random() * len(self.adapted_datas))
            curr_in, curr_out = self.adapted_datas[rnd_img_id].drawn()
            inputs.append(np.expand_dims(curr_in, 0))
            outputs.append(curr_out)

        return np.concatenate(inputs, 0), outputs


    @staticmethod
    def build(cocoAnnotFile, cocoImgDir, inputSize, batchSize):

        coco = CocoInterface.build(cocoAnnotFile, cocoImgDir)

        mask = DataProvider.active_pose_mask(coco)

        body_cover = BodyCover(0.3)

        padding, jitter = 0.4, 0.3

        data_augment = DataAugmentation()

        return DataProvider(coco, inputSize, batchSize, padding, jitter, mask=mask, body_cover=body_cover, data_augment=data_augment)


    @staticmethod
    def active_pose_mask(coco):

        # at least k labeled joints
        min_annots_per_person = 10
        # at least subject bbox width or height > n_pix
        min_subject_size = 130
        # max overlapped IOU person :
        max_overlaps = 0.1

        mask = []
        for imgId in range(coco.size()):
            curr_mask = []
            for personId in range(coco.get_total_person_on(imgId)):
                is_annotation_selected = True

                # at least k labeled joints
                is_annotation_selected &= coco.get_pose(imgId, personId).total_labeled_joints() >= min_annots_per_person

                # at least subject bbox width or height > n_pix
                subject_bbox = coco.get_pose(imgId, personId).to_bbox()
                img_shape = coco.get_image_shape(imgId)
                subject_width = (subject_bbox.get_max_x() - subject_bbox.get_min_x()) * img_shape[1]
                subject_height = (subject_bbox.get_max_y() - subject_bbox.get_min_y()) * img_shape[0]
                max_size = max(subject_width, subject_height)

                is_annotation_selected &= max_size > min_subject_size

                # max overlapped IOU person :

                is_overlapped_ok = True

                for another_person_id in range(coco.get_total_person_on(imgId)):

                    if another_person_id == personId:
                        continue

                    bbox2 = coco.get_pose(imgId, another_person_id).to_bbox()
                    bbox1 = coco.get_pose(imgId, personId).to_bbox()
                    interbox = bbox1.intersect(bbox2)

                    if isinstance(interbox, type(None)):
                        continue

                    if (bbox1.get_width()*bbox1.get_height()) <= 0:
                        continue

                    overlaps = (interbox.get_width()*interbox.get_height())/(bbox1.get_width()*bbox1.get_height())

                    if overlaps >= max_overlaps:
                        is_overlapped_ok = False
                        break

                is_annotation_selected &= is_overlapped_ok

                curr_mask.append(is_annotation_selected)

            mask.append(curr_mask)

        return mask