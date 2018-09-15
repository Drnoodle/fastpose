import copy
import json
import threading
import time
import numpy as np

from src.system.identity_tracker import IdentityTracker
from src.system.model_factory import ModelFactory
from src.utils.pose import PoseConfig, Pose2D, Pose3D


class AnnotatorInterface:


    EXTRA_OBJECT_DETECTION_DELAY = 0.50



    def __init__(self, bbox_model, pose_2d_model, pose_3d_model, max_persons):

        self.bbox_model = bbox_model
        self.pose_2d_model = pose_2d_model
        self.pose_3d_model = pose_3d_model


        self.persons = {}

        self.last_image = None

        self.persons_lock = threading.Lock()

        self.available_person_id = set(range(max_persons))

        self.last_object_detector_timestamp = 0

        # kill threaded object detector routine
        self.object_detector_kill_trigger = False

        self.person_hash_provider = 0

        threading.Thread(target=self.person_identification_routine).start()






    """
    Build the annotator interface using the model defined in the model factory
    """
    @staticmethod
    def build(max_persons=1):

        bbox_model = ModelFactory.build_object_detection_interface()
        pose_2d_model = ModelFactory.build_pose_2d_interface()
        pose_3d_model = ModelFactory.build_pose_3d_interface()

        return AnnotatorInterface(bbox_model, pose_2d_model, pose_3d_model, max_persons)



    """
    Create a new person annotation
    """
    def _new_person(self, person_id):

        self.person_hash_provider = (self.person_hash_provider +1)%1000000

        return {
            'id': person_id,
            'bbox':None,
            'pose_2d':None,
            'pose_3d':None,
            'confidence':np.array([0.25 for _ in range(PoseConfig.get_total_joints())]),
            'hash':self.person_hash_provider
        }




    """
    Background routine started in a thread in the init method
    used to manage incoming and outgoing people
    """
    def person_identification_routine(self):


        while not self.object_detector_kill_trigger:

            time.sleep(0.10)
            pid_to_remove, detected_boxes_to_add = set(), set()

            with self.persons_lock:
                curr_persons = [p for p in copy.deepcopy(self.persons).values() if p['pose_2d'] is not None]

            # filter pose with a too low confidence [pid_to_remove]

            for pid in range(len(curr_persons)):

                # score person confidence using most easily recognizable joints
                person_confidence = 5*curr_persons[pid]['confidence'][PoseConfig.HEAD]
                person_confidence += curr_persons[pid]['confidence'][PoseConfig.R_SHOULDER]
                person_confidence += curr_persons[pid]['confidence'][PoseConfig.L_SHOULDER]
                person_confidence += curr_persons[pid]['confidence'][PoseConfig.R_HIP]
                person_confidence += curr_persons[pid]['confidence'][PoseConfig.L_HIP]
                person_confidence += curr_persons[pid]['confidence'][PoseConfig.R_KNEE]
                person_confidence += curr_persons[pid]['confidence'][PoseConfig.L_KNEE]
                person_confidence /= 11

                # if confidence is too low : remove the person from the annotator
                if person_confidence < 0.25:
                    pid_to_remove.add(curr_persons[pid]['id'])

                # remove person sharing the same location in the screen
                # (avoid detecting twice the same person when tricky situation happens)
                for other_pid in range(pid+1, len(curr_persons)):

                    bbox1 = curr_persons[pid]['bbox']
                    bbox2 = curr_persons[other_pid]['bbox']
                    bbox_inter = bbox1.intersect(bbox2)

                    # is intersection empty
                    if bbox_inter is None:
                        continue

                    ratio1 = bbox_inter.get_width() * bbox_inter.get_height()
                    ratio1 /= bbox1.get_width() * bbox1.get_height()

                    ratio2 = bbox_inter.get_width() * bbox_inter.get_height()
                    ratio2 /= bbox2.get_width() * bbox2.get_height()

                    ratio = max(ratio1, ratio2)

                    # modified IoU > 0.85 => remove other_pid
                    if ratio > 0.85:
                        pid_to_remove.add(curr_persons[other_pid]['id'])


            # update the local version of persons
            curr_persons = [curr_persons[pid] for pid in range(len(curr_persons)) if curr_persons[pid]['id'] not in pid_to_remove]


            is_requiring_inference = self.last_image is not None
            is_requiring_inference = is_requiring_inference and len(self.available_person_id) > 0
            tmp = time.time() - self.last_object_detector_timestamp > AnnotatorInterface.EXTRA_OBJECT_DETECTION_DELAY
            is_requiring_inference = is_requiring_inference and tmp


            # add new incoming people [person_to_add]

            if is_requiring_inference:


                curr_bboxes = [p['bbox'] for p in curr_persons]

                new_bboxes, confidences = self.bbox_model.predict(self.last_image)


                # simple matching from old bbox to new :
                # return matches, unmachted from arg1, unmatched from arg2
                match_ids, unmatched_curr_ids, unmatched_new_ids = IdentityTracker.match_bboxes(curr_bboxes, new_bboxes)


                # add the most confident persons as long as an allocation available for it

                total_available_allocations = len(pid_to_remove) + len(self.available_person_id)

                tmp = [ [new_bboxes[i], confidences[i]] for i in unmatched_new_ids]
                tmp.sort(key=lambda x:x[1], reverse=True)
                for box in [box_and_conf[0] for box_and_conf in tmp][:total_available_allocations]:
                    detected_boxes_to_add.add(box)

                self.last_object_detector_timestamp = time.time()


            with self.persons_lock:

                # update removed persons
                for pid in pid_to_remove:
                    del self.persons[pid]
                    self.available_person_id.add(pid)

                for box in detected_boxes_to_add:
                    p = self._new_person(self.available_person_id.pop())
                    p['bbox'] = box
                    self.persons[p['id']] = p




    """ Terminates the object detection routine executed in background"""
    def terminate(self):
        self.object_detector_kill_trigger = True


    """
    Return the person's annotations
    """
    def get_persons(self):

        with self.persons_lock:
            persons = copy.deepcopy(self.persons)

        return [p for p in persons.values() if p['pose_2d'] is not None]




    """
    Update the pose with the given image and return the new annotation
    """
    def update(self, image):

        if len(image.shape) != 3:
            raise Exception("image need to be shaped as hxwx3 or hxwx4 for png")

        # remove alpha channel if any
        if image.shape[2] == 4:
            image = image[:,:,:3]


        # if there is at least one person to detect
        if len(self.persons) > 0:


            # update curr persons annotations in local
            with self.persons_lock:
                curr_persons = copy.deepcopy(self.persons)

            pids, bboxes, poses_2d = [], [], []
            for p in curr_persons.values():
                pids.append(p['id'])
                bboxes.append(p['bbox'])
                poses_2d.append(p['pose_2d'])

            new_poses_2d, confidences = self.pose_2d_model.predict(image, bboxes, poses_2d)
            confidences = np.array(confidences)
            new_poses_3d = self.pose_3d_model.predict(new_poses_2d)

            for i,pid in enumerate(pids):
                curr_persons[pid]['bbox'] = new_poses_2d[i].to_bbox()
                curr_persons[pid]['pose_2d'] = new_poses_2d[i]
                curr_persons[pid]['pose_3d'] = new_poses_3d[i]
                curr_persons[pid]['confidence'] = confidences[i]

            # update the annotations in the scope of the class
            with self.persons_lock:

                for curr_person in curr_persons.values():

                    # a person['id'] could be removed and a new person could added with the same label
                    is_same_person = curr_person['id'] in self.persons and self.persons[curr_person['id']]['hash'] == curr_person['hash']

                    # if possible add a bit of smoothing between predictions

                    if is_same_person and self.persons[curr_person['id']]['pose_2d'] is not None:

                        smoothed_joints_2d = 0.85 * curr_person['pose_2d'].get_joints() \
                                             + 0.15 * self.persons[curr_person['id']]['pose_2d'].get_joints()

                        # note 2d joints are smoothed twice (todo)
                        smoothed_joints_3d = 0.85 * curr_person['pose_3d'].get_joints() \
                                             + 0.15 * self.persons[curr_person['id']]['pose_3d'].get_joints()

                        smoothed_confidence = 0.85 * curr_person['confidence'] + 0.15 * self.persons[curr_person['id']]['confidence']

                        curr_person['pose_2d'] = Pose2D(smoothed_joints_2d)
                        curr_person['pose_3d'] = Pose3D(smoothed_joints_3d)
                        curr_person['confidence'] = smoothed_confidence


                    curr_person['bbox'] = curr_person['pose_2d'].to_bbox()

                    # does not update a people removed in between by the threaded routine
                    if is_same_person:
                        self.persons[curr_person['id']] = curr_person


        self.last_image = image

        return self.get_persons()





    """
    Return a json containing the annotations
    """
    def jsonify(self):

        persons = self.get_persons()
        annotations = []

        for i in range(len(persons)):

            joints2d = persons[i]['pose_2d'].get_joints()
            joints3d = persons[i]['pose_3d'].get_joints()

            annot = {'id':persons[i]['id'], 'pose_2d': {}, 'pose_3d': {}, 'confidence':persons[i]['confidence'].tolist()}

            for i in range(PoseConfig.get_total_joints()):
                joint2d = {'x': float(joints2d[i][0]), 'y': float(joints2d[i][1])}
                annot['pose_2d'][PoseConfig.NAMES[i]] = joint2d
                joint3d = {'x': float(joints3d[i][0]), 'y': float(joints3d[i][1]), 'z': float(joints3d[i][2])}
                annot['pose_3d'][PoseConfig.NAMES[i]] = joint3d

            annotations.append(annot)

        return json.dumps(annotations, ensure_ascii=False)



