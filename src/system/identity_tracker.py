import numpy as np



class IdentityTracker:


    """
    Simple match of 2 lists of bounding boxes by center proximity.
    Return matches as a dict id1 => id2, unmatched id in bbox1, unmatched id in bbox2
    """
    @staticmethod
    def match_bboxes(bboxes1, bboxes2):


        # build (box1, box2, distance^2) for each possible candidates

        candidates = []

        for id1 in range(len(bboxes1)):
            for id2 in range(len(bboxes2)):

                x1, y1 = bboxes1[id1].get_center_position()
                x2, y2 = bboxes2[id2].get_center_position()

                distance = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

                candidates.append([id1, id2, distance])



        # match the boxes by proximity

        match_mapping = {}

        while len(candidates) > 0:


            # select the current best match

            match = min(candidates, key= lambda x:x[2])
            match_id1, match_id2 = match[0], match[1]


            # filters the bbox id matched in the selected matches

            for i in reversed(range(len(candidates))):

                if candidates[i][0] == match_id1 or candidates[i][1] == match_id2:
                    del candidates[i]

            # record the match :

            match_mapping[match_id1] = match_id2


        # build the unmateched ids

        matchedIds1 = set(match_mapping.keys())
        matchedIds2 = set(match_mapping.values())

        unmatechedIds1 = [i for i in range(len(bboxes1)) if not i in matchedIds1]
        unmatechedIds2 = [i for i in range(len(bboxes2)) if not i in matchedIds2]


        return match_mapping, unmatechedIds1, unmatechedIds2

