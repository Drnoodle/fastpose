import numpy as np

"""
Bounding box defined min x, max x, min y, max y all in %
"""
class BBox():


    def __init__(self, min_x, max_x, min_y, max_y):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    """ Return the min x position in %"""
    def get_min_x(self, img=None):
        return self.min_x if img is None else int(self.min_x * img.shape[1])

    """ Return the max x position in %"""
    def get_max_x(self, img=None):
        return self.max_x if img is None else int(self.max_x * img.shape[1])

    """ Return the min y position in %"""
    def get_min_y(self, img=None):
        return self.min_y if img is None else int(self.min_y * img.shape[0])

    """ Return the max y position in %"""
    def get_max_y(self, img=None):
        return self.max_y if img is None else int(self.max_y * img.shape[0])

    """ Crop the given image on the bounding box field, zero pad if the box goes outside the frame"""
    def crop(self, np_img):

        min_x, max_x = int(max(self.get_min_x(np_img), 0)), int(min(self.get_max_x(np_img), np_img.shape[1]))
        min_y, max_y = int(max(self.get_min_y(np_img), 0)), int(min(self.get_max_y(np_img), np_img.shape[0]))

        cropped_img = np_img.copy()[min_y:max_y, min_x:max_x, :]

        if self.get_min_y() < 0:
            pad = np.zeros((-self.get_min_y(np_img), cropped_img.shape[1], cropped_img.shape[2]))
            cropped_img = np.vstack([pad, cropped_img])

        if self.get_max_y() > 1.0:
            pad = np.zeros((self.get_max_y(np_img) - np_img.shape[0], cropped_img.shape[1], cropped_img.shape[2]))
            cropped_img = np.vstack([cropped_img, pad])

        if self.get_min_x() < 0:
            pad = np.zeros((cropped_img.shape[0], -self.get_min_x(np_img), cropped_img.shape[2]))
            cropped_img = np.hstack([pad, cropped_img])

        if self.get_max_x() > 1.0:
            pad = np.zeros((cropped_img.shape[0], self.get_max_x(np_img) - np_img.shape[1], cropped_img.shape[2]))
            cropped_img = np.hstack([cropped_img, pad])

        return cropped_img


    """ Return the width size in %"""
    def get_width(self):
        return self.get_max_x() - self.get_min_x()

    """ Return the height size in %"""
    def get_height(self):
        return self.get_max_y() - self.get_min_y()


    """
    Return a new bounding box at the same center position with the same required extra padding
    Used at 0.3 very frequently
    """
    def get_with_padding(self, padding):

        center_x = float(self.get_max_x() + self.get_min_x()) / 2
        center_y = float(self.get_max_y() + self.get_min_y()) / 2

        width = (self.get_max_x() - self.get_min_x()) * (1 + padding)
        height = (self.get_max_y() - self.get_min_y()) * (1 + padding)

        min_x, max_x = center_x - 0.5 * width, center_x + 0.5 * width
        min_y, max_y = center_y - 0.5 * height, center_y + 0.5 * height

        return BBox(min_x, max_x, min_y, max_y)

    """
    Return a new bounding box at the same center position with the required extra padding.
    The size of the bounding box is adapted to make it squared over the aspect ratio of
    the given image
    """
    def to_squared(self, np_img, padding=0.0):

        box_size = max(self.get_max_x(np_img) - self.get_min_x(np_img), self.get_max_y(np_img) - self.get_min_y(np_img))
        box_size += box_size * padding

        center_x = float(self.get_max_x(np_img) + self.get_min_x(np_img)) / 2
        center_y = float(self.get_max_y(np_img) + self.get_min_y(np_img)) / 2

        min_x, max_x = center_x - 0.5 * box_size, center_x + 0.5 * box_size
        min_y, max_y = center_y - 0.5 * box_size, center_y + 0.5 * box_size

        min_x, max_x, min_y, max_y = min_x / np_img.shape[1], max_x / np_img.shape[1], min_y / np_img.shape[0], max_y / \
                                 np_img.shape[0]

        return BBox(min_x, max_x, min_y, max_y)


    """Return a new bounding box translated by trans_x, trans_y"""
    def translate(self, trans_x, trans_y):
        return BBox(self.min_x + trans_x, self.max_x + trans_x, self.min_y + trans_y, self.max_y + trans_y)


    """Return a new bounding box insuring that the value will belong to [0,1]"""
    def clip(self, min_bound, max_bound):
        minX = max(min(self.min_x, max_bound), min_bound)
        maxX = max(min(self.max_x, max_bound), min_bound)
        minY = max(min(self.min_y, max_bound), min_bound)
        maxY = max(min(self.max_y, max_bound), min_bound)
        return BBox(minX, maxX, minY, maxY)


    """
    Return a new bounding box corresponding to the intersection of this box with the given box,
    or none if the intersection is empty
    """
    def intersect(self, that_bbox):

        isEmpty = that_bbox.get_min_x() >= self.get_max_x()
        isEmpty = isEmpty or that_bbox.get_max_x() <= self.get_min_x()
        isEmpty = isEmpty or that_bbox.get_min_y() >= self.get_max_y()
        isEmpty = isEmpty or that_bbox.get_max_y() <= self.get_min_y()

        if isEmpty:
            return None

        maxX = min(that_bbox.get_max_x(), self.get_max_x())
        maxY = min(that_bbox.get_max_y(), self.get_max_y())
        minX = max(that_bbox.get_min_x(), self.get_min_x())
        minY = max(that_bbox.get_min_y(), self.get_min_y())

        return BBox(minX, maxX, minY, maxY)



    """Return a new BBox scaled by scale_x, scale_y"""
    def scale(self, scale_x, scale_y):
        return BBox(self.get_min_x() * scale_x, self.get_max_x() * scale_x, self.get_min_y() * scale_y, self.get_max_y() * scale_y)

    """Return true if the given x,y is inside the box """
    def is_inside(self, x, y):
        isInX = self.min_x <= x and x <= self.max_x
        isInY = self.min_y <= y and y <= self.max_y
        return isInX and isInY

    """Return the center position of the bbox"""
    def get_center_position(self):
        return (self.get_min_x() + self.get_max_x()) / 2.0, (self.get_min_y() + self.get_max_y()) / 2.0


    def __str__(self):

        res = "[minX:" + str(self.get_min_x()) + ", "
        res += "maxX:" + str(self.get_max_x()) + ", "
        res += "minY:" + str(self.get_min_y()) + ", "
        res += "maxY:" + str(self.get_max_y()) + "]"

        return res