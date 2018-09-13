import cv2

from src.utils.bbox import BBox
from .darknet import Darknet
from torch.autograd import Variable
from .cfg import parse_cfg
import torch

class YoloInterface:

    def __init__(self, cfgFile, weightFile, conf_thresh, nms_thresh):

        self.isCudaActivated = True

        parsedCfg = parse_cfg(cfgFile)
        self.model = Darknet(parsedCfg)
        self.model.print_network()
        self.model.load_weights(weightFile)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        if torch.cuda.is_available() and self.isCudaActivated:
            self.model.cuda()


        for param in self.model.parameters():
            param.requires_grad = False


    def setConfidenceThreshold(self, conf_thresh):
        self.conf_thresh = conf_thresh

    def setNmsThreshold(self, nms_thresh):
        self.nms_thresh = nms_thresh


    """
    Person Detection

    Args:
        * img : numpy array of size : width*height*rgb with values belonging to [0,255]

    Return:
        * a list of src.utils.BBox
        * a list of confidence

    """

    def predict(self, img):


        if len(img.shape) != 3 or img.shape[0] == 0 or img.shape[1] == 0 or img.shape[2] != 3:
            raise Exception("wrong image shape : " + str(img.shape))

        if img.shape[1] != self.model.width or img.shape[0] != self.model.height:
            # todo gpu
            img = cv2.resize(img, (self.model.width, self.model.height))

        img = torch.from_numpy(img.transpose(2, 0, 1))

        if torch.cuda.is_available() and self.isCudaActivated:
            img = img.cuda()

        img = img.float()
        # TODO in place
        img = img.div(255.0)
        img = img.unsqueeze(0)
        img = torch.autograd.Variable(img)

        output = self.model(img)
        output = output.data
        boxes = self.get_region_boxes(output,
                                      self.conf_thresh,
                                      self.model.num_classes,
                                      self.model.anchors,
                                      self.model.num_anchors)[0]

        boxes = self.nms(boxes, self.nms_thresh)

        # modify boxes such that :
        # object_detection format <centerX in %, centerY in %, width in %, height in %, conf in %> => <minX in %, minY in %, maxY in %, maxY in %, conf in %>
        # predict : List<{'minX':minX, 'minY':minY, 'maxX':maxX, 'maxY':maxY, 'confidence':box[4]}> in pix
        processed_boxes = []

        for box in boxes:
            # box size in px
            width, height = float(box[2].data), float(box[3].data)
            centerX, centerY = float(box[0].data), float(box[1].data)

            minX, maxX = (centerX - width / 2), (centerX + width / 2)
            minY, maxY = (centerY - height / 2), (centerY + height / 2)

            processed_boxes.append((BBox(minX, maxX, minY, maxY), box[4].data.tolist()))

        processed_boxes = sorted(processed_boxes, key=lambda box: box[1], reverse=True)


        return [b[0] for b in processed_boxes], [b[1] for b in processed_boxes]


    def convert2cpu(self, gpu_matrix):
        return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

    def convert2cpu_long(self, gpu_matrix):
        return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

    def get_region_boxes(self, output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False):

        anchor_step = int(len(anchors)/num_anchors)

        if output.dim() == 3:
            output = output.unsqueeze(0)

        batch = output.size(0)
        assert(output.size(1) == (5+num_classes)*num_anchors)
        h = output.size(2)
        w = output.size(3)

        all_boxes = []
        output = output.view(batch*num_anchors, 5+num_classes, h*w).transpose(0,1).contiguous().view(5+num_classes, batch*num_anchors*h*w)


        grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w)
        grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w)

        if torch.cuda.is_available() and self.isCudaActivated:
            grid_x = grid_x.cuda()
            grid_y = grid_y.cuda()

        xs = torch.sigmoid(output[0]) + grid_x
        ys = torch.sigmoid(output[1]) + grid_y

        anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
        anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
        anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w)
        anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w)

        if torch.cuda.is_available() and self.isCudaActivated:
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()


        ws = torch.exp(output[2]) * anchor_w
        hs = torch.exp(output[3]) * anchor_h

        det_confs = torch.sigmoid(output[4])

        cls_confs = torch.nn.Softmax()(Variable(output[5:5+num_classes].transpose(0,1))).data
        cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
        cls_max_confs = cls_max_confs.view(-1)
        cls_max_ids = cls_max_ids.view(-1)

        sz_hw = h*w
        sz_hwa = sz_hw*num_anchors
        det_confs = self.convert2cpu(det_confs)
        cls_max_confs = self.convert2cpu(cls_max_confs)
        cls_max_ids = self.convert2cpu_long(cls_max_ids)
        xs = self.convert2cpu(xs)
        ys = self.convert2cpu(ys)
        ws = self.convert2cpu(ws)
        hs = self.convert2cpu(hs)
        if validation:
            cls_confs = self.convert2cpu(cls_confs.view(-1, num_classes))

        for b in range(batch):
            boxes = []
            for cy in range(h):
                for cx in range(w):
                    for i in range(num_anchors):
                        ind = b*sz_hwa + i*sz_hw + cy*w + cx
                        det_conf =  det_confs[ind]
                        if only_objectness:
                            conf =  det_confs[ind]
                        else:
                            conf = det_confs[ind] * cls_max_confs[ind]

                        if conf > conf_thresh:
                            bcx = xs[ind]
                            bcy = ys[ind]
                            bw = ws[ind]
                            bh = hs[ind]
                            cls_max_conf = cls_max_confs[ind]
                            cls_max_id = cls_max_ids[ind]
                            box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
                            if (not only_objectness) and validation:
                                for c in range(num_classes):
                                    tmp_conf = cls_confs[ind][c]
                                    if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh:
                                        box.append(tmp_conf)
                                        box.append(c)
                            boxes.append(box)
            all_boxes.append(boxes)

        return all_boxes


    def nms(self, boxes, nms_thresh):
        if len(boxes) == 0:
            return boxes

        det_confs = torch.zeros(len(boxes))
        for i in range(len(boxes)):
            det_confs[i] = 1-boxes[i][4]

        _,sortIds = torch.sort(det_confs)
        out_boxes = []
        for i in range(len(boxes)):
            box_i = boxes[sortIds[i]]
            if box_i[4] > 0:
                out_boxes.append(box_i)
                for j in range(i+1, len(boxes)):
                    box_j = boxes[sortIds[j]]
                    if self.bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                        box_j[4] = 0
        return out_boxes

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        if x1y1x2y2:
            mx = min(box1[0], box2[0])
            Mx = max(box1[2], box2[2])
            my = min(box1[1], box2[1])
            My = max(box1[3], box2[3])
            w1 = box1[2] - box1[0]
            h1 = box1[3] - box1[1]
            w2 = box2[2] - box2[0]
            h2 = box2[3] - box2[1]
        else:
            mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
            Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
            my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
            My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
            w1 = box1[2]
            h1 = box1[3]
            w2 = box2[2]
            h2 = box2[3]
        uw = Mx - mx
        uh = My - my
        cw = w1 + w2 - uw
        ch = h1 + h2 - uh

        if cw <= 0 or ch <= 0:
            return 0.0

        area1 = w1 * h1
        area2 = w2 * h2
        carea = cw * ch
        uarea = area1 + area2 - carea
        return carea/uarea



