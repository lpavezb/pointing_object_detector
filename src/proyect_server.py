#!/usr/bin/env python
from centernet.src.lib.detectors.detector_factory import detector_factory
from opts import opts
import numpy as np
import cv2
import math

from hand import Hand
import util

import rospy
import rospkg
from uchile_srvs.srv import ObjectDetection, ObjectDetectionResponse
from uchile_msgs.msg import Rect
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime

class Bbox:
    def __init__(self, bbox, tag):
        self.bbox = bbox
        self.tag = tag

    def contains(self, x, y):
        l_x = self.bbox[0]
        r_x = self.bbox[2]

        u_y = self.bbox[1]
        d_y = self.bbox[3]

        if l_x < x < r_x:
            if u_y < y < d_y:
                return True
        return False 


def detect_pointing_object_handler(req):
    rospack = rospkg.RosPack()
    models  = rospack.get_path('pointing_object_detector') + "/models"
    images  = rospack.get_path('pointing_object_detector') + "/images"

    img = images + "/image2.png"
    
    image = cv2.imread(img)
    
    # rospy.sleep(2)
    # rospy.logwarn("START")
    # imagemsg = rospy.wait_for_message("/maqui/camera/front/image_raw", Image)

    # try:
    #     image = CvBridge().imgmsg_to_cv2(imagemsg, "bgr8")
    # except CvBridgeError as e:
    #     print(e)

    #cv2.imwrite(images + "/" + datetime.now().strftime("%Y%m%d%H%M%S") + "_image.png", image)
    MODEL_PATH = models + "/multi_pose_dla_3x.pth"
    TASK = 'multi_pose' # or 'multi_pose' for human pose estimation
    opt  = opts().init('{} --load_model {} --vis_thresh 0.5'.format(TASK, MODEL_PATH).split(' '))
    detector = detector_factory[opt.task](opt)


    results = detector.run(img)['results']
    sq = []
    for bbox in results[1]:
        if bbox[4] > detector.opt.vis_thresh:
          sq = bbox[5:39]

    #sq = [611.8907470703125, 7.758312225341797, 612.183349609375, -3.5637474060058594, 618.2316284179688, -3.8290023803710938, 608.5698852539062, -19.117870330810547, 639.1412963867188, -19.960002899169922, 596.852783203125, 21.959877014160156, 631.96826171875, 26.92244529724121, 466.5745849609375, 141.83990478515625, 466.5745849609375, 141.83990478515625, 341.8582458496094, 202.49754333496094, 341.8582458496094, 202.49754333496094, 591.8981323242188, 372.029541015625, 612.02294921875, 381.86041259765625, 583.378173828125, 501.3885498046875, 598.6640625, 521.9430541992188, 551.1273193359375, 502.0205078125, 589.4862060546875, 478.24517822265625]

    if len(sq) == 0:
        rospy.loginfo("Person not detected")
        return ObjectDetectionResponse()

    points = np.array(sq, dtype=np.int32).reshape(17, 2)

    ratioWristElbow = 0.33
    image_height, image_width = image.shape[0:2]

    x1, y1 = (points[6, 0], points[6, 1])   # shoulder
    x2, y2 = (points[8, 0], points[8, 1])   # elbow
    x3, y3 = (points[10, 0], points[10, 1]) # wrist

    x = x3 + ratioWristElbow * (x3 - x2)
    y = y3 + ratioWristElbow * (y3 - y2)
    distanceWristElbow    = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)

    x -= width / 2
    y -= width / 2  # width = height

    if x < 0: x = 0
    if y < 0: y = 0
    width1 = width
    width2 = width
    if x + width > image_width: width1 = image_width - x
    if y + width > image_height: width2 = image_height - y
    width = min(width1, width2)



    hand_estimation = Hand(models + '/hand_pose_model.pth')

    x = int(x)
    y = int(y)
    w = int(width)
    cv2.imshow("hand", image[y:y+w, x:x+w, :])
    cv2.waitKey(0)
    peaks = hand_estimation(image[y:y+w, x:x+w, :])
    peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
    peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)

    #peaks = np.array([[336, 207],[327, 221],[315, 231],[304, 244],[296, 253],[296, 225],[276, 235],[266, 243],[256, 250],[297, 228],[288, 248],[296, 259],[307, 262],[302, 230],[293, 249],[298, 259],[308, 263],[307, 233],[304, 246],[309, 252],[318, 250]])
    if peaks.all() == 0:
        rospy.loginfo("No hand detected")
        return ObjectDetectionResponse()

    is_left = (x3 - x1) < 0
    if is_left:
        crop = image[:,:x3]
    else:
        crop = image[:,x3:]


    MODEL_PATH = models + "/ctdet_coco_dla_2x.pth"
    TASK = 'ctdet' # or 'multi_pose' for human pose estimation
    opt  = opts().init('{} --load_model {} --vis_thresh 0.5'.format(TASK, MODEL_PATH).split(' '))
    detector = detector_factory[opt.task](opt)
    results  = detector.run(crop)['results']
    
    coco_class_name = [
         'person', 'bicycle', 'car', 'motorcycle', 'airplane',
         'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
         'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
         'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
         'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
         'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
         'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
         'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
         'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
         'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    bbxs = []
    for j in range(2, 80 + 1):
        for bbox in results[j]:
            if bbox[4] > detector.opt.vis_thresh:
                bbxs.append(Bbox(bbox[:4], coco_class_name[j-1]))


    # bbxs = [
    # Bbox(np.array([148.29974, 249.3421,  187.45439, 376.2686 ], dtype=np.int32), "bottle"),
    # Bbox(np.array([230.92738, 285.68988, 287.7627,  346.40366], dtype=np.int32), "cup"),
    # ]

    for bbox in bbxs:
        box = bbox.bbox
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), 0, 2)

    x1, y1 = peaks[6]
    x2, y2 = peaks[8]

    m = 1.0 * (y2 - y1) / (x2 - x1)
    n = y1 - x1 * m


    if x1 > x2:
        rang = range(x2, 0, -1)
    else:
        rang = range(x2, image.shape[1])

    tag = ""
    end = False
    objbbox = []
    for x in rang:
        y = m * x + n
        for bbox in bbxs:
            if bbox.contains(x, y):
                tag = bbox.tag
                objbbox = bbox.bbox 
                end = True
        if end:
            break

    if tag == "":
        rospy.loginfo("Object pointed not detected")
        return ObjectDetectionResponse()

    response = ObjectDetectionResponse()

    rect = Rect()
    rect.x = objbbox[0]
    rect.y = objbbox[1]
    rect.width  = objbbox[2] - objbbox[0]
    rect.height = objbbox[3] - objbbox[1]

    response.label.append(tag)
    response.poses.append(PoseStamped()) # TODO: detect PoseStamped
    response.BBoxes.append(rect)

    image = util.draw_handpose(image, [peaks])
    cv2.line(image, (x2, y2), (int(x), int(y)), (255, 0, 0), 3)
    cv2.imwrite(images + "/" + datetime.now().strftime("%Y%m%d%H%M%S") + "_result.png", image)
    #cv2.imwrite(images + "/result.png" , image)
    return response


def detect_pointing_object_server():
    rospy.init_node('detect_pointing_object_server')
    s = rospy.Service('detect_pointing_object', ObjectDetection, detect_pointing_object_handler)
    rospy.spin()

if __name__ == "__main__":
    detect_pointing_object_server()