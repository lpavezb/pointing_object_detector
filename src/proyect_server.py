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
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from image_geometry import PinholeCameraModel
from datetime import datetime

class Bbox:
    def __init__(self, bbox, tag):
        self.bbox = bbox
        self.tag = tag
        self.distance = 0

    def contains(self, x, y):
        l_x = self.bbox[0]
        r_x = self.bbox[2]

        u_y = self.bbox[1]
        d_y = self.bbox[3]

        if l_x < x < r_x:
            if u_y < y < d_y:
                return True
        return False 

    def center(self):
    	return [int((self.bbox[2] - self.bbox[0]) / 2), int((self.bbox[3] - self.bbox[1]) / 2)]

def detect_pointing_object_handler(req):
    rospack = rospkg.RosPack()
    models  = rospack.get_path('pointing_object_detector') + "/models"
    images  = rospack.get_path('pointing_object_detector') + "/images"
    camera_model = PinholeCameraModel()

    # img = images + "/image2.png"
    
    # image = cv2.imread(img)
    
    rospy.sleep(2)
    rospy.logwarn("START")
    depthmsg = rospy.wait_for_message("/maqui/camera/depth_registered/image_rect", Image)
    infodepth = rospy.wait_for_message("/maqui/camera/depth_registered/camera_info", CameraInfo)

    bridge = CvBridge()
    try:
        depth = bridge.imgmsg_to_cv2(depthmsg, "16UC1")
    except CvBridgeError as e:
        print(e)


    imagemsg = rospy.wait_for_message("/maqui/camera/front/image_raw", Image)
    infoimage = rospy.wait_for_message("/maqui/camera/front/camera_info", CameraInfo)
    camera_model.fromCameraInfo(infoimage)

    try:
        image = bridge.imgmsg_to_cv2(imagemsg, "bgr8")
    except CvBridgeError as e:
        print(e)

    date = datetime.now().strftime("%Y%m%d%H%M%S")
    cv2.imwrite(images + "/" + date + "_image.png", image)
    
    MODEL_PATH = models + "/multi_pose_dla_3x.pth"
    TASK = 'multi_pose' # or 'multi_pose' for human pose estimation
    opt  = opts().init('{} --load_model {} --vis_thresh 0.5'.format(TASK, MODEL_PATH).split(' '))
    detector = detector_factory[opt.task](opt)


    results = detector.run(image)['results']
    sq = []
    for bbox in results[1]:
        if bbox[4] > detector.opt.vis_thresh:
          sq = bbox[5:39]

    if len(sq) == 0:
        rospy.loginfo("Person not detected")
        return ObjectDetectionResponse()

    points = np.array(sq, dtype=np.int32).reshape(17, 2)

    ratioWristElbow = 0.33
    image_height, image_width = image.shape[0:2]

    x_shoulder, y_shoulder = (points[6, 0], points[6, 1])   # shoulder
    x_elbow, y_elbow       = (points[8, 0], points[8, 1])   # elbow
    x_wrist, y_wrist       = (points[10, 0], points[10, 1]) # wrist

    x = x_wrist + ratioWristElbow * (x_wrist - x_elbow)
    y = y_wrist + ratioWristElbow * (y_wrist - y_elbow)
    distanceWristElbow    = math.sqrt((x_wrist - x_elbow) ** 2 + (y_wrist - y_elbow) ** 2)
    distanceElbowShoulder = math.sqrt((x_elbow - x_shoulder) ** 2 + (y_elbow - y_shoulder) ** 2)
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

    #cv2.imwrite(images + "/" + date + "_hand.png", image[y:y+w, x:x+w, :])

    
    peaks = hand_estimation(image[y:y+w, x:x+w, :])
    #cv2.imwrite(images + "/" + date + "_hand_sq.png", util.draw_handpose(image[y:y+w, x:x+w, :], [peaks]))
    
    peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
    peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)


    if peaks.all() == 0:
        rospy.loginfo("No hand detected")
        return ObjectDetectionResponse()

    #image = cv2.resize(image, (image.shape[1]/2, image.shape[0]/2))
    obj_left = (x_wrist - x_shoulder) < 0
    if obj_left:
        crop = image[:,:x_wrist]
    else:
        crop = image[:,x_wrist:]

    #cv2.imwrite(images + "/" + date + "_crop.png", crop)
    

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


    for bbox in bbxs:
        box = bbox.bbox
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), 0, 2)
        cv2.rectangle(crop, (box[0], box[1]), (box[2], box[3]), 0, 2)

    #cv2.imwrite(images + "/" + date + "_crop_bbox.png", crop)

    x1, y1 = peaks[6]
    x2, y2 = peaks[8]

    m = 1.0 * (y2 - y1) / (x2 - x1)
    n = y1 - x1 * m

    if x1 > x2:
        rang = range(x2, 0, -1)
    else:
        rang = range(x2, image.shape[1])


    x_wrist, y_wrist = peaks[6] 
    x_finger, y_finger = peaks[8]

    z_wrist = depth[:, x_wrist-1:x_wrist+1]
    z_wrist = np.min(np.where(z_wrist == 0, 90000, z_wrist))
    
    z_finger = depth[:, x_finger-1:x_finger+1]
    z_finger = np.min(np.where(z_finger == 0, 90000, z_finger))
    

    (x_wrist, y_wrist) = camera_model.rectifyPoint((x_wrist, y_wrist))
    wrist = camera_model.projectPixelTo3dRay((x_wrist, y_wrist))
    wrist = np.array([wrist[0], wrist[1], z_wrist])

    x_finger, y_finger = camera_model.rectifyPoint((x_finger, y_finger))
    finger = camera_model.projectPixelTo3dRay((x_finger, y_finger))
    finger = np.array([finger[0], finger[1], z_finger])


    print "wrist"
    print wrist
    print "finger"
    print finger


    tag = ""
    end = False
    objbbox = []
    for bbox in bbxs:
        print bbox.tag

        x_obj, y_obj = bbox.center()
        z_obj = depth[y_obj-2:y_obj+2:, x_obj-2:x_obj+2]
        z_obj = np.min(np.where(z_obj == 0, 90000, z_obj))

        x_obj, y_obj = camera_model.rectifyPoint((x_obj, y_obj))
        obj = camera_model.projectPixelTo3dRay((x_obj, y_obj))
        onj = np.array([obj[0], obj[1], z_obj])

        cross = np.cross((obj - finger), (obj - wrist))
        num = np.linalg.norm(cross)
        bbox.distance = num / np.linalg.norm(wrist - finger)

    max_dist = -1
    for x in rang:
        y = m * x + n
        for bbox in bbxs:
            if bbox.contains(x, y) and (bbox.distance < max_dist or max_dist == -1):
                max_dist = bbox.distance
                tag = bbox.tag
                objbbox = bbox.bbox 
                end = True
        if end:
            break

    if tag == "":
        rospy.loginfo("Object pointed not detected")
        return ObjectDetectionResponse()

    for bbox in bbxs:
        print bbox.tag
        print bbox.distance
        
    # print "object"
    # print m2 * center[0] + n2
    # print np.max(depth[center[1]-1:center[1]+1, center[0]-1:center[0]+1])

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
    #image = cv2.resize(image, (image.shape[1]/2, image.shape[0]/2))

    cv2.line(image, (x2, y2), (int(x), int(y)), (255, 0, 0), 2)
    cv2.imwrite(images + "/" + date + "_result.png", image)
    #cv2.imwrite(images + "/result.png" , image)
    return response


def detect_pointing_object_server():
    rospy.init_node('detect_pointing_object_server')
    s = rospy.Service('detect_pointing_object', ObjectDetection, detect_pointing_object_handler)
    rospy.spin()

if __name__ == "__main__":
    detect_pointing_object_server()