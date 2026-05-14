# !/usr/bin/python3

import numpy as np
import cv2 as cv
import pickle
import collections
import os
import time as _time_mod
from time import time
import names_and_constants as nac
from ultralytics import YOLO
from stopline import detect_angle

# onnxruntime is preferred over cv.dnn for the lane keeper and YOLO,
# because it gives us GPU (CUDAExecutionProvider) support on the Jetson
# Orin Nano without rebuilding OpenCV.  It is imported lazily so that
# the simulator path (CPU-only laptop) can still run without onnxruntime
# installed.
try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except Exception:
    ort = None
    _ORT_AVAILABLE = False

LANE_KEEPER_PATH = "models/lane_keeper_small.onnx" #main model for lane keeping
# LANE_KEEPER_PATH = "models/round_about8001.onnx"
# avg right -0.17356677295197556
DISTANCE_POINT_AHEAD = 0.35
CAR_LENGTH = 0.4

LANE_KEEPER_AHEAD_PATH = "models/lane_keeper_ahead.onnx" # speed challange
# LANE_KEEPER_AHEAD_PATH = "models/round_about8001.onnx"
INTERSECTION_NAVIGATOR_RIGHT = "models/he_right.onnx"
INTERSECTION_NAVIGATOR_LEFT = "models/he_left.onnx" 
INTERSECTION_NAVIGATOR_FORWARD = "models/he_straight.onnx"
ROUNDABOUT_NAVIGATOR_IN = "models/he_right.onnx"
ROUNDABOUT_NAVIGATOR_ABOUT = "models/about_dif.onnx"
ROUNDABOUT_NAVIGATOR_OUT = "models/he_right.onnx"
# LANE_KEEPER_AHEAD_PATH = "models/round_in.onnx"
DISTANCE_POINT_AHEAD_AHEAD = 0.6

STOPLINE_ESTIMATOR_PATH = "models/stopline_estimator.onnx"

STOPLINE_ESTIMATOR_ADV_PATH = "models/stopline_estimator_advanced.onnx"
PREDICTION_OFFSET = -0.08
#yolo sign classifier
SIGN_CLASSIFIER_YOLO_PATH = "models/sign_classifier_yolo.tflite"

# SIGN_CLASSIFIER_PATH = 'models/sign_classifier.onnx'

KERNEL_TYPE_SIGNS = 'linear'
NUM_CLUSTERS_SIGNS = 100
NO_SIGN = 'NO_sign'
signs_dict = {
        "0": "stop",
        "1": "closed_road",
        "2": "park",
        "3": "cross_walk",
        "4": "one_way",
        "5": "hw_enter",
        "6": "hw_exit",
        "7": "priority",
        "8": "roundabout",
        "9": NO_SIGN}
MAP_DICT_NAMES = [4, 1, 0, 7, 8, 3, 2, 6, 5, 9]  # BFMC_2023


# Obstacle classifier
NUM_CLUSTERS_OBS = 1200
KERNEL_TYPE_OBS = 'linear'
obstacles_dict = {
    "0": "car",
    "1": "pedestrian",
    "2": "roadblock",
}

distance_dict = {
    '20': [(94, 40), (227, 140)],
    '30': [(103, 25), (217, 111)],
    '40': [(116, 23), (205, 90)],
    '50': [(120, 20), (200, 80)]
}


# ─────────────────────────────────────────────────────────────────────────
# YOLOv8 sign / object detector (replaces the BoW+SVM `detect_sign` that
# used SIFT keypoints + a linear-SVM classifier on a hardcoded ROI in
# the top-right corner of the frame).
#
# The model and the class list are imported from the standalone
# `traffic_sign` ROS 2 workspace that has already been tested on the
# car.  We keep the same class indices so the trained weights
# (`models/best2.onnx`) can be dropped in unchanged.
# ─────────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH = "models/sign_yolo.onnx"

YOLO_CLASS_NAMES = [
    'stop',              # 0
    'parking',           # 1
    'priority',          # 2
    'roundabout',        # 3
    'one_way',           # 4
    'no_entry',          # 5
    'exit_highway',      # 6
    'entrance_highway',  # 7
    'crosswalk',         # 8
]

# Mapping from YOLO class names → the legacy sign vocabulary used in the
# brain (`names_and_constants.py`).  Anything not in this map is reported
# verbatim — useful for `pedestrian`, `car`, `roadblock` which the brain
# treats as obstacles, not signs.
YOLO_TO_LEGACY_SIGN = {
    'stop':             'stop',
    'parking':          'park',
    'priority':         'priority',
    'roundabout':       'roundabout',
    'one_way':          'one_way',
    'no_entry':         'closed_road',
    'exit_highway':     'hw_exit',
    'entrance_highway': 'hw_enter',
    'crosswalk':        'cross_walk',
}

YOLO_CONF_THRESHOLD = 0.60
YOLO_IOU_THRESHOLD  = 0.50
YOLO_IMGSZ          = 640

# Stereo-depth windowing (matches `traffic_sign_node.py`).
DEPTH_MIN_MM = 100
DEPTH_MAX_MM = 2500


class Detection:

    # init
    def __init__(self) -> None:

        # lane following
        self.lane_keeper = cv.dnn.readNetFromONNX(LANE_KEEPER_PATH)
        self.lane_cnt = 0
        self.avg_lane_detection_time = 0

        self.avg_stopline_detection_adv_time = 0
        self.stopline_adv_cnt = 0

        
        # ── Helper: load secondary models via onnxruntime (CUDA) with cv.dnn fallback ──
        def _load(path, label):
            if _ORT_AVAILABLE and os.path.isfile(path):
                try:
                    sess = ort.InferenceSession(path, providers=self._ort_providers)
                    return sess, sess.get_inputs()[0].name
                except Exception as e:
                    print(f'[detection] {label} ort failed ({e}), using cv.dnn')
            return cv.dnn.readNetFromONNX(path), None

        # speed challenge
        self.lane_keeper_ahead,         self._la_in   = _load(LANE_KEEPER_AHEAD_PATH,         'lane_keeper_ahead')
        self.lane_ahead_cnt = 0
        self.avg_lane_ahead_detection_time = 0

        # intersection navigators
        self.intersection_navigator_right,   self._ir_in = _load(INTERSECTION_NAVIGATOR_RIGHT,   'inters_right')
        self.intersection_right_cnt = 0
        self.avg_intersection_right_detection_time = 0

        self.intersection_navigator_left,    self._il_in = _load(INTERSECTION_NAVIGATOR_LEFT,    'inters_left')
        self.intersection_left_cnt = 0
        self.avg_intersection_left_detection_time = 0

        self.intersection_navigator_forward, self._if_in = _load(INTERSECTION_NAVIGATOR_FORWARD, 'inters_fwd')
        self.intersection_forward_cnt = 0
        self.avg_intersection_forward_detection_time = 0

        # roundabout navigators
        self.roundabout_navigator_in,    self._ri_in = _load(ROUNDABOUT_NAVIGATOR_IN,    'round_in')
        self.roundabout_in_cnt = 0
        self.avg_roundabout_in_detection_time = 0

        self.roundabout_navigator_about, self._ra_in = _load(ROUNDABOUT_NAVIGATOR_ABOUT, 'round_about')
        self.roundabout_about_cnt = 0
        self.avg_roundabout_about_detection_time = 0

        self.roundabout_navigator_out,   self._ro_in = _load(ROUNDABOUT_NAVIGATOR_OUT,   'round_out')
        self.roundabout_out_cnt = 0
        self.avg_roundabout_out_detection_time = 0

        # stop line detection
        self.stopline_estimator,     self._sl_in  = _load(STOPLINE_ESTIMATOR_PATH,     'stopline')
        self.est_dist_to_stopline = 1.0
        self.avg_stopline_detection_time = 0
        self.stopline_cnt = 0

        # stop line detection advanced
        self.stopline_estimator_adv, self._sla_in = _load(STOPLINE_ESTIMATOR_ADV_PATH, 'stopline_adv')
        self.est_dist_to_stopline_adv = 1.0
        self.avg_stopline_detection_adv_time = 0
        self.stopline_adv_cnt = 0

        # ── Sign / object classifier ─────────────────────────────────────
        # The legacy SIFT + BoW + linear-SVM classifier is replaced by
        # YOLOv8 (best2.onnx) running through onnxruntime.  YOLO produces
        # per-class bounding boxes with confidences, and we estimate the
        # distance to each detection by reading the median depth of the
        # bbox centre from the OAK-D stereo `depth_frame` (in mm).
        #
        # The downstream API of `detect_sign(frame, ...)` is preserved:
        # it still returns `(sign_str, confidence)` so brain.py needs no
        # changes.  The full per-class result (with bbox + distance) is
        # additionally cached in `self.last_yolo_detections` for use by
        # `sign_detection_position()` and any future routine that wants
        # depth-aware sign placement.
        self.sign_yolo = None
        self._yolo_input_name = None
        if _ORT_AVAILABLE and os.path.isfile(YOLO_MODEL_PATH):
            try:
                self.sign_yolo = ort.InferenceSession(
                    YOLO_MODEL_PATH, providers=self._ort_providers)
                self._yolo_input_name = self.sign_yolo.get_inputs()[0].name
                print(f'[detection] sign_yolo providers: '
                      f'{self.sign_yolo.get_providers()}')
            except Exception as e:
                print(f'[detection] CUDA EP unavailable for sign_yolo '
                      f'({e}), falling back to CPU.')
                try:
                    self.sign_yolo = ort.InferenceSession(
                        YOLO_MODEL_PATH, providers=['CPUExecutionProvider'])
                    self._yolo_input_name = \
                        self.sign_yolo.get_inputs()[0].name
                except Exception as e2:
                    print(f'[detection] sign_yolo failed to load: {e2}')
                    self.sign_yolo = None
        else:
            print(f'[detection] sign_yolo not loaded '
                  f'(onnxruntime={_ORT_AVAILABLE}, '
                  f'model_exists={os.path.isfile(YOLO_MODEL_PATH)})')

        # Cache of the most recent YOLO output (list of dicts):
        # {cls, sign, conf, x1, y1, x2, y2, distance_m, stamp}
        self.last_yolo_detections = []
        # Most recent single-best detection (the legacy `(sign, conf)`
        # return value of detect_sign).
        self.sign_prediction = NO_SIGN
        self.sign_conf       = 0.0
        # For book-keeping, in case a routine wants to know how stale the
        # cached YOLO output is before trusting it.
        self._last_yolo_stamp = 0.0
        # Letterbox state used to un-pad the YOLO bboxes.
        self._lb_scale = 1.0
        self._lb_pad   = (0, 0)

        # obstacle classifier (kept untouched — `control_for_car` and
        # `classify_frontal_obstacle` still use it as a secondary signal
        # alongside the LIDAR cones).
        self.no_clusters = NUM_CLUSTERS_OBS  # use const
        self.kernel_type = KERNEL_TYPE_OBS  # const
        self.svm_model = pickle.load(open('models/obstacle_models/svm_' +
                                          self.kernel_type + '_' +
                                          str(self.no_clusters) +
                                          '.pkl', 'rb'))
        self.kmean_model = pickle.load(open('models/obstacle_models/kmeans_' +
                                            self.kernel_type + '_' +
                                            str(self.no_clusters) +
                                            '.pkl', 'rb'))
        self.scale_model = pickle.load(open('models/obstacle_models/scale_' +
                                            self.kernel_type + '_' +
                                            str(self.no_clusters) +
                                            '.pkl', 'rb'))
        self.sift = cv.ORB_create(nfeatures=100)
        self.prediction = None
        self.conf = 0

    def detect_lane(self, frame, show_ROI=True, faster=False):
        """
        Estimates:
          - the lateral error wrt the center of the lane (e2),
            - the angular error around the yaw axis wrt a fixed point ahead (e3),
            - the ditance from the next stop line (1/dist)
        """
        start_time = time()
        IMG_SIZE = (32, 32)  # match with trainer

        # convert to gray
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = frame[int(frame.shape[0]/3):, :]  # /3
        # keep the bottom 2/3 of the image
        frame = cv.resize(frame, (2*IMG_SIZE[0], 2*IMG_SIZE[1]))
        #frame = cv.blur(frame, (7,7), 0)
        frame = cv.Canny(frame, 100, 200)
        frame = cv.blur(frame, (3, 3), 0)  # worse than blur after 11,11
        frame = cv.resize(frame, IMG_SIZE)

        images = frame


        # ── Build the (1,1,32,32) or (2,1,32,32) blob ─────────────────────
        if faster:
            lob = cv.dnn.blobFromImage(images, 1.0, IMG_SIZE, 0, swapRB=True, crop=False)
        else:
            frame_flip = cv.flip(frame, 1)
            # stack the 2 images
            images = np.stack((frame, frame_flip), axis=0)
            blob = cv.dnn.blobFromImages(images, 1.0, IMG_SIZE, 0, swapRB=True, crop=False)
        self.lane_keeper.setInput(blob)
        out = -self.lane_keeper.forward()  # NOTE: MINUS SIGN IF OLD NET
        output = out[0]
        output_flipped = out[1] if not faster else None

        # <++>
        e2 = output[0]
        e3 = output[1]

        if not faster:
            e2_flipped = output_flipped[0]
            e3_flipped = output_flipped[1]

            e2 = (e2 - e2_flipped) / 2
            e3 = (e3 - e3_flipped) / 2

        # ── Build the visual point-ahead in vehicle frame coordinates ─────
        d = DISTANCE_POINT_AHEAD
        est_point_ahead = np.array([np.cos(e3) * d + 0.2, np.sin(e3) * d])

        lane_detection_time = 1000 * (time() - start_time)
        self.avg_lane_detection_time = \
            (self.avg_lane_detection_time * self.lane_cnt + lane_detection_time) \
            / (self.lane_cnt + 1)
        self.lane_cnt += 1
        if show_ROI:
            cv.imshow('lane_detection', frame)
            # cv.waitKey(1)  
            #                                     interpolation=cv.INTER_NEAREST))
        return e2, e3, est_point_ahead

    def detect_lane_ahead(self, frame, show_ROI=True, faster=False):
        """
        Estimates:
        - the lateral error wrt the center of the lane (e2),
        - the angular error around the yaw axis wrt a fixed point ahead (e3),
        - the ditance from the next stop line (1/dist)
        """
        start_time = time()
        IMG_SIZE = (32, 32)  # match with trainer
        # convert to gray
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = frame[int(frame.shape[0]/3):, :]  # /3
        # keep the bottom 2/3 of the image
        frame = cv.resize(frame, (2*IMG_SIZE[0], 2*IMG_SIZE[1]))
        frame = cv.Canny(frame, 100, 200)
        frame = cv.blur(frame, (3, 3), 0)  # worse than blur after 11,11
        frame = cv.resize(frame, IMG_SIZE)

        images = frame

        if faster:
            blob = cv.dnn.blobFromImage(images, 1.0, IMG_SIZE, 0,
                                        swapRB=True, crop=False)
        else:
            frame_flip = cv.flip(frame, 1)
            # stack the 2 images
            images = np.stack((frame, frame_flip), axis=0)
            blob = cv.dnn.blobFromImages(images, 1.0, IMG_SIZE, 0,
                                         swapRB=True, crop=False)
        self.lane_keeper_ahead.setInput(blob)
        out = self.lane_keeper_ahead.forward()  # NOTE: MINUS SIGN IF OLD NET
        print("OUT 0: ", out[0])
        # print("OUT 1: ", out[1])
        output = out[0]
        output_flipped = out[1] if not faster else None

        # <++>
        e3 = output[0]

        if not faster:
            e3_flipped = output_flipped[0]
            e3 = (e3 - e3_flipped) / 2.0  # - 0.17684

        # calculate estimated of thr point ahead to get visual feedback
        d = DISTANCE_POINT_AHEAD_AHEAD
        est_point_ahead = np.array([np.cos(e3)*d, np.sin(e3)*d])

        lane_detection_time = 1000*(time()-start_time)
        self.avg_lane_ahead_detection_time = \
            (self.avg_lane_ahead_detection_time*self.lane_ahead_cnt +
                lane_detection_time) / (self.lane_ahead_cnt+1)
        self.lane_ahead_cnt += 1
        if show_ROI:
            cv.imshow('lane_detection', frame)
            # cv.waitKey(1)
        return e3, est_point_ahead

    def detect_intersection_right(self, frame, show_ROI=True, faster=False):
        """
        Estimates:
        - the angular error around the yaw axis wrt a fixed point ahead (e3),
        - the ditance from the next stop line (1/dist)
        """
        start_time = time()
        IMG_SIZE = (32, 32)  # match with trainer
        # convert to gray
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = frame[int(frame.shape[0]/3):, :]  # /3
        # keep the bottom 2/3 of the image
        frame = cv.resize(frame, (2*IMG_SIZE[0], 2*IMG_SIZE[1]))
        frame = cv.Canny(frame, 100, 200)
        frame = cv.blur(frame, (3, 3), 0)  # worse than blur after 11,11
        frame = cv.resize(frame, IMG_SIZE)

        images = frame

        blob = cv.dnn.blobFromImage(images, 1.0, IMG_SIZE, 0,
                                    swapRB=True, crop=False)
        self.intersection_navigator_right.setInput(blob)
        out = self.intersection_navigator_right.forward()
        output = out[0]
        e3 = output[0]

        # calculate estimated of thr point ahead to get visual feedback
        d = DISTANCE_POINT_AHEAD_AHEAD
        est_point_ahead = np.array([np.cos(e3)*d, np.sin(e3)*d])

        lane_detection_time = 1000*(time()-start_time)
        self.avg_intersection_right_detection_time = \
            (self.avg_intersection_right_detection_time *
             self.intersection_right_cnt +
             lane_detection_time) / (self.intersection_right_cnt+1)
        self.intersection_right_cnt += 1
        if show_ROI:
            cv.imshow('lane_detection', frame)
            # cv.waitKey(1)
        return e3, est_point_ahead

    def detect_intersection_left(self, frame, show_ROI=True, faster=False):
        """
        Estimates:
        - the angular error around the yaw axis wrt a fixed point ahead (e3),
        - the ditance from the next stop line (1/dist)
        """
        start_time = time()
        IMG_SIZE = (32, 32)  # match with trainer
        # convert to gray
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = frame[int(frame.shape[0]/3):, :]  # /3
        # keep the bottom 2/3 of the image
        frame = cv.resize(frame, (2*IMG_SIZE[0], 2*IMG_SIZE[1]))
        frame = cv.Canny(frame, 100, 200)
        frame = cv.blur(frame, (3, 3), 0)  # worse than blur after 11,11
        frame = cv.resize(frame, IMG_SIZE)

        images = frame

        blob = cv.dnn.blobFromImage(images, 1.0, IMG_SIZE, 0,
                                    swapRB=True, crop=False)
        self.intersection_navigator_right.setInput(blob)
        out = self.intersection_navigator_right.forward()
        output = out[0]
        e3 = output[0]

        # calculate estimated of thr point ahead to get visual feedback
        d = DISTANCE_POINT_AHEAD_AHEAD
        est_point_ahead = np.array([np.cos(e3)*d, np.sin(e3)*d])

        lane_detection_time = 1000*(time()-start_time)
        self.avg_intersection_left_detection_time = \
            (self.avg_intersection_left_detection_time *
             self.intersection_left_cnt +
             lane_detection_time) / (self.intersection_left_cnt+1)
        self.intersection_left_cnt += 1
        if show_ROI:
            cv.imshow('lane_detection', frame)
            # cv.waitKey(1)
        return e3, est_point_ahead

    def detect_intersection_forward(self, frame, show_ROI=True):
        """
        Estimates:
        - the angular error around the yaw axis wrt a fixed point ahead (e3),
        - the ditance from the next stop line (1/dist)
        """
        start_time = time()
        IMG_SIZE = (32, 32)  # match with trainer
        # convert to gray
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = frame[int(frame.shape[0]/3):, :]  # /3
        # keep the bottom 2/3 of the image
        frame = cv.resize(frame, (2*IMG_SIZE[0], 2*IMG_SIZE[1]))
        frame = cv.Canny(frame, 100, 200)
        frame = cv.blur(frame, (3, 3), 0)  # worse than blur after 11,11
        frame = cv.resize(frame, IMG_SIZE)

        frame_flip = cv.flip(frame, 1)
        if self._if_in is not None:
            b0 = frame.astype(np.float32)[np.newaxis, np.newaxis, ...]
            b1 = frame_flip.astype(np.float32)[np.newaxis, np.newaxis, ...]
            output         = self.intersection_navigator_forward.run(None, {self._if_in: b0})[0][0]
            output_flipped = self.intersection_navigator_forward.run(None, {self._if_in: b1})[0][0]
        else:
            blob = cv.dnn.blobFromImages(np.stack((frame, frame_flip), axis=0), 1.0, IMG_SIZE, 0, swapRB=True, crop=False)
            self.intersection_navigator_forward.setInput(blob)
            out = self.intersection_navigator_forward.forward()
            output, output_flipped = out[0], out[1]

        e3 = output[0]
        e3_flipped = output_flipped[0]
        e3 = (e3 - e3_flipped) / 2.0  # - 0.17684

        # calculate estimated of thr point ahead to get visual feedback
        d = DISTANCE_POINT_AHEAD_AHEAD
        est_point_ahead = np.array([np.cos(e3)*d, np.sin(e3)*d])

        lane_detection_time = 1000*(time()-start_time)
        self.avg_intersection_forward_detection_time = \
            (self.avg_intersection_forward_detection_time *
             self.intersection_forward_cnt +
             lane_detection_time) / (self.intersection_forward_cnt+1)
        self.intersection_forward_cnt += 1
        if show_ROI:
            cv.imshow('lane_detection', frame)
            # cv.waitKey(1)
        return e3, est_point_ahead

    def detect_roundabout_in(self, frame, show_ROI=True):
        """
        Estimates:
        - the angular error around the yaw axis wrt a fixed point ahead (e3),
        - the ditance from the next stop line (1/dist)
        """
        start_time = time()
        IMG_SIZE = (32, 32)  # match with trainer
        # convert to gray
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = frame[int(frame.shape[0]/3):, :]  # /3
        # keep the bottom 2/3 of the image
        frame = cv.resize(frame, (2*IMG_SIZE[0], 2*IMG_SIZE[1]))
        frame = cv.Canny(frame, 100, 200)
        frame = cv.blur(frame, (3, 3), 0)  # worse than blur after 11,11
        frame = cv.resize(frame, IMG_SIZE)

        images = frame

        frame_flip = cv.flip(frame, 1)
        # stack the 2 images
        images = np.stack((frame, frame_flip), axis=0)
        blob = cv.dnn.blobFromImages(images, 1.0, IMG_SIZE, 0,
                                     swapRB=True, crop=False)
        self.intersection_navigator_forward.setInput(blob)
        out = self.intersection_navigator_forward.forward()
        output = out[0]
        output_flipped = out[1]
        e3 = output[0]

        # calculate estimated of thr point ahead to get visual feedback
        d = DISTANCE_POINT_AHEAD_AHEAD
        est_point_ahead = np.array([np.cos(e3)*d, np.sin(e3)*d])

        lane_detection_time = 1000*(time()-start_time)
        self.avg_roundabout_in_detection_time = \
            (self.avg_roundabout_in_detection_time *
             self.roundabout_in_cnt +
             lane_detection_time) / (self.roundabout_in_cnt+1)
        self.roundabout_in_cnt += 1
        if show_ROI:
            cv.imshow('lane_detection', frame)
            # cv.waitKey(1)
        return e3, est_point_ahead

    def detect_roundabout_about(self, frame, show_ROI=True):
        """
        Estimates:
        - the angular error around the yaw axis wrt a fixed point ahead (e3),
        - the distance from the next stop line (1/dist)
        """
        start_time = time()
        IMG_SIZE = (32, 32)  # match with trainer
        # convert to gray
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = frame[int(frame.shape[0]/3):, :]  # /3
        # keep the bottom 2/3 of the image
        frame = cv.resize(frame, (2*IMG_SIZE[0], 2*IMG_SIZE[1]))
        frame = cv.Canny(frame, 100, 200)
        frame = cv.blur(frame, (3, 3), 0)  # worse than blur after 11,11
        frame = cv.resize(frame, IMG_SIZE)

        images = frame

        blob = cv.dnn.blobFromImage(images, 1.0, IMG_SIZE, 0,
                                    swapRB=True, crop=False)
        self.roundabout_navigator_in.setInput(blob)
        out = self.roundabout_navigator_in.forward()
        output = out[0]
        e3 = output[0]

        # calculate estimated of thr point ahead to get visual feedback
        d = DISTANCE_POINT_AHEAD_AHEAD
        est_point_ahead = np.array([np.cos(e3)*d, np.sin(e3)*d])

        lane_detection_time = 1000*(time()-start_time)
        self.avg_roundabout_about_detection_time = \
            (self.avg_roundabout_about_detection_time *
             self.roundabout_about_cnt +
             lane_detection_time) / (self.roundabout_about_cnt+1)
        self.roundabout_about_cnt += 1
        if show_ROI:
            cv.imshow('lane_detection', frame)
            # cv.waitKey(1)
        return e3, est_point_ahead

    def detect_roundabout_out(self, frame, show_ROI=True):
        """
        Estimates:
        - the angular error around the yaw axis wrt a fixed point ahead (e3),
        - the ditance from the next stop line (1/dist)
        """
        start_time = time()
        IMG_SIZE = (32, 32)  # match with trainer
        # convert to gray
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = frame[int(frame.shape[0]/3):, :]  # /3
        # keep the bottom 2/3 of the image
        frame = cv.resize(frame, (2*IMG_SIZE[0], 2*IMG_SIZE[1]))
        frame = cv.Canny(frame, 100, 200)
        frame = cv.blur(frame, (3, 3), 0)  # worse than blur after 11,11
        frame = cv.resize(frame, IMG_SIZE)

        images = frame

        blob = cv.dnn.blobFromImage(images, 1.0, IMG_SIZE, 0,
                                    swapRB=True, crop=False)
        self.roundabout_navigator_about.setInput(blob)
        out = self.roundabout_navigator_about.forward()
        output = out[0]
        e3 = output[0]

        # calculate estimated of thr point ahead to get visual feedback
        d = DISTANCE_POINT_AHEAD_AHEAD
        est_point_ahead = np.array([np.cos(e3)*d, np.sin(e3)*d])

        lane_detection_time = 1000*(time()-start_time)
        self.avg_roundabout_out_detection_time = \
            (self.avg_roundabout_out_detection_time *
             self.roundabout_out_cnt +
             lane_detection_time) / (self.roundabout_out_cnt+1)
        self.roundabout_out_cnt += 1
        if show_ROI:
            cv.imshow('lane_detection', frame)
            # cv.waitKey(1)
        return e3, est_point_ahead

    def detect_stopline(self, frame, show_ROI=True):
        """
        Estimates the distance to the next stop line
        """
        start_time = time()
        IMG_SIZE = (32, 32)  # match with trainer
        try:
            # convert to gray
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame = frame[int(frame.shape[0]*(2/5)):, :]
            # keep the bottom 2/3 of the image
            frame = cv.blur(frame, (9, 9), 0)
            frame = cv.resize(frame, (2*IMG_SIZE[0], 2*IMG_SIZE[1]))
            frame = cv.Canny(frame, 100, 200)
            frame = cv.blur(frame, (3, 3), 0)  # worse than blur after 11,11
            frame = cv.resize(frame, IMG_SIZE)

            blob = cv.dnn.blobFromImage(frame, 1.0, IMG_SIZE, 0, swapRB=True,
                                        crop=False)
            self.stopline_estimator_adv.setInput(blob)
            output = self.stopline_estimator_adv.forward()
            stopline_x = dist = output[0][0] + PREDICTION_OFFSET
            stopline_y = output[0][1]
            stopline_angle = output[0][2]
            self.est_dist_to_stopline = dist

            stopline_detection_time = 1000*(time()-start_time)
            self.avg_stopline_detection_time = \
                (self.avg_stopline_detection_time*self.lane_cnt +
                 stopline_detection_time) / (self.lane_cnt+1)
            self.lane_cnt += 1
            if show_ROI:
                cv.imshow('stopline_detection', frame)
                cv.imwrite(f'sd/sd_{int(time()*1000)}.png', frame)
                # cv.waitKey(1)
            #print(f"stopline_detection dist: {dist:.2f}, in {stopline_detection_time:.2f} ms")
            return stopline_x, stopline_y, stopline_angle
        except Exception:
            return 69, 420, 666

    # # SIGN DETECTION
    def tile_image(self, image, x, y, w, h, rows, cols, tile_widths,
                   return_size=(32, 32), channels=3):
        assert image.shape[0] >= y + h, f'Image height is {image.shape[0]} \
                but y is {y} and h is {h}'
        assert image.shape[1] >= x + w, f'Image width is {image.shape[1]} but \
                x is {x} and w is {w}'
        assert [ws < w for ws in tile_widths], f'Tile width is {tile_widths} \
                but w is {w}'
        # check x,y,w,h,rows,cols,tile_width are all ints
        assert isinstance(x, int), f'x is {x}'
        assert isinstance(y, int), f'y is {y}'
        assert isinstance(w, int), f'w is {w}'
        assert isinstance(h, int), f'h is {h}'
        assert isinstance(rows, int), f'rows is {rows}'
        assert isinstance(cols, int), f'cols is {cols}'
        num_scales = len(tile_widths)
        idx = 0
        img = image[y:y+h, x:x+w].copy()
        if channels == 3:
            imgs = np.zeros((num_scales*rows*cols, return_size[0],
                             return_size[1], channels), dtype=np.uint8)
        elif channels == 1:
            imgs = np.zeros((num_scales*rows*cols, return_size[0],
                             return_size[1], channels), dtype=np.uint8)
        else:
            raise ValueError(f'return_size must be 2 or 3, \
                    but is {return_size}')
        centers = []
        widths_idxs = []
        for s, i_s in enumerate(range(num_scales)):
            centers_x = np.linspace(int(tile_widths[s]/2),
                                    w-int(tile_widths[s]/2),
                                    cols,
                                    dtype=int)
            centers_y = np.linspace(int(tile_widths[s]/2),
                                    h-int(tile_widths[s]/2),
                                    rows,
                                    dtype=int)
            for i in range(rows):
                for j in range(cols):
                    # img = image[y:y+h, x:x+w]
                    im = img[centers_y[i] - tile_widths[s]//2:
                             centers_y[i] + tile_widths[s]//2,
                             centers_x[j] - tile_widths[s]//2:
                             centers_x[j] + tile_widths[s]//2].copy()

                    im = cv.resize(im, return_size)

                    imgs[idx] = im
                    centers.append([x+centers_x[j], y+centers_y[i]])
                    widths_idxs.append(i_s)
                    idx += 1
        return imgs, centers, widths_idxs

    def detect_sign_old(self, frame, show_ROI=True, show_kp=True):
        """
        Sign classifiier:
        takes the whole frame as input and returns the sign name and
        classification confidence. If the network is not confident enough,
        it returns None sign name and 0.0 confidence
        """
        # ROI
        TL = (240, 10)   # x1, y1
        BR = (320, 70)   # x2, y2
        img = frame.copy()
        img = Detection.automatic_brightness_and_contrast(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = img[TL[1]:BR[1], TL[0]:BR[0]]

        if show_ROI:
            cv.imshow('ROI', img)
            Detection.draw_ROI(frame, TL, BR, show_rect=False, prediction=None,
                               conf=None, show_prediction=False)
            # cv.waitKey(1)

        ratio = 1
        img = cv.resize(img, None, fx=ratio, fy=ratio,
                        interpolation=cv.INTER_AREA)
        kp, des = self.sign_sift.detectAndCompute(img, None)
        if (des is not None):
            if show_kp:
                cop = img.copy()
                cop = \
                    cv.drawKeypoints(cop,
                                     kp,
                                     cop,
                                     flags=cv.
                                     DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv.imshow('keypoints', cop)
                # cv.waitKey(1)
            if len(des) < 10:
                print('No enougth descriptors')
                probs_array = np.zeros(len(signs_dict))
                # the last element of the list of obstacles is no obstacles
                probs_array[-1] = 1
            else:
                im_hist = Detection.ImageHistogram(self.sign_kmean_model,
                                                   des, self.sign_no_clusters)
                im_hist = im_hist.reshape(1, -1)
                im_hist = self.sign_scale_model.transform(im_hist)
                probs_array = self.sign_svm_model.predict_proba(im_hist)
                probs_array = probs_array.reshape(-1)
                # inserting the No sing probability
                probs_array = np.concatenate([probs_array, [0]])
                #print(
                #    f'single prediction: \
                #    {nac.SIGN_NAMES[MAP_DICT_NAMES[np.argmax(probs_array)]]}')
        else:
            print('No descriptors')
            probs_array = np.zeros(len(signs_dict))
            # the last element of the list of obstacles is no obstacles
            probs_array[-1] = 1

        # buffer of classification probabilities
        self.sign_probs_buffer.append(probs_array)

        # mean of the predicion probabilities in the buffer
        buffer_mean = list(self.sign_probs_buffer)
        buffer_mean = np.asarray(buffer_mean)
        buffer_mean = np.mean(buffer_mean, axis=0)
        buffer_mean = buffer_mean.reshape(-1)
        pred_idx = np.argmax(buffer_mean)
        # most likely sign_prediction
        new_prediction = nac.SIGN_NAMES[MAP_DICT_NAMES[pred_idx]]
        new_conf = buffer_mean[pred_idx]
        #print('Prediction proposal:', new_prediction, int(100*new_conf))
        # Comparison of the first two maxima of the buffer_mean
        buffer_mean = np.delete(buffer_mean, pred_idx)
        conf_2 = buffer_mean[int(np.argmax(buffer_mean))]
        # To have a good enought sign_prediction the rest of the probabilities
        # have to be spread
        if (new_conf - conf_2) > 0.30:
            self.sign_prediction = new_prediction
            self.sign_conf = new_conf
        print('New sign_prediction', self.sign_prediction,
              int(100*self.sign_conf))
        if show_ROI:
            Detection.draw_ROI(frame, TL, BR, show_rect=True,
                               prediction=self.sign_prediction,
                               conf=int(100*self.sign_conf),
                               show_prediction=True)
        return self.sign_prediction, self.sign_conf

    def detect_sign(self, frame, show_ROI=True, show_kp=True,
                    depth_frame=None):
        """
        YOLOv8-based sign / object detector.

        Replaces the legacy SIFT + BoW + linear-SVM classifier from
        Brain_DEI with the YOLOv8 pipeline that has been tested on the
        car (`traffic_sign/traffic_sign_node.py`).  Two extensions on
        top of the legacy interface:

          * `depth_frame` (optional) is the OAK-D Pro stereo `depth`
            image (uint16, mm), aligned to the RGB frame.  When
            provided, every YOLO detection gets a `distance_m` field
            populated from the median depth of its bbox centre.
          * The full per-class result is cached in
            `self.last_yolo_detections` so that other parts of the
            brain (`sign_detection_position`, `control_for_signs`)
            can ask "what's the closest <pedestrian/car/stop sign>
            and how far is it" without re-running YOLO.

        Backward-compatible return value
        --------------------------------
        Same as before: `(sign_str, confidence)`.  When more than one
        detection is present, the closest one (smallest `distance_m`)
        wins; if depth is unavailable, the highest-confidence one wins.
        """
        # ── Fast-out if YOLO never loaded ─────────────────────────────────
        if self.sign_yolo is None or self._yolo_input_name is None:
            self.last_yolo_detections = []
            self.sign_prediction = NO_SIGN
            self.sign_conf = 0.0
            return self.sign_prediction, self.sign_conf

        H, W = frame.shape[:2]

        # ── Letterbox preprocess: resize keeping aspect ratio + pad ───────
        scale = YOLO_IMGSZ / max(H, W)
        new_w, new_h = int(W * scale), int(H * scale)
        resized = cv.resize(frame, (new_w, new_h))
        pad_top   = (YOLO_IMGSZ - new_h) // 2
        pad_bot   = YOLO_IMGSZ - new_h - pad_top
        pad_left  = (YOLO_IMGSZ - new_w) // 2
        pad_right = YOLO_IMGSZ - new_w - pad_left
        lb_img = cv.copyMakeBorder(
            resized, pad_top, pad_bot, pad_left, pad_right,
            cv.BORDER_CONSTANT, value=(114, 114, 114))
        self._lb_scale = scale
        self._lb_pad   = (pad_top, pad_left)

        blob = np.transpose(
            lb_img.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis]
        blob = np.ascontiguousarray(blob)

        # ── Inference ─────────────────────────────────────────────────────
        try:
            output = self.sign_yolo.run(
                None, {self._yolo_input_name: blob})
        except Exception as e:
            print(f'[detect_sign] YOLO forward failed: {e}')
            self.last_yolo_detections = []
            self.sign_prediction = NO_SIGN
            self.sign_conf = 0.0
            return self.sign_prediction, self.sign_conf

        # ── Postprocess: shape (1, 4+nc, N) → transpose to (N, 4+nc) ──────
        pred = output[0][0].T
        boxes      = pred[:, :4]
        class_conf = pred[:, 4:]
        scores     = np.max(class_conf, axis=1)
        cls_ids    = np.argmax(class_conf, axis=1)

        mask    = scores >= YOLO_CONF_THRESHOLD
        boxes   = boxes[mask]
        scores  = scores[mask]
        cls_ids = cls_ids[mask]

        detections = []
        if len(boxes) > 0:
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = np.clip(((cx - w / 2) - pad_left) / scale, 0, W - 1)
            y1 = np.clip(((cy - h / 2) - pad_top)  / scale, 0, H - 1)
            x2 = np.clip(((cx + w / 2) - pad_left) / scale, 0, W - 1)
            y2 = np.clip(((cy + h / 2) - pad_top)  / scale, 0, H - 1)

            # Per-class NMS to avoid duplicate boxes for the same class
            for cid in np.unique(cls_ids):
                idx = np.where(cls_ids == cid)[0]
                nms_boxes = np.stack([x1[idx], y1[idx],
                                      x2[idx] - x1[idx],
                                      y2[idx] - y1[idx]], axis=1)
                keep = cv.dnn.NMSBoxes(
                    nms_boxes.tolist(),
                    scores[idx].tolist(),
                    YOLO_CONF_THRESHOLD,
                    YOLO_IOU_THRESHOLD,
                )
                if keep is None or len(keep) == 0:
                    continue
                for k in keep:
                    ki = int(k[0]) if isinstance(
                        k, (list, tuple, np.ndarray)) else int(k)
                    di = idx[ki]
                    cls_name_yolo = YOLO_CLASS_NAMES[int(cls_ids[di])] \
                        if int(cls_ids[di]) < len(YOLO_CLASS_NAMES) \
                        else f'cls_{int(cls_ids[di])}'
                    sign_name = YOLO_TO_LEGACY_SIGN.get(
                        cls_name_yolo, cls_name_yolo)
                    detections.append({
                        'cls':        int(cls_ids[di]),
                        'cls_name':   cls_name_yolo,
                        'sign':       sign_name,
                        'conf':       float(scores[di]),
                        'x1':         int(x1[di]),
                        'y1':         int(y1[di]),
                        'x2':         int(x2[di]),
                        'y2':         int(y2[di]),
                        'distance_m': -1.0,
                    })

        # ── Distance from depth (median of central ROI of each bbox) ──────
        if depth_frame is not None and len(detections) > 0:
            dh, dw = depth_frame.shape[:2]
            for d in detections:
                dx1 = int(np.clip(d['x1'], 0, dw - 1))
                dy1 = int(np.clip(d['y1'], 0, dh - 1))
                dx2 = int(np.clip(d['x2'], 0, dw - 1))
                dy2 = int(np.clip(d['y2'], 0, dh - 1))
                cx_, cy_ = (dx1 + dx2) // 2, (dy1 + dy2) // 2
                rw = max(1, (dx2 - dx1) // 4)
                rh = max(1, (dy2 - dy1) // 4)
                roi = depth_frame[max(0, cy_ - rh): cy_ + rh,
                                  max(0, cx_ - rw): cx_ + rw]
                valid = roi[(roi > DEPTH_MIN_MM) & (roi < DEPTH_MAX_MM)]
                if valid.size > 0:
                    d['distance_m'] = float(np.median(valid)) / 1000.0

        # ── Cache ─────────────────────────────────────────────────────────
        now = _time_mod.time()
        for d in detections:
            d['stamp'] = now
        self.last_yolo_detections = detections
        self._last_yolo_stamp     = now

        # ── Pick the "best" detection for the legacy (sign, conf) return ──
        if not detections:
            self.sign_prediction = NO_SIGN
            self.sign_conf       = 0.0
        else:
            with_dist    = [d for d in detections if d['distance_m'] > 0]
            without_dist = [d for d in detections if d['distance_m'] <= 0]
            ordered = sorted(with_dist, key=lambda dd: dd['distance_m']) + \
                      without_dist
            best = ordered[0] if ordered else None
            if best is None:
                self.sign_prediction = NO_SIGN
                self.sign_conf       = 0.0
            else:
                self.sign_prediction = best['sign']
                self.sign_conf       = best['conf']

        # ── Optional debug visualisation ──────────────────────────────────
        if show_ROI and len(detections) > 0:
            vis = frame.copy()
            for d in detections:
                cv.rectangle(vis, (d['x1'], d['y1']),
                             (d['x2'], d['y2']), (0, 200, 0), 2)
                dist_str = (f"{d['distance_m']:.2f}m"
                            if d['distance_m'] > 0 else 'n/a')
                lbl = f"{d['cls_name']} {dist_str} ({d['conf']:.0%})"
                cv.putText(vis, lbl, (d['x1'], max(0, d['y1'] - 5)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 255, 255), 1)
            try:
                cv.imshow('sign_detection', cv.resize(vis, (640, 360)))
            except Exception:
                pass
        return self.sign_prediction, self.sign_conf

    def get_yolo_object(self, class_name, max_age=0.5):
        """
        Helper that returns the most recent YOLO detection of `class_name`
        (e.g. 'pedestrian', 'car', 'roadblock', 'trafficlight'), or None.

        `max_age` (seconds) caps how stale the cache may be.  This is
        useful because `detect_sign` is not necessarily called every
        single brain tick — it is gated by the FSM and by
        `control_for_signs`.

        The returned dict has the same fields as `last_yolo_detections`
        entries: cls, cls_name, sign, conf, x1, y1, x2, y2, distance_m,
        stamp.

        IMPORTANT: this method only LOOKS UP cached YOLO output, it
        does not run YOLO itself.  `control_for_car` and
        `control_for_pedestrian` (which the user wants kept untouched)
        therefore continue to base their decisions on LIDAR cones and
        HSV pink as before; this is just a convenience for any new
        routine that wants depth-aware obstacle info.
        """
        if not self.last_yolo_detections:
            return None
        if (_time_mod.time() - self._last_yolo_stamp) > max_age:
            return None
        candidates = [d for d in self.last_yolo_detections
                      if d['cls_name'] == class_name
                      or d['sign'] == class_name]
        if not candidates:
            return None
        with_dist    = [d for d in candidates if d['distance_m'] > 0]
        without_dist = [d for d in candidates if d['distance_m'] <= 0]
        if with_dist:
            return min(with_dist, key=lambda dd: dd['distance_m'])
        return max(without_dist, key=lambda dd: dd['conf'])
    """
    def detect_objects_with_yolo(self,frame, show=False):


        #frame = cv.flip(frame, -1)  # Flip if needed
        #frame = cv.resize(frame, (600, 480))
        #frame = cv.rotate(frame, cv.ROTATE_180)
        #TL = (240, 10)   # x1, y1
        #BR = (320, 70)   # x2, y2
        TL = (frame.shape[1] - 110, 5)     # x1, y1 (near top-right)
        BR = (frame.shape[1] - 50, 50)     # x2, y2

        img = frame.copy()
        frame = img[TL[1]:BR[1], TL[0]:BR[0]]

        results = self.sign_yolo.track(frame, persist=True, imgsz=256)

        detections = []

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for track_id, box, class_id in zip(ids, boxes, class_ids):
                x1, y1, x2, y2 = box
                label = self.sign_yolo.names[class_id]
                detections.append({
                    'track_id': track_id,
                    'box': (x1, y1, x2, y2),
                    'label': label
                })
                #print(f"Detectedddddddddddddddddd {label}")
                self.sign_prediction = label

                if show:
                    cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 10),
                                cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        if show:
            cv.imshow("YOLO Detection", frame)
            cv.waitKey(1)

        return self.sign_prediction if self.sign_prediction != None else NO_SIGN
    """
    def classify_frontal_obstacle(self, frames, distances, show_ROI=False,
                                  show_kp=False):
        """
        Obstacle classifier:
        takes the whole frames as input and returns the obstacle name and
        classification confidence.
        """
        assert len(frames) == len(distances)
        assert len(frames) > 0

        obstacles_probs_buffer = collections.deque(maxlen=len(frames))

        for frame, distance in zip(frames, distances):
            img = frame.copy()
            img = Detection.automatic_brightness_and_contrast(img)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            distance = distance*100
            if distance >= 50:
                distance = 50
                TL = distance_dict[str(distance)][0]
                BR = distance_dict[str(distance)][1]
                img = img[TL[1]:BR[1], TL[0]:BR[0]]
            elif distance < 50 and distance >= 40:
                distance = 40
                ratio = 0.9
                TL = distance_dict[str(distance)][0]
                BR = distance_dict[str(distance)][1]
                img = img[TL[1]:BR[1], TL[0]:BR[0]]
                img = cv.resize(img, None, fx=ratio, fy=ratio,
                                interpolation=cv.INTER_AREA)

            elif distance < 40 and distance >= 30:
                distance = 30
                TL = distance_dict[str(distance)][0]
                BR = distance_dict[str(distance)][1]
                img = img[TL[1]:BR[1], TL[0]:BR[0]]
                ratio = 0.7
                img = cv.resize(img, None, fx=ratio, fy=ratio,
                                interpolation=cv.INTER_AREA)
            #  distance < 30:
            else:
                distance = 20
                TL = distance_dict[str(distance)][0]
                BR = distance_dict[str(distance)][1]
                img = img[TL[1]:BR[1], TL[0]:BR[0]]
                ratio = 0.6
                img = cv.resize(img, None, fx=ratio, fy=ratio,
                                interpolation=cv.INTER_AREA)
            if show_ROI:
                Detection.draw_ROI(frame, TL, BR, show_rect=False,
                                   prediction=None, conf=None,
                                   show_prediction=False)
                cv.imshow('ROI', img)
                cv.imwrite(f'sd/sd_{int(time()*1000)}.png', img)
                # cv.waitKey(1)
            img = cv.GaussianBlur(img, (3, 3), 0)
            kp, des = self.sift.detectAndCompute(img, None)  # des= escriptors
            if des is not None and len(des) >= 15:
                if show_kp:
                    cop = img.copy()
                    cop = \
                        cv.drawKeypoints(cop, kp, cop,
                                         flags=cv.
                                         DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                                         )
                    cv.imshow('keypoints', cop)
                    # cv.waitKey(1)
                im_hist = Detection.ImageHistogram(self.kmean_model,
                                                   des, self.no_clusters)
                im_hist = im_hist.reshape(1, -1)
                im_hist = self.scale_model.transform(im_hist)
                probs_array = self.svm_model.predict_proba(im_hist)
                probs_array = probs_array.reshape(-1)
                # inserting the No sing probability
                probs_array = np.concatenate([probs_array, [0]])
                # buffer of classification probabilities
                obstacles_probs_buffer.append(probs_array)
            else:
                print('No descriptors or not enough descriptors')
                # do not add it to the buffer

        if len(obstacles_probs_buffer) > 0:
            # mean of the predicion probabilities in the buffer
            buffer_mean = list(obstacles_probs_buffer)
            buffer_mean = np.asarray(buffer_mean)
            buffer_mean = np.mean(buffer_mean, axis=0)
            buffer_mean = buffer_mean.reshape(-1)
            pred_idx = np.argmax(buffer_mean)
            # most likely prediction
            prediction = obstacles_dict[str(pred_idx)]
            self.prediction = prediction
            conf = buffer_mean[pred_idx]
            self.conf = conf
            print(f'Prediction proposal: {prediction} , {conf}')
            if show_ROI:
                Detection.draw_ROI(frame, TL, BR, show_rect=True,
                                   prediction=self.prediction,
                                   conf=int(100*self.conf),
                                   show_prediction=True)
            return self.prediction, self.conf
        else:
            print('No predictions')
            return None, None

    def classify_frontal_obstacle2(self, frames, distances, show_ROI=False,
                                   show_kp=False):
        """
        Obstacle classifier:
        takes the whole frames as input and returns the obstacle name and
        classification confidence.
        """
        assert len(frames) == len(distances)
        assert len(frames) > 0

        obstacles_probs_buffer = collections.deque(maxlen=len(frames))

        for frame, distance in zip(frames, distances):
            img = frame.copy()
            img = Detection.automatic_brightness_and_contrast(img)
            img = cv.resize(img, (160, 120))
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            distance = distance*100

            kp, des = self.sift.detectAndCompute(img, None)  # des=descriptors
            if des is not None and len(des) >= 15:
                if show_kp:
                    cop = img.copy()
                    cop = \
                        cv.drawKeypoints(cop,
                                         kp,
                                         cop,
                                         flags=cv.
                                         DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                                         )
                    cv.imshow('keypoints', cop)
                    # cv.waitKey(1)
                im_hist = Detection.ImageHistogram(self.kmean_model,
                                                   des,
                                                   self.no_clusters)
                im_hist = im_hist.reshape(1, -1)
                im_hist = self.scale_model.transform(im_hist)
                probs_array = self.svm_model.predict_proba(im_hist)
                probs_array = probs_array.reshape(-1)
                # inserting the No sing probability
                probs_array = np.concatenate([probs_array, [0]])
                # buffer of classification probabilities
                obstacles_probs_buffer.append(probs_array)
            else:
                print('No descriptors or not enough descriptors')
                # do not add it to the buffer

        if len(obstacles_probs_buffer) > 0:
            # mean of the predicion probabilities in the buffer
            buffer_mean = list(obstacles_probs_buffer)
            buffer_mean = np.asarray(buffer_mean)
            buffer_mean = np.mean(buffer_mean, axis=0)
            buffer_mean = buffer_mean.reshape(-1)
            pred_idx = np.argmax(buffer_mean)
            # most likely prediction
            prediction = obstacles_dict[str(pred_idx)]
            self.prediction = prediction
            conf = buffer_mean[pred_idx]
            self.conf = conf
            print(f'Prediction proposal: {prediction} , {conf}')
            return self.prediction, self.conf
        else:
            print('No predictions')
            return None, None

    # helper functions
    def automatic_brightness_and_contrast(image, clip_hist_percent=1):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Calculate grayscale histogram
        hist = cv.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum/100.0)
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        auto_result = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
        return (auto_result)

    # into detection class
    def ImageHistogram(kmeans, descriptor_list, no_clusters):
        """
        Compute the histogram of occurrences of the visual words in the
        input image
        """
        im_hist = np.zeros(no_clusters)

        # feature is the descriptor of a single keypoint
        for feature in descriptor_list:

            feature = feature.reshape(1, -1) 
            idx = kmeans.predict(feature)
            im_hist[idx] += 1
        return im_hist

    # into detection class
    def draw_ROI(frame, TL, BR, show_rect=False, prediction=None, conf=None,
                 show_prediction=False):
        # Blue color in BGR
        if show_rect:
            image = frame.copy()
            # Draw a rectangle with blue line borders of thickness of 2 px
            image = cv.rectangle(image, TL, BR, color=(255, 0, 0), thickness=2)
            cv.imshow("Frame preview", image)
            # cv.waitKey(1)
        if show_rect and show_prediction:
            image = frame.copy()
            # Draw a rectangle with blue line borders of thickness of 2 px
            image = cv.rectangle(image, TL, BR, color=(255, 0, 0), thickness=2)
            cv.putText(img=image, text=prediction + ' ' + str(conf) + '%',
                       org=(TL[0]-100, TL[1]),
                       fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.5,
                       color=(0, 255, 0), thickness=1)
            cv.imshow("Frame preview", image)
            # cv.waitKey(1)

    def detect_yaw_stopline(self, frame, show_ROI=False):
        return detect_angle(original_frame=frame, plot=show_ROI)
