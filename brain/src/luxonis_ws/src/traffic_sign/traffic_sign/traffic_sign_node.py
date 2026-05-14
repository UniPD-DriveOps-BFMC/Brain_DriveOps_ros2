"""
traffic_sign_node.py — BFMC Traffic Sign Detection
No cv_bridge dependency — uses raw NumPy conversion for ROS2 Image messages.
Three modes: SIM (ROS2 topic) | MOCK (webcam/video) | OAK-D Pro (depthai)

"""

import os
import json
import time
import numpy as np
import cv2
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

_PACKAGE_DIR   = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODEL = os.path.join(_PACKAGE_DIR, 'okokok.onnx')

CLASS_NAMES = [
    'stop',              # 0
    'parking',           # 1
    'priority',          # 2
    'roundabout',        # 3
    'one_way',           # 4
    'no_entry',          # 5
    'exit_highway',      # 6
    'entrance_highway',  # 7
    'crosswalk'         # 8
]

# BGR colors (note: OpenCV uses BGR, not RGB).
CLASS_COLORS_BGR = [
    (50,   50, 220),   # stop              — red
    (0,   150, 220),   # parking           — orange
    (0,   180,   0),   # priority          — green
    (180,   0, 180),   # roundabout        — magenta
    (190, 190,   0),   # one_way           — cyan
    (30,   30, 150),   # no_entry          — dark red
    (0,    90, 160),   # exit_highway      — brown/orange
    (220, 120,   0),   # entrance_highway  — blue
    (50,  200,  50)    # crosswalk         — light green
]

# Trigger distance per class (meters). When the closest detection is within
# this distance, the node emits a TRIGGER message on /traffic_sign/detection.
# New dynamic-obstacle classes use shorter distances so the controller reacts
# only when they are actually in the path.
_DEFAULT_TRIGGER_DIST = {
    'stop':             0.60,
    'parking':          0.80,
    'priority':         0.60,
    'roundabout':       0.70,
    'one_way':          0.70,
    'no_entry':         0.80,
    'exit_highway':     0.60,
    'entrance_highway': 0.80,
    'crosswalk':        0.50
}

# Cooldown per class (seconds) — minimum time between two TRIGGER events
# for the same class. Dynamic obstacles get a short cooldown so the
# controller is updated continuously while the obstacle is in view.
_DEFAULT_COOLDOWN = {
    'stop':             5.0,
    'parking':          8.0,
    'priority':         4.0,
    'roundabout':       5.0,
    'one_way':          5.0,
    'no_entry':         5.0,
    'exit_highway':     8.0,
    'entrance_highway': 8.0,
    'crosswalk':        3.0
}

# Supported ROS2 Image encodings → NumPy dtype + OpenCV conversion
_ENCODING_MAP = {
    'rgb8':    (np.uint8,  cv2.COLOR_RGB2BGR),
    'bgr8':    (np.uint8,  None),
    'rgba8':   (np.uint8,  cv2.COLOR_RGBA2BGR),
    'bgra8':   (np.uint8,  cv2.COLOR_BGRA2BGR),
    'mono8':   (np.uint8,  cv2.COLOR_GRAY2BGR),
    'mono16':  (np.uint16, cv2.COLOR_GRAY2BGR),
    '8UC3':    (np.uint8,  None),
    '8UC4':    (np.uint8,  cv2.COLOR_BGRA2BGR),
}


def imgmsg_to_cv2(msg: Image) -> np.ndarray:
    """
    Convert a ROS2 sensor_msgs/Image to a BGR OpenCV frame.
    Pure NumPy — no cv_bridge required.
    """
    enc   = msg.encoding.lower()
    dtype, cvt = _ENCODING_MAP.get(enc, (np.uint8, None))

    # Determine number of channels from step and width
    channels = msg.step // (msg.width * np.dtype(dtype).itemsize)
    channels = max(channels, 1)

    arr = np.frombuffer(msg.data, dtype=dtype).reshape(
        (msg.height, msg.width, channels) if channels > 1
        else (msg.height, msg.width)
    )

    if cvt is not None:
        arr = cv2.cvtColor(arr, cvt)
    elif arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    return arr.copy()   # writable copy


def cv2_to_imgmsg(frame: np.ndarray) -> Image:
    """
    Convert a BGR OpenCV frame to a ROS2 sensor_msgs/Image.
    Pure NumPy — no cv_bridge required.
    """
    msg          = Image()
    msg.height   = frame.shape[0]
    msg.width    = frame.shape[1]
    msg.encoding = 'bgr8'
    msg.step     = frame.shape[1] * frame.shape[2]
    msg.data     = frame.tobytes()
    return msg


class TrafficSignNode(Node):

    def __init__(self):
        super().__init__('traffic_sign_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('model_path',       '')
        self.declare_parameter('detection_topic',  '/traffic_sign/detection')
        self.declare_parameter('image_topic',      '/traffic_sign/image')
        self.declare_parameter('conf_threshold',   0.50)
        self.declare_parameter('iou_threshold',    0.40)
        self.declare_parameter('imgsz',            640)
        self.declare_parameter('depth_min_mm',     100)
        self.declare_parameter('depth_max_mm',     2000)
        self.declare_parameter('debug_view',       True)
        self.declare_parameter('use_sim',          False)
        self.declare_parameter('sim_image_topic',  '/oak/rgb/image_raw')
        self.declare_parameter('use_mock_camera',  False)
        self.declare_parameter('mock_source',      '0')

        for cls in CLASS_NAMES:
            self.declare_parameter(
                f'trigger_dist_{cls}', _DEFAULT_TRIGGER_DIST[cls])
            self.declare_parameter(
                f'cooldown_{cls}', _DEFAULT_COOLDOWN[cls])

        # ── Read ──────────────────────────────────────────────────────────────
        model_path          = self.get_parameter('model_path').value \
                              or _DEFAULT_MODEL
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.iou_threshold  = self.get_parameter('iou_threshold').value
        self.imgsz          = self.get_parameter('imgsz').value
        self.depth_min_mm   = self.get_parameter('depth_min_mm').value
        self.depth_max_mm   = self.get_parameter('depth_max_mm').value
        self.debug_view     = self.get_parameter('debug_view').value
        self.use_sim        = self.get_parameter('use_sim').value
        self.sim_topic      = self.get_parameter('sim_image_topic').value
        self.use_mock       = self.get_parameter('use_mock_camera').value
        self.mock_source    = self.get_parameter('mock_source').value
        detection_topic     = self.get_parameter('detection_topic').value
        image_topic         = self.get_parameter('image_topic').value

        self.trigger_dist = {
            cls: self.get_parameter(f'trigger_dist_{cls}').value
            for cls in CLASS_NAMES
        }
        self.cooldown = {
            cls: self.get_parameter(f'cooldown_{cls}').value
            for cls in CLASS_NAMES
        }

        # ── ONNX Runtime ──────────────────────────────────────────────────────
        if not os.path.isfile(model_path):
            self.get_logger().error(f'ONNX model not found: {model_path}')
            raise FileNotFoundError(model_path)

        self.session    = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.get_logger().info(
            f'ONNX loaded: {os.path.basename(model_path)}'
        )

        # Sanity-check: warn loudly if the model output channels don't match
        # the expected 14 classes. YOLOv8 output shape is [1, 4 + nc, N].
        try:
            out_shape = self.session.get_outputs()[0].shape
            # out_shape may contain strings ("1","84","8400") or ints
            nc_axis = out_shape[1]
            if isinstance(nc_axis, int):
                expected = 4 + len(CLASS_NAMES)
                if nc_axis != expected:
                    self.get_logger().warn(
                        f'Model output has {nc_axis} channels, but '
                        f'CLASS_NAMES has {len(CLASS_NAMES)} entries '
                        f'(expected {expected}). Class IDs may be misaligned!'
                    )
                else:
                    self.get_logger().info(
                        f'Model OK: {len(CLASS_NAMES)} classes detected'
                    )
        except Exception:
            pass

        # ── Publishers (no cv_bridge — use raw imgmsg) ────────────────────────
        self.det_pub = self.create_publisher(String, detection_topic, 10)
        self.img_pub = self.create_publisher(Image,  image_topic,     10)

        self.last_trigger: dict[str, float] = {cls: 0.0 for cls in CLASS_NAMES}
        self._frame  = None
        self._cap    = None
        self._device = None

        # ── Mode selection ────────────────────────────────────────────────────
        if self.use_sim:
            self.create_subscription(
                Image, self.sim_topic, self._sim_image_cb, 10)
            self.get_logger().info(
                f'SIM mode — subscribing to {self.sim_topic}')

        elif self.use_mock:
            src = int(self.mock_source) if self.mock_source.isdigit() \
                  else self.mock_source
            self._cap = cv2.VideoCapture(src)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f'Cannot open mock source: {self.mock_source}')
            self.get_logger().info(f'MOCK mode — source: {self.mock_source}')

        else:
            self._build_oak_pipeline()
            self.get_logger().info('OAK-D Pro mode')

        self.create_timer(1.0 / 20.0, self._cb)

        mode = ('SIM'  if self.use_sim  else
                'MOCK' if self.use_mock else 'OAK')
        self.get_logger().info(
            f'TrafficSignNode ready [{mode}]\n'
            f'  model     → {model_path}\n'
            f'  detection → {detection_topic}\n'
            f'  image out → {image_topic}\n'
            f'  debug     → {self.debug_view}\n'
            f'  classes   → {len(CLASS_NAMES)}'
        )
        for cls in CLASS_NAMES:
            self.get_logger().info(
                f'  {cls:<20} trigger={self.trigger_dist[cls]:.2f}m '
                f'cooldown={self.cooldown[cls]:.1f}s'
            )

    # ── Simulator image callback — pure NumPy, no cv_bridge ───────────────────
    def _sim_image_cb(self, msg: Image) -> None:
        try:
            self._frame = imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().warn(f'imgmsg_to_cv2 error: {e}')

    # ── OAK-D Pro pipeline (DepthAI v3 API) ─────────────────────────────────
    def _build_oak_pipeline(self):
        import depthai as dai

        self._device   = dai.Device()
        self._pipeline = dai.Pipeline(self._device)

        # ── RGB camera ────────────────────────────────────────────────────────
        cam = self._pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A)
        #self._cam_out = cam.requestOutput(
            #(self.imgsz, self.imgsz), dai.ImgFrame.Type.BGR888p)
        
        self._cam_out = cam.requestOutput((1280, 720), dai.ImgFrame.Type.BGR888p)
        # ── Mono cameras ──────────────────────────────────────────────────────
        mono_l = self._pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_B)
        mono_r = self._pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_C)

        # ── Stereo depth ──────────────────────────────────────────────────────
        # self._stereo = self._pipeline.create(dai.node.StereoDepth)

        # # v3 preset enum is PresetMode (not PresetType — that was v2)
        # self._stereo.setDefaultProfilePreset(
        #     dai.node.StereoDepth.PresetMode.FAST_DENSITY)
        # self._stereo.setLeftRightCheck(True)
        # self._stereo.setSubpixel(True)

        # self._stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        # self._stereo.setOutputSize(1280, 720)

        # ── Stereo depth ──────────────────────────────────────────────────────
        self._stereo = self._pipeline.create(dai.node.StereoDepth)

        # self._stereo.initialConfig.postProcessing.temporalFilter.enable = True
        # self._stereo.initialConfig.postProcessing.temporalFilter.persistencyMode = (
        #     dai.StereoDepthConfig.PostProcessing.TemporalFilter
        #     .PersistencyMode.VALID_2_IN_LAST_4
        # )
        # self._stereo.initialConfig.postProcessing.spatialFilter.enable = True
        # self._stereo.initialConfig.postProcessing.spatialFilter.holeFillingRadius = 2
        # self._stereo.initialConfig.postProcessing.spatialFilter.numIterations = 1
        # self._stereo.initialConfig.postProcessing.thresholdFilter.minRange = self.depth_min_mm
        # self._stereo.initialConfig.postProcessing.thresholdFilter.maxRange = self.depth_max_mm

        self._stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
        self._stereo.setLeftRightCheck(True)

        # FIX 1: Disable Subpixel and Enable Extended Disparity for close-range objects
        self._stereo.setSubpixel(False)
        self._stereo.setExtendedDisparity(True)

        self._stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        self._stereo.setOutputSize(1280, 720)

        mono_l.requestOutput(
            (640, 400), dai.ImgFrame.Type.GRAY8).link(self._stereo.left)
        mono_r.requestOutput(
            (640, 400), dai.ImgFrame.Type.GRAY8).link(self._stereo.right)

        self._q_rgb   = self._cam_out.createOutputQueue(
            maxSize=2, blocking=False)
        self._q_depth = self._stereo.depth.createOutputQueue(
            maxSize=2, blocking=False)
        
        self._q_disp_vis = self._stereo.disparity.createOutputQueue(maxSize=2, blocking=False)
        self._max_disp = self._stereo.initialConfig.getMaxDisparity()

        self._pipeline.start()
    
    # ── Preprocess ────────────────────────────────────────────────────────────
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
    #     self._cam_out = cam.requestOutput(
    # (1280, 720), dai.ImgFrame.Type.BGR888p)img  = cv2.resize(frame, (self.imgsz, self.imgsz))
    #     img  = img.astype(np.float32) / 255.0
    #     blob = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
    #     return np.ascontiguousarray(blob)
        """Letterbox frame to imgsz×imgsz, preserving aspect ratio."""
        fh, fw = frame.shape[:2]
        scale  = self.imgsz / max(fh, fw)          # 640 / 1280 = 0.5
        new_w  = int(fw * scale)                    # 640
        new_h  = int(fh * scale)                    # 360
        resized = cv2.resize(frame, (new_w, new_h))

        # Padding to make it square
        pad_top  = (self.imgsz - new_h) // 2       # 140
        pad_bot  = self.imgsz - new_h - pad_top     # 140
        pad_left = (self.imgsz - new_w) // 2       # 0
        pad_right= self.imgsz - new_w - pad_left    # 0

        blob_img = cv2.copyMakeBorder(
            resized, pad_top, pad_bot, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Store for postprocess
        self._lb_scale = scale
        self._lb_pad   = (pad_top, pad_left)

        blob = np.transpose(blob_img.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis]
        return np.ascontiguousarray(blob)

    # ── YOLOv8 postprocess ────────────────────────────────────────────────────
    #def _postprocess(self, output: np.ndarray, orig_h: int, orig_w: int):
        # pred       = output[0].T
        # boxes      = pred[:, :4]
        # class_conf = pred[:, 4:]
        # scores     = np.max(class_conf, axis=1)
        # cls_ids    = np.argmax(class_conf, axis=1)

        # mask    = scores >= self.conf_threshold
        # boxes   = boxes[mask]
        # scores  = scores[mask]
        # cls_ids = cls_ids[mask]

        # if len(boxes) == 0:
        #     return []

        # # YOLOv8 ONNX outputs pixel coords in imgsz space — just clip, no scaling
        # cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        # # Scale from imgsz space → original frame space
        # sx = orig_w / self.imgsz
        # sy = orig_h / self.imgsz

        # x1 = np.clip((cx - w / 2) * sx, 0, orig_w - 1)
        # y1 = np.clip((cy - h / 2) * sy, 0, orig_h - 1)
        # x2 = np.clip((cx + w / 2) * sx, 0, orig_w - 1)
        # y2 = np.clip((cy + h / 2) * sy, 0, orig_h - 1)

        # detections = []
        # for cls_id in np.unique(cls_ids):
        #     idx = np.where(cls_ids == cls_id)[0]

        #     # FIX: NMSBoxes wants [x, y, w, h] — not [x1, y1, x2, y2]
        #     nms_boxes = np.stack([
        #         x1[idx],
        #         y1[idx],
        #         x2[idx] - x1[idx],   # width
        #         y2[idx] - y1[idx],   # height
        #     ], axis=1)

        #     s    = scores[idx]
        #     keep = cv2.dnn.NMSBoxes(
        #         nms_boxes.tolist(), s.tolist(),
        #         self.conf_threshold, self.iou_threshold
        #     )

        #     for k in keep:
        #         ki = k[0] if isinstance(k, (list, tuple, np.ndarray)) else int(k)
        #         detections.append({
        #             'cls':  int(cls_ids[idx[ki]]),
        #             'conf': float(scores[idx[ki]]),
        #             'x1':   int(x1[idx[ki]]),
        #             'y1':   int(y1[idx[ki]]),
        #             'x2':   int(x2[idx[ki]]),
        #             'y2':   int(y2[idx[ki]]),
        #         })

        # return detections

    def _postprocess(self, output: np.ndarray, orig_h: int, orig_w: int):
        pred       = output[0].T
        class_conf = pred[:, 4:]
        scores     = np.max(class_conf, axis=1)
        cls_ids    = np.argmax(class_conf, axis=1)
        boxes      = pred[:, :4]

        mask    = scores >= self.conf_threshold
        boxes   = boxes[mask]
        scores  = scores[mask]
        cls_ids = cls_ids[mask]

        if len(boxes) == 0:
            return []

        cx, cy, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]

        # YOLO coords are in letterboxed imgsz space — undo letterbox
        pad_top, pad_left = self._lb_pad
        scale             = self._lb_scale

        x1 = np.clip(((cx - w/2) - pad_left) / scale, 0, orig_w - 1)
        y1 = np.clip(((cy - h/2) - pad_top)  / scale, 0, orig_h - 1)
        x2 = np.clip(((cx + w/2) - pad_left) / scale, 0, orig_w - 1)
        y2 = np.clip(((cy + h/2) - pad_top)  / scale, 0, orig_h - 1)

        detections = []
        for cls_id in np.unique(cls_ids):
            idx = np.where(cls_ids == cls_id)[0]
            nms_boxes = np.stack([
                x1[idx], y1[idx],
                x2[idx] - x1[idx],   # w
                y2[idx] - y1[idx],   # h
            ], axis=1)
            keep = cv2.dnn.NMSBoxes(
                nms_boxes.tolist(), scores[idx].tolist(),
                self.conf_threshold, self.iou_threshold)
            for k in keep:
                ki = k[0] if isinstance(k, (list, tuple, np.ndarray)) else int(k)
                detections.append({
                    'cls':  int(cls_ids[idx[ki]]),
                    'conf': float(scores[idx[ki]]),
                    'x1':   int(x1[idx[ki]]),
                    'y1':   int(y1[idx[ki]]),
                    'x2':   int(x2[idx[ki]]),
                    'y2':   int(y2[idx[ki]]),
                })
        return detections   

    # ── Distance (OAK mode only) ───────────────────────────────────────────────
    # def _get_distance_m(self, depth_frame: np.ndarray,
    #                 x1: int, y1: int, x2: int, y2: int) -> float:
    #     """
    #     Sample the central ROI of a bbox from the depth frame.
    #     depth_frame: uint16 numpy array, values in mm, aligned to RGB.
    #     Returns distance in metres, or -1.0 if unreliable.
    #     """
        # if depth_frame is None:
        #     return -1.0

        # dh, dw = depth_frame.shape[:2]

        # # Scale bbox from RGB-frame space → depth-frame space.
        # # After setDepthAlign these should match, but scaling is
        # # safe even if resolutions differ slightly.
        # sx = dw / self.imgsz
        # sy = dh / self.imgsz
        # dx1 = int(np.clip(x1 * sx, 0, dw - 1))
        # dy1 = int(np.clip(y1 * sy, 0, dh - 1))
        # dx2 = int(np.clip(x2 * sx, 0, dw - 1))
        # dy2 = int(np.clip(y2 * sy, 0, dh - 1))

        # # Central 50% of the bbox — avoids sign edges bleeding
        # # into the (farther) background
        # cx, cy = (dx1 + dx2) // 2, (dy1 + dy2) // 2
        # rw = max(1, (dx2 - dx1) // 4)
        # rh = max(1, (dy2 - dy1) // 4)
        # roi = depth_frame[max(0, cy - rh): cy + rh,
        #                 max(0, cx - rw): cx + rw]

        # valid = roi[(roi > self.depth_min_mm) & (roi < self.depth_max_mm)]
        # if valid.size == 0:
        #     return -1.0

        # return float(np.median(valid)) / 1000.0   # mm → metres

    # def _get_distance_m(self, depth_frame, x1, y1, x2, y2):
    #     if depth_frame is None:
    #         return -1.0

    #     # No scaling needed — depth is already resized to RGB frame space
    #     cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    #     rw = max(1, (x2 - x1) // 6)
    #     rh = max(1, (y2 - y1) // 6)

    #     dh, dw = depth_frame.shape[:2]
    #     roi = depth_frame[
    #         max(0, cy - rh): min(dh, cy + rh),
    #         max(0, cx - rw): min(dw, cx + rw)
    #     ]

    #     valid = roi[(roi > self.depth_min_mm) & (roi < self.depth_max_mm)]
    #     if valid.size == 0:
    #         return -1.0

    #     return float(np.percentile(valid, 15)) / 1000.0

    def _get_distance_m(self, depth_frame, x1, y1, x2, y2) -> float:
        if depth_frame is None:
            return -1.0

        dh, dw = depth_frame.shape[:2]   # 720, 1280
        # bbox coords are already in 1280×720 space — just clip, no scaling
        dx1 = int(np.clip(x1, 0, dw - 1))
        dy1 = int(np.clip(y1, 0, dh - 1))
        dx2 = int(np.clip(x2, 0, dw - 1))
        dy2 = int(np.clip(y2, 0, dh - 1))

        cx, cy = (dx1 + dx2) // 2, (dy1 + dy2) // 2
        rw = max(1, (dx2 - dx1) // 4)
        rh = max(1, (dy2 - dy1) // 4)

        roi = depth_frame[max(0, cy-rh): cy+rh,
                        max(0, cx-rw): cx+rw]

        valid = roi[(roi > self.depth_min_mm) & (roi < self.depth_max_mm)]
        if valid.size == 0:
            return -1.0
        return float(np.median(valid)) / 1000.0

    # ── Publish detection ─────────────────────────────────────────────────────
    def _publish_detection(self, class_name, dist_m, conf):
        payload  = json.dumps({
            'sign':       class_name,
            'distance_m': round(dist_m, 3) if dist_m > 0 else None,
            'confidence': round(conf, 3),
        })
        msg      = String()
        msg.data = payload
        self.det_pub.publish(msg)
        self.get_logger().info(f'TRIGGER → {payload}')

    # ── Helper: safe class-name lookup ────────────────────────────────────────
    @staticmethod
    def _cls_name(cls_id: int) -> str:
        return CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) \
               else f'cls_{cls_id}'

    # ── 20 Hz callback ────────────────────────────────────────────────────────
    # def _cb(self) -> None:
    #     disp = None

    #     if self.use_sim:
    #         frame = self._frame
    #     elif self.use_mock:
    #         ret, frame = self._cap.read()
    #         if not ret:
    #             self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #             ret, frame = self._cap.read()
    #         frame = frame if ret else None
    #     else:
    #         try:
    #             in_rgb  = self._q_rgb.tryGet()
    #             #in_depth    = self._q_depth.tryGet()
    #             frame   = in_rgb.getCvFrame()  if in_rgb  else None
    #             #depth_frame = in_depth.getFrame() if in_depth else None
    #             # In _cb, right after getting depth_frame from the queue:
    #             in_depth = self._q_depth.tryGet()
    #             if in_depth is not None:
    #                 depth_frame = in_depth.getFrame()
    #                 # Resize depth to match RGB frame so coordinates align 1:1
    #                 depth_frame = cv2.resize(
    #                     depth_frame,
    #                     (frame.shape[1], frame.shape[0]),  # (W, H) of RGB frame
    #                     interpolation=cv2.INTER_NEAREST    # NEAREST — never interpolate depth values
    #                 )
    #         except Exception as e:
    #             self.get_logger().warn(f'Queue error: {e}')
    #             return # Esce dalla callback e aspetta il prossimo ciclo

    #     if frame is None:
    #         return

    #     H, W = frame.shape[:2]

    #     output = self.session.run(
    #         None, {self.input_name: self._preprocess(frame)})
    #     dets   = self._postprocess(output[0], H, W)

    #     for det in dets:
    #         det['dist_m'] = self._get_distance_m(
    #             depth_frame, det['x1'], det['y1'], det['x2'], det['y2'])

    #     # Sort by distance if available, otherwise keep all detections
    #     dets_with_dist = [d for d in dets if d['dist_m'] > 0]
    #     if dets_with_dist:
    #         dets = sorted(dets_with_dist, key=lambda d: d['dist_m'])
    #     # else: keep all dets without depth filter (close range or flat objects)

    #     # Draw
    #     annotated = frame.copy()
    #     mode_str  = 'SIM' if self.use_sim else ('MOCK' if self.use_mock else 'OAK')
    #     cv2.putText(annotated, f'[{mode_str}] {len(dets)} sign(s)',
    #                 (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    #     for i, det in enumerate(dets):
    #         cls_id     = det['cls']
    #         class_name = self._cls_name(cls_id)
    #         color      = CLASS_COLORS_BGR[cls_id % len(CLASS_COLORS_BGR)]
    #         x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']

    #         cv2.rectangle(annotated, (x1, y1), (x2, y2), color,
    #                       3 if i == 0 else 1)
    #         if i == 0:
    #             cv2.rectangle(annotated,
    #                           (x1-3, y1-3), (x2+3, y2+3), (255,255,255), 1)

    #         dist_str = f"{det['dist_m']:.2f}m" if det['dist_m'] > 0 else 'n/a'
    #         label    = f"{class_name} {dist_str} ({det['conf']:.0%})"
    #         (tw, th), _ = cv2.getTextSize(
    #             label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    #         cv2.rectangle(annotated,
    #                       (x1, y1-th-10), (x1+tw+4, y1), color, -1)
    #         cv2.putText(annotated, label, (x1+2, y1-5),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    #     # Trigger
    #     now = time.time()
    #     if dets:
    #         det        = dets[0]
    #         cls_id     = det['cls']
    #         class_name = self._cls_name(cls_id)
    #         dist_m     = det['dist_m']
    #         conf       = det['conf']
    #         trig_dist  = self.trigger_dist.get(class_name, 0.60)
    #         trig_cool  = self.cooldown.get(class_name, 5.0)

    #         # In OAK mode: use distance if available, else trigger on confidence only
    #         distance_ok = (dist_m <= trig_dist) if dist_m > 0 else True
    #         cooldown_ok = (now - self.last_trigger.get(class_name, 0.0)) \
    #                       >= trig_cool

    #         if distance_ok and cooldown_ok:
    #             self._publish_detection(class_name, dist_m, conf)
    #             self.last_trigger[class_name] = now
    #             cv2.putText(annotated, f'*** TRIGGER: {class_name} ***',
    #                         (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
    #                         0.8, (0,0,255), 2)

    #         label_hdr = (f"CLOSEST: {class_name}  {dist_m:.2f}m"
    #                      if dist_m > 0 else f"DETECTED: {class_name}")
    #         cv2.putText(annotated, label_hdr, (10, 55),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    #     # Publish annotated image — pure NumPy, no cv_bridge
    #     try:
    #         self.img_pub.publish(cv2_to_imgmsg(annotated))
    #     except Exception as e:
    #         self.get_logger().warn(f'img publish: {e}')

    #     # Debug window
    #     if self.debug_view:
    #         debug_frame = annotated.copy()

    #         # # ── Depth heatmap ─────────────────────────────────────────────────────
    #         # in_disp_vis = self._q_disp_vis.tryGet() if not self.use_sim \
    #         #                                         and not self.use_mock else None
    #         # if in_disp_vis is not None:
    #         #     disp_raw = in_disp_vis.getFrame()   # uint16 or uint8 depending on subpixel

    #         #     # Normalize to 0-255 for colormap
    #         #     disp_norm = (disp_raw.astype(np.float32) / self._max_disp * 255).clip(0, 255)
    #         #     disp_color = cv2.applyColorMap(
    #         #         disp_norm.astype(np.uint8), cv2.COLORMAP_MAGMA)

    #         #     # Resize to match RGB frame
    #         #     disp_color = cv2.resize(disp_color, (self.imgsz, self.imgsz))

    #         #     # Draw bboxes on heatmap too so you can verify alignment
    #         #     for det in dets:
    #         #         cls_id = det['cls']
    #         #         color  = CLASS_COLORS_BGR[cls_id % len(CLASS_COLORS_BGR)]
    #         #         cv2.rectangle(disp_color,
    #         #                     (det['x1'], det['y1']),
    #         #                     (det['x2'], det['y2']), color, 2)
    #         #         # Draw the ROI that is actually sampled for distance
    #         #         cx = (det['x1'] + det['x2']) // 2
    #         #         cy = (det['y1'] + det['y2']) // 2
    #         #         rw = max(1, (det['x2'] - det['x1']) // 4)
    #         #         rh = max(1, (det['y2'] - det['y1']) // 4)
    #         #         cv2.rectangle(disp_color,
    #         #                     (cx - rw, cy - rh), (cx + rw, cy + rh),
    #         #                     (255, 255, 255), 1)   # white = sampled ROI
    #         #         dist_str = f"{det['dist_m']:.2f}m" if det['dist_m'] > 0 else 'n/a'
    #         #         cv2.putText(disp_color, dist_str, (det['x1'], det['y1'] - 5),
    #         #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    #         #     # Add legend
    #         #     cv2.putText(disp_color, 'DEPTH (warm=near, cold=far)',
    #         #                 (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    #         #     debug_frame = np.hstack([annotated, disp_color])

    #         # # ── Depth heatmap ─────────────────────────────────────────────────────
    #         # # FIX 2: Visualize the aligned depth_frame, not the unaligned disparity queue
    #         # if depth_frame is not None and not self.use_sim and not self.use_mock:
                
    #         #     # depth_frame is in mm. Clip to a sensible max distance for visualization (e.g., 5000mm)
    #         #     max_vis_dist_mm = 5000
    #         #     disp_norm = (depth_frame.astype(np.float32) / max_vis_dist_mm * 255).clip(0, 255)
                
    #         #     # Invert so close objects are warm/bright, and far objects are cold/dark
    #         #     disp_norm = 255 - disp_norm
    #         #     disp_color = cv2.applyColorMap(disp_norm.astype(np.uint8), cv2.COLORMAP_MAGMA)

    #         #     # Resize to match the RGB frame dynamically
    #         #     disp_color = cv2.resize(disp_color, (annotated.shape[1], annotated.shape[0]))

    #         #     # Draw bboxes on the aligned heatmap
    #         #     for det in dets:
    #         #         cls_id = det['cls']
    #         #         color  = CLASS_COLORS_BGR[cls_id % len(CLASS_COLORS_BGR)]
    #         #         cv2.rectangle(disp_color, (det['x1'], det['y1']), (det['x2'], det['y2']), color, 2)
                    
    #         #         dist_str = f"{det['dist_m']:.2f}m" if det['dist_m'] > 0 else 'n/a'
    #         #         cv2.putText(disp_color, dist_str, (det['x1'], det['y1'] - 5),
    #         #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    #         #     debug_frame = np.hstack([annotated, disp_color])

    #         # ── Depth heatmap ─────────────────────────────────────────────────────
    #         in_disp_vis = self._q_disp_vis.tryGet() if not self.use_sim and not self.use_mock else None
            
    #         if in_disp_vis is not None:
    #             disp_raw = in_disp_vis.getFrame()
                
    #             # Normalizza la disparità (0-190 per Extended Disparity) a 0-255
    #             disp_norm = (disp_raw.astype(np.float32) / 190.0 * 255).clip(0, 255)
    #             disp_color = cv2.applyColorMap(disp_norm.astype(np.uint8), cv2.COLORMAP_MAGMA)

    #             # Ridimensiona per combaciare dinamicamente con il frame RGB
    #             disp_color = cv2.resize(disp_color, (annotated.shape[1], annotated.shape[0]))

    #             # Disegna i bounding box sulla heatmap
    #             for det in dets:
    #                 cls_id = det['cls']
    #                 color  = CLASS_COLORS_BGR[cls_id % len(CLASS_COLORS_BGR)]
    #                 cv2.rectangle(disp_color,
    #                             (det['x1'], det['y1']),
    #                             (det['x2'], det['y2']), color, 2)
                    
    #                 # Disegna la ROI centrale usata per calcolare la distanza
    #                 cx = (det['x1'] + det['x2']) // 2
    #                 cy = (det['y1'] + det['y2']) // 2
    #                 rw = max(1, (det['x2'] - det['x1']) // 4)
    #                 rh = max(1, (det['y2'] - det['y1']) // 4)
    #                 cv2.rectangle(disp_color,
    #                             (cx - rw, cy - rh), (cx + rw, cy + rh),
    #                             (255, 255, 255), 1)   
                    
    #                 dist_str = f"{det['dist_m']:.2f}m" if det['dist_m'] > 0 else 'n/a'
    #                 cv2.putText(disp_color, dist_str, (det['x1'], det['y1'] - 5),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    #             # Aggiungi la legenda
    #             cv2.putText(disp_color, 'DEPTH (warm=near, cold=far)',
    #                         (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    #             debug_frame = np.hstack([annotated, disp_color])

    #         cv2.imshow('Traffic Sign Detection', debug_frame)

    #         # ── Console stats for every detection ─────────────────────────────────
    #         if dets:
    #             self.get_logger().info('─' * 55)
    #             for i, det in enumerate(dets):
    #                 tag = ' ← NEAREST' if i == 0 else ''
    #                 self.get_logger().info(
    #                     f"  [{i}] {self._cls_name(det['cls']):<20} "
    #                     f"dist={det['dist_m']:>6.3f}m  "
    #                     f"conf={det['conf']:.0%}  "
    #                     f"bbox=({det['x1']},{det['y1']})-({det['x2']},{det['y2']})"
    #                     f"{tag}"
    #                 )

    #         cv2.waitKey(1)

    def _cb(self) -> None:
        depth_frame = None

        # ── Frame acquisition ─────────────────────────────────────────────────
        if self.use_sim:
            frame = self._frame

        elif self.use_mock:
            ret, frame = self._cap.read()
            if not ret:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
            frame = frame if ret else None

        else:
            try:
                in_rgb   = self._q_rgb.tryGet()
                in_depth = self._q_depth.tryGet()
                frame       = in_rgb.getCvFrame()   if in_rgb   else None
                depth_frame = in_depth.getFrame()   if in_depth else None
            except Exception as e:
                self.get_logger().warn(f'Queue error: {e}')
                return

        if frame is None:
            return

        H, W = frame.shape[:2]   # 720, 1280 in OAK mode

        # ── Inference ─────────────────────────────────────────────────────────
        output = self.session.run(
            None, {self.input_name: self._preprocess(frame)})
        dets = self._postprocess(output[0], H, W)

        # ── Attach distance to every detection ────────────────────────────────
        for det in dets:
            det['dist_m'] = self._get_distance_m(
                depth_frame, det['x1'], det['y1'], det['x2'], det['y2'])

        # ── Sort by distance (valid first, then unknown) ───────────────────────
        dets_with_dist    = [d for d in dets if d['dist_m'] > 0]
        dets_without_dist = [d for d in dets if d['dist_m'] <= 0]
        dets = sorted(dets_with_dist, key=lambda d: d['dist_m']) + dets_without_dist

        # ── Annotate RGB frame ────────────────────────────────────────────────
        annotated = frame.copy()
        mode_str  = 'SIM' if self.use_sim else ('MOCK' if self.use_mock else 'OAK')
        cv2.putText(annotated, f'[{mode_str}] {len(dets)} sign(s)',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        

        # Find nearest candidate here so we can highlight it while drawing
        nearest = next((d for d in dets if d['dist_m'] > 0), None)
        if nearest is None and dets:
            nearest = max(dets, key=lambda d: d['conf'])

        for i, det in enumerate(dets):
            cls_id     = det['cls']
            class_name = self._cls_name(cls_id)
            color      = CLASS_COLORS_BGR[cls_id % len(CLASS_COLORS_BGR)]
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']

            is_nearest = (det is nearest)

            # Thicker box + white outline for the nearest sign
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color,
                        3 if i == 0 else 1)
            if is_nearest:
                cv2.rectangle(annotated,
                            (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                            (255, 255, 255), 1)

            dist_str = f"{det['dist_m']:.2f}m" if det['dist_m'] > 0 else 'n/a'
            label    = f"{class_name} {dist_str} ({det['conf']:.0%})"
            if is_nearest:
                label += ' [NEAREST]'

            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated,
                        (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # ── Trigger logic ─────────────────────────────────────────────────────
        now = time.time()
        if nearest is not None:
            cls_id     = nearest['cls']
            class_name = self._cls_name(cls_id)
            dist_m     = nearest['dist_m']
            conf       = nearest['conf']
            trig_dist  = self.trigger_dist.get(class_name, 0.60)
            trig_cool  = self.cooldown.get(class_name, 5.0)

            distance_ok = (0 < dist_m <= trig_dist)
            cooldown_ok = (now - self.last_trigger.get(class_name, 0.0)) >= trig_cool

            if distance_ok and cooldown_ok:
                self._publish_detection(class_name, dist_m, conf)
                self.last_trigger[class_name] = now
                cv2.putText(annotated, f'*** TRIGGER: {class_name} ***',
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

            label_hdr = (f"NEAREST: {class_name}  {dist_m:.2f}m"
                        if dist_m > 0 else f"DETECTED: {class_name} (no depth)")
            cv2.putText(annotated, label_hdr, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # ── Publish annotated image ────────────────────────────────────────────
        try:
            # Resize to imgsz square for the ROS topic
            pub_frame = cv2.resize(annotated, (self.imgsz, self.imgsz))
            self.img_pub.publish(cv2_to_imgmsg(pub_frame))
        except Exception as e:
            self.get_logger().warn(f'img publish: {e}')

        # ── Debug window ──────────────────────────────────────────────────────
        if self.debug_view:
            # Resize RGB to 640×360 for display (keeps 16:9)
            rgb_vis = cv2.resize(annotated, (640, 360))

            # Depth heatmap panel
            in_disp_vis = None
            if not self.use_sim and not self.use_mock:
                in_disp_vis = self._q_disp_vis.tryGet()

            if in_disp_vis is not None:
                disp_raw = in_disp_vis.getFrame()

                # Normalise disparity → 0-255 colourmap
                disp_norm  = (disp_raw.astype(np.float32) / self._max_disp * 255
                            ).clip(0, 255)
                disp_color = cv2.applyColorMap(
                    disp_norm.astype(np.uint8), cv2.COLORMAP_MAGMA)

                # Resize to 640×360 to match rgb_vis
                disp_color = cv2.resize(disp_color, (640, 360))

                # Draw bboxes on depth panel — coords are in 1280×720,
                # scale down to 640×360 for display
                sx_vis = 640 / W
                sy_vis = 360 / H
                for det in dets:
                    cls_id = det['cls']
                    color  = CLASS_COLORS_BGR[cls_id % len(CLASS_COLORS_BGR)]
                    is_nearest = (det is nearest)
                    vx1 = int(det['x1'] * sx_vis)
                    vy1 = int(det['y1'] * sy_vis)
                    vx2 = int(det['x2'] * sx_vis)
                    vy2 = int(det['y2'] * sy_vis)
                    
                    cv2.rectangle(disp_color, (vx1, vy1), (vx2, vy2), color, 2)

                    # White inner rectangle = sampled ROI
                    cx = (vx1 + vx2) // 2
                    cy = (vy1 + vy2) // 2
                    rw = max(1, (vx2 - vx1) // 4)
                    rh = max(1, (vy2 - vy1) // 4)
                    cv2.rectangle(disp_color,
                                (cx - rw, cy - rh), (cx + rw, cy + rh),
                                (255, 255, 255), 1)

                    dist_str = f"{det['dist_m']:.2f}m" if det['dist_m'] > 0 else 'n/a'
                    cv2.putText(disp_color, dist_str, (vx1, vy1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                cv2.putText(disp_color, 'DEPTH (warm=near, cold=far)',
                            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                debug_frame = np.hstack([rgb_vis, disp_color])
            else:
                # No depth frame yet (SIM / MOCK mode or stereo not ready)
                debug_frame = rgb_vis

            cv2.imshow('Traffic Sign Detection', debug_frame)

            # ── Console stats ─────────────────────────────────────────────────
            # ── Console stats — all detections ranked ─────────────────────────
            if dets:
                self.get_logger().info('─' * 55)
                for i, det in enumerate(dets):
                    tag = ' ← TRIGGER CANDIDATE' if det is nearest else ''
                    self.get_logger().info(
                        f"  [{i}] {self._cls_name(det['cls']):<20} "
                        f"dist={det['dist_m']:>6.3f}m  "
                        f"conf={det['conf']:.0%}  "
                        f"bbox=({det['x1']},{det['y1']})-"
                        f"({det['x2']},{det['y2']})"
                        f"{tag}"
                    )

            cv2.waitKey(1)


    # ── Cleanup ───────────────────────────────────────────────────────────────
    def destroy_node(self):
        if self._cap:
            self._cap.release()
        if self._device:
            self._device.close()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrafficSignNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()