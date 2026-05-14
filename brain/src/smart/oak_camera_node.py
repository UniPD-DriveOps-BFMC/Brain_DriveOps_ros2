"""
oak_camera_node.py — OAK-D Pro Camera + Traffic Sign Detection

Standalone ROS2 node. In OAK-D mode it opens the camera pipeline directly
and publishes all streams; main_brain.py subscribes to the sign topics.

Published topics (OAK-D mode):
  /oak/rgb/image_raw        (sensor_msgs/Image,      bgr8,   1280×720)
  /oak/left/image_raw       (sensor_msgs/Image,      mono8,  640×400)
  /oak/right/image_raw      (sensor_msgs/Image,      mono8,  640×400)
  /oak/stereo/image_raw     (sensor_msgs/Image,      mono16, 1280×720, mm)
  /oak/rgb/camera_info      (sensor_msgs/CameraInfo)
  /oak/left/camera_info     (sensor_msgs/CameraInfo)
  /oak/right/camera_info    (sensor_msgs/CameraInfo)
  /traffic_sign/detection   (std_msgs/String,         JSON)
  /traffic_sign/distance    (std_msgs/Float32,        metres, nearest sign)
  /traffic_sign/image       (sensor_msgs/Image,       annotated RGB)

In SIM mode (use_sim:=true) the node subscribes to /oak/rgb/image_raw
from Gazebo and only publishes the three traffic_sign topics.

No cv_bridge required — pure NumPy image serialisation throughout.

Run standalone (OAK-D):
  ros2 run smart oak_camera_node

Run in simulation:
  ros2 run smart oak_camera_node --ros-args -p use_sim:=true
"""

import os
import json
import threading
import numpy as np
import cv2
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Float32
from ament_index_python.packages import get_package_share_directory

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_MODEL = os.path.join(
    get_package_share_directory('smart'), 'models', 'sign_yolo.onnx')

CLASS_NAMES = [
    'stop', 'parking', 'priority', 'roundabout', 'one_way',
    'no_entry', 'exit_highway', 'entrance_highway', 'crosswalk',
]

CLASS_COLORS_BGR = [
    (50,  50, 220), (0,  150, 220), (0,  180,   0), (180,  0, 180),
    (190, 190,  0), (30,  30, 150), (0,   90, 160), (220, 120,   0),
    (50, 200,  50),
]


# Supported encodings for SIM-mode ROS Image → OpenCV
_ENCODING_MAP = {
    'rgb8':   (np.uint8,  cv2.COLOR_RGB2BGR),
    'bgr8':   (np.uint8,  None),
    'rgba8':  (np.uint8,  cv2.COLOR_RGBA2BGR),
    'bgra8':  (np.uint8,  cv2.COLOR_BGRA2BGR),
    'mono8':  (np.uint8,  cv2.COLOR_GRAY2BGR),
    'mono16': (np.uint16, cv2.COLOR_GRAY2BGR),
    '8uc3':   (np.uint8,  None),
    '8uc4':   (np.uint8,  cv2.COLOR_BGRA2BGR),
}


# ── Pure-NumPy Image serialisers (no cv_bridge) ───────────────────────────────

def _imgmsg_to_cv2(msg: Image) -> np.ndarray:
    enc  = msg.encoding.lower()
    dtype, cvt = _ENCODING_MAP.get(enc, (np.uint8, None))
    ch   = msg.step // (msg.width * np.dtype(dtype).itemsize)
    ch   = max(ch, 1)
    arr  = np.frombuffer(msg.data, dtype=dtype).reshape(
        (msg.height, msg.width, ch) if ch > 1 else (msg.height, msg.width))
    if cvt is not None:
        arr = cv2.cvtColor(arr, cvt)
    elif arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return arr.copy()


def _make_imgmsg(frame: np.ndarray, encoding: str, stamp=None,
                 frame_id: str = '') -> Image:
    msg = Image()
    msg.height   = frame.shape[0]
    msg.width    = frame.shape[1]
    msg.encoding = encoding
    if encoding in ('bgr8', 'rgb8'):
        msg.step = frame.shape[1] * 3
        if encoding == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif encoding == 'mono8':
        msg.step = frame.shape[1]
    elif encoding == 'mono16':
        msg.step = frame.shape[1] * 2
        frame = frame.astype(np.uint16)
    msg.data = frame.tobytes()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    return msg


# ── Node ──────────────────────────────────────────────────────────────────────

class OakCameraNode(Node):

    def __init__(self):
        super().__init__('oak_camera_node')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('use_sim',         False)
        self.declare_parameter('sim_image_topic', '/oak/rgb/image_raw')
        self.declare_parameter('model_path',      _DEFAULT_MODEL)
        self.declare_parameter('conf_threshold',  0.50)
        self.declare_parameter('iou_threshold',   0.40)
        self.declare_parameter('imgsz',           640)
        self.declare_parameter('depth_min_mm',    100)
        self.declare_parameter('depth_max_mm',    2000)
        self.declare_parameter('camera_hz',       30.0)
        self.declare_parameter('detection_hz',    20.0)
        self.declare_parameter('debug_view',      False)

        use_sim             = self.get_parameter('use_sim').value
        sim_topic           = self.get_parameter('sim_image_topic').value
        model_path          = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.iou_threshold  = self.get_parameter('iou_threshold').value
        self.imgsz          = self.get_parameter('imgsz').value
        self.depth_min_mm   = self.get_parameter('depth_min_mm').value
        self.depth_max_mm   = self.get_parameter('depth_max_mm').value
        camera_hz           = self.get_parameter('camera_hz').value
        detection_hz        = self.get_parameter('detection_hz').value
        self.debug_view     = self.get_parameter('debug_view').value

        # ── ONNX ──────────────────────────────────────────────────────────
        if not os.path.isfile(model_path):
            self.get_logger().error(f'ONNX model not found: {model_path}')
            raise FileNotFoundError(model_path)
        self.session    = ort.InferenceSession(
            model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.get_logger().info(f'ONNX loaded: {os.path.basename(model_path)}')

        # ── Shared frame state ────────────────────────────────────────────
        self._lock        = threading.Lock()
        self._rgb_frame   = None   # uint8 HxWx3 BGR
        self._depth_frame = None   # uint16 HxW mm (None in SIM mode)
        self._left_frame  = None   # uint8 HxW mono (None in SIM mode)
        self._right_frame = None   # uint8 HxW mono (None in SIM mode)
        self._cam_info_rgb = None
        self._cam_info_lft = None
        self._cam_info_rgt = None

        # ── Detection state ───────────────────────────────────────────────
        self._lb_scale    = 1.0
        self._lb_pad      = (0, 0)
        self._device      = None
        self._stopped     = threading.Event()

        # ── Publishers ────────────────────────────────────────────────────
        self._pub_det    = self.create_publisher(String,     '/traffic/detection', 10)
        self._pub_dist   = self.create_publisher(Float32,    '/traffic/distance',  10)
        self._pub_img    = self.create_publisher(Image,      '/traffic/image',     10)

        if not use_sim:
            self._pub_rgb      = self.create_publisher(Image,      '/oak/rgb/image_raw',      1)
            self._pub_left     = self.create_publisher(Image,      '/oak/left/image_raw',     1)
            self._pub_right    = self.create_publisher(Image,      '/oak/right/image_raw',    1)
            self._pub_stereo   = self.create_publisher(Image,      '/oak/stereo/image_raw',   1)
            self._pub_rgb_info = self.create_publisher(CameraInfo, '/oak/rgb/camera_info',    1)
            self._pub_lft_info = self.create_publisher(CameraInfo, '/oak/left/camera_info',   1)
            self._pub_rgt_info = self.create_publisher(CameraInfo, '/oak/right/camera_info',  1)
            self._build_oak_pipeline()
            self.create_timer(1.0 / camera_hz,    self._camera_cb)
        else:
            self.create_subscription(Image, sim_topic, self._sim_image_cb, 1)
            self.get_logger().info(f'SIM mode — subscribing to {sim_topic}')

        self.create_timer(1.0 / detection_hz, self._detection_cb)

        mode = 'SIM' if use_sim else 'OAK-D'
        self.get_logger().info(
            f'OakCameraNode ready [{mode}]\n'
            f'  model      → {model_path}\n'
            f'  camera_hz  → {camera_hz:.0f} Hz\n'
            f'  detect_hz  → {detection_hz:.0f} Hz\n'
            f'  debug_view → {self.debug_view}'
        )

    # ── OAK-D pipeline ───────────────────────────────────────────────────────

    def _build_oak_pipeline(self):
        import depthai as dai

        self._device   = dai.Device()
        self._pipeline = dai.Pipeline(self._device)

        # RGB 1280×720
        cam     = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam_out = cam.requestOutput((1280, 720), dai.ImgFrame.Type.BGR888p)

        # Mono cams 640×400
        mono_l = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        mono_r = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        # Stereo depth aligned to RGB
        stereo = self._pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(False)
        stereo.setExtendedDisparity(True)   # better accuracy at short range
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(1280, 720)

        # Link mono → stereo; keep output handles for separate queues
        left_out  = mono_l.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8)
        right_out = mono_r.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8)
        left_out.link(stereo.left)
        right_out.link(stereo.right)

        self._q_rgb   = cam_out.createOutputQueue(maxSize=2, blocking=False)
        self._q_depth = stereo.depth.createOutputQueue(maxSize=2, blocking=False)
        self._q_left  = left_out.createOutputQueue(maxSize=2, blocking=False)
        self._q_right = right_out.createOutputQueue(maxSize=2, blocking=False)

        self._pipeline.start()

        # Camera intrinsics for camera_info topics
        try:
            calib = self._device.readCalibration()
            self._cam_info_rgb = self._make_cam_info(
                calib, dai.CameraBoardSocket.CAM_A, 1280, 720, 'oak_rgb')
            self._cam_info_lft = self._make_cam_info(
                calib, dai.CameraBoardSocket.CAM_B, 640,  400, 'oak_left')
            self._cam_info_rgt = self._make_cam_info(
                calib, dai.CameraBoardSocket.CAM_C, 640,  400, 'oak_right')
        except Exception as e:
            self.get_logger().warn(f'Calibration read failed: {e}')

        # Daemon thread polls queues at camera frame rate
        self._poll_thread = threading.Thread(
            target=self._poll_loop, name='oak_poll', daemon=True)
        self._poll_thread.start()
        self.get_logger().info('OAK-D Pro pipeline started')

    def _poll_loop(self):
        """Daemon thread: drain DepthAI queues and write to shared state."""
        while not self._stopped.is_set():
            try:
                in_rgb   = self._q_rgb.tryGet()
                in_depth = self._q_depth.tryGet()
                in_left  = self._q_left.tryGet()
                in_right = self._q_right.tryGet()
            except Exception:
                self._stopped.wait(0.05)
                continue

            updated = False
            with self._lock:
                if in_rgb   is not None:
                    self._rgb_frame   = in_rgb.getCvFrame()
                    updated = True
                if in_depth is not None:
                    self._depth_frame = in_depth.getFrame()
                    updated = True
                if in_left  is not None:
                    self._left_frame  = in_left.getCvFrame()
                    updated = True
                if in_right is not None:
                    self._right_frame = in_right.getCvFrame()
                    updated = True
            if not updated:
                self._stopped.wait(0.005)

    # ── SIM mode subscriber ───────────────────────────────────────────────────

    def _sim_image_cb(self, msg: Image) -> None:
        try:
            frame = _imgmsg_to_cv2(msg)
            with self._lock:
                self._rgb_frame = frame
        except Exception as e:
            self.get_logger().warn(f'imgmsg_to_cv2 error: {e}')

    # ── Camera publish timer (OAK-D mode only) ────────────────────────────────

    def _camera_cb(self):
        with self._lock:
            rgb   = self._rgb_frame
            depth = self._depth_frame
            left  = self._left_frame
            right = self._right_frame

        now = self.get_clock().now().to_msg()

        if rgb is not None:
            self._pub_rgb.publish(_make_imgmsg(rgb,   'bgr8',   now, 'oak_rgb'))
            if self._cam_info_rgb is not None:
                self._cam_info_rgb.header.stamp = now
                self._pub_rgb_info.publish(self._cam_info_rgb)

        if depth is not None:
            self._pub_stereo.publish(_make_imgmsg(depth, 'mono16', now, 'oak_stereo'))

        if left is not None:
            self._pub_left.publish(_make_imgmsg(left,  'mono8',  now, 'oak_left'))
            if self._cam_info_lft is not None:
                self._cam_info_lft.header.stamp = now
                self._pub_lft_info.publish(self._cam_info_lft)

        if right is not None:
            self._pub_right.publish(_make_imgmsg(right, 'mono8', now, 'oak_right'))
            if self._cam_info_rgt is not None:
                self._cam_info_rgt.header.stamp = now
                self._pub_rgt_info.publish(self._cam_info_rgt)

    # ── Detection timer (20 Hz) ───────────────────────────────────────────────

    def _detection_cb(self):
        with self._lock:
            frame = self._rgb_frame
            depth = self._depth_frame

        if frame is None:
            return

        H, W = frame.shape[:2]
        output = self.session.run(None, {self.input_name: self._preprocess(frame)})
        dets   = self._postprocess(output[0], H, W)

        for det in dets:
            det['dist_m'] = self._get_distance_m(
                depth, det['x1'], det['y1'], det['x2'], det['y2'])

        # Sort: valid distance first (nearest first), then detections with no depth
        dets = (sorted([d for d in dets if d['dist_m'] > 0], key=lambda d: d['dist_m'])
                + [d for d in dets if d['dist_m'] <= 0])

        # Nearest candidate: closest by distance, or highest confidence if no depth
        nearest = next((d for d in dets if d['dist_m'] > 0), None)
        if nearest is None and dets:
            nearest = max(dets, key=lambda d: d['conf'])

        # Always publish current distance (-1.0 when nothing seen)
        dist_val = float(nearest['dist_m']) if nearest and nearest['dist_m'] > 0 else -1.0
        self._pub_dist.publish(Float32(data=dist_val))

        # Publish nearest sign on every frame — no cooldown
        if nearest is not None:
            cls_id     = nearest['cls']
            class_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f'cls_{cls_id}'
            dist_m     = nearest['dist_m']
            conf       = nearest['conf']
            payload = json.dumps({
                'sign':       class_name,
                'distance_m': round(dist_m, 3) if dist_m > 0 else None,
                'confidence': round(conf, 3),
            })
            self._pub_det.publish(String(data=payload))
            self.get_logger().info(f'SIGN → {payload}')

        # Publish annotated image
        annotated = self._annotate(frame.copy(), dets, nearest)
        try:
            self._pub_img.publish(_make_imgmsg(annotated, 'rgb8'))
        except Exception as e:
            self.get_logger().warn(f'annotated img publish: {e}')

        if self.debug_view:
            cv2.imshow('OAK Camera Node', cv2.resize(annotated, (640, 360)))
            cv2.waitKey(1)

    # ── YOLO helpers ─────────────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Letterbox to imgsz×imgsz, store scale/pad for postprocess."""
        fh, fw   = frame.shape[:2]
        scale    = self.imgsz / max(fh, fw)
        new_w    = int(fw * scale)
        new_h    = int(fh * scale)
        resized  = cv2.resize(frame, (new_w, new_h))
        pad_top  = (self.imgsz - new_h) // 2
        pad_bot  = self.imgsz - new_h - pad_top
        pad_left = (self.imgsz - new_w) // 2
        pad_right= self.imgsz - new_w - pad_left
        blob_img = cv2.copyMakeBorder(
            resized, pad_top, pad_bot, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114))
        self._lb_scale = scale
        self._lb_pad   = (pad_top, pad_left)
        blob = np.transpose(blob_img.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis]
        return np.ascontiguousarray(blob)

    def _postprocess(self, output: np.ndarray, orig_h: int, orig_w: int) -> list:
        pred       = output[0].T
        class_conf = pred[:, 4:]
        scores     = np.max(class_conf, axis=1)
        cls_ids    = np.argmax(class_conf, axis=1)
        boxes      = pred[:, :4]
        mask       = scores >= self.conf_threshold
        boxes, scores, cls_ids = boxes[mask], scores[mask], cls_ids[mask]
        if len(boxes) == 0:
            return []

        pad_top, pad_left = self._lb_pad
        scale = self._lb_scale
        cx, cy, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        x1 = np.clip(((cx - w/2) - pad_left) / scale, 0, orig_w - 1)
        y1 = np.clip(((cy - h/2) - pad_top)  / scale, 0, orig_h - 1)
        x2 = np.clip(((cx + w/2) - pad_left) / scale, 0, orig_w - 1)
        y2 = np.clip(((cy + h/2) - pad_top)  / scale, 0, orig_h - 1)

        detections = []
        for cls_id in np.unique(cls_ids):
            idx       = np.where(cls_ids == cls_id)[0]
            nms_boxes = np.stack([x1[idx], y1[idx], x2[idx]-x1[idx], y2[idx]-y1[idx]], axis=1)
            keep      = cv2.dnn.NMSBoxes(
                nms_boxes.tolist(), scores[idx].tolist(),
                self.conf_threshold, self.iou_threshold)
            for k in keep:
                ki = k[0] if isinstance(k, (list, tuple, np.ndarray)) else int(k)
                detections.append({
                    'cls':  int(cls_ids[idx[ki]]),
                    'conf': float(scores[idx[ki]]),
                    'x1':   int(x1[idx[ki]]), 'y1': int(y1[idx[ki]]),
                    'x2':   int(x2[idx[ki]]), 'y2': int(y2[idx[ki]]),
                })
        return detections

    def _get_distance_m(self, depth_frame, x1, y1, x2, y2) -> float:
        """Sample central ROI of bbox from aligned depth frame (mm → metres)."""
        if depth_frame is None:
            return -1.0
        dh, dw = depth_frame.shape[:2]
        dx1 = int(np.clip(x1, 0, dw-1))
        dy1 = int(np.clip(y1, 0, dh-1))
        dx2 = int(np.clip(x2, 0, dw-1))
        dy2 = int(np.clip(y2, 0, dh-1))
        cx, cy = (dx1+dx2)//2, (dy1+dy2)//2
        rw = max(1, (dx2-dx1)//4)
        rh = max(1, (dy2-dy1)//4)
        roi   = depth_frame[max(0, cy-rh): cy+rh, max(0, cx-rw): cx+rw]
        valid = roi[(roi > self.depth_min_mm) & (roi < self.depth_max_mm)]
        if valid.size == 0:
            return -1.0
        return float(np.median(valid)) / 1000.0

    def _annotate(self, frame: np.ndarray, dets: list, nearest) -> np.ndarray:
        cv2.putText(frame, f'{len(dets)} sign(s)',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        for det in dets:
            cls_id = det['cls']
            name   = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f'cls_{cls_id}'
            color  = CLASS_COLORS_BGR[cls_id % len(CLASS_COLORS_BGR)]
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            is_nearest = (det is nearest)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_nearest else 1)
            if is_nearest:
                cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (255, 255, 255), 1)
            dist_str = f"{det['dist_m']:.2f}m" if det['dist_m'] > 0 else 'n/a'
            label    = f"{name} {dist_str} ({det['conf']:.0%})" + (' [NEAREST]' if is_nearest else '')
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+4, y1), color, -1)
            cv2.putText(frame, label, (x1+2, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return frame

    # ── CameraInfo builder ────────────────────────────────────────────────────

    def _make_cam_info(self, calib, socket, w: int, h: int, frame_id: str) -> CameraInfo:
        msg = CameraInfo()
        msg.width, msg.height = w, h
        msg.header.frame_id   = frame_id
        msg.distortion_model  = 'plumb_bob'
        try:
            K_raw = calib.getCameraIntrinsics(socket, w, h)
            msg.k = [float(v) for row in K_raw for v in row]
            msg.d = [float(v) for v in calib.getDistortionCoefficients(socket)]
        except Exception:
            msg.k = [0.0] * 9
            msg.d = []
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        k = msg.k
        msg.p = [k[0], k[1], k[2], 0.0,
                 k[3], k[4], k[5], 0.0,
                 k[6], k[7], k[8], 0.0]
        return msg

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        self._stopped.set()
        if hasattr(self, '_poll_thread'):
            try:
                self._poll_thread.join(timeout=1.0)
            except Exception:
                pass
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
        cv2.destroyAllWindows()
        super().destroy_node()


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = OakCameraNode()
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
