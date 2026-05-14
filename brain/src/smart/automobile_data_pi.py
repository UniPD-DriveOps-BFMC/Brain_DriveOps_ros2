#!/usr/bin/env python3
"""
automobile_data_pi.py — ROS2 Jazzy
All publishers, subscribers, and callbacks for the physical car.

Camera path
-----------
This file supports OPTION A from the integration plan: the brain
process opens the OAK-D Pro pipeline directly (via DepthAI) and
publishes both the RGB frame on `self.frame` and the stereo depth
on `self.depth_frame`.  The depthai_ros_driver node is therefore
NOT required at runtime; only `main_brain.py` and the micro-ROS
agent (for the Nucleo) need to be running.

If `trig_cam_oak=True` (the default in this option-A build), a
small daemon thread polls the DepthAI output queues at the camera
rate (~30 Hz) and updates `self.frame` / `self.depth_frame`
in-place.  The legacy ROS-image subscriber (`trig_cam=True`,
listening on /oak/rgb/image_raw) is preserved for the simulator
path and for laptop-side testing.
"""

import threading
import rclpy
from rclpy.node import Node
import collections
import numpy as np

from std_msgs.msg    import Float32, Bool, String, UInt8
from sensor_msgs.msg import LaserScan, Image, CameraInfo
try:
    from cv_bridge import CvBridge
    _CV_BRIDGE_AVAILABLE = True
except ImportError:
    CvBridge = None
    _CV_BRIDGE_AVAILABLE = False
from utils.msg       import IMU, Localisation, Vehicles, Conditions

from automobile_data_interface import Automobile_Data
import helper_functions as hf

# DepthAI is imported lazily so that the simulator path (no OAK-D
# attached, no DepthAI installed) still works.
try:
    import depthai as dai
    _DEPTHAI_AVAILABLE = True
except Exception:
    dai = None
    _DEPTHAI_AVAILABLE = False

SONAR_THRESHOLD      = 5
SONAR_DEQUE_LENGTH   = 20
TOF_DEQUE_LENGTH     = 10
IMU_DEQUE_LENGTH     = 10
CLASSIFY_DEQUE_LENGTH = 4


# ─────────────────────────────────────────────────────────────────────────
# OAK-D Pro direct pipeline (option A)
# ─────────────────────────────────────────────────────────────────────────
class OakDPipeline:
    """
    Wraps the DepthAI device + pipeline for the OAK-D Pro mounted on
    the car.  Runs in a daemon thread that polls the RGB and stereo
    depth output queues and writes the latest values into
    `self.rgb_frame` / `self.depth_frame`.  The brain reads these
    attributes via the wrappers `Automobile_Data.frame` /
    `Automobile_Data.depth_frame`.

    Mirrors the pipeline used by `traffic_sign_node.py` so the YOLO
    weights (`models/best2.onnx`) operate on identical inputs:

        ┌──────────────┐           ┌────────────────┐
        │  CAM_A RGB   │ 1280×960  │   self.rgb     │
        └──────────────┘           └────────────────┘
        ┌──────────────┐                    ▲
        │ CAM_B mono   │ 640×400  ┐         │
        ├──────────────┤          │ Stereo  │ 1280×960
        │ CAM_C mono   │ 640×400  ┘ Depth   │ depth aligned
        └──────────────┘            FAST_   │ to CAM_A
                                    DENSITY ▼
                                    ┌──────────────┐
                                    │ self.depth   │
                                    └──────────────┘
    """

    def __init__(self, rgb_size=(1280, 960), mono_size=(640, 400),
                 stereo_preset='FAST_DENSITY'):
        if not _DEPTHAI_AVAILABLE:
            raise RuntimeError(
                'DepthAI is not installed; cannot open OAK-D Pro pipeline. '
                'Install with `pip install depthai>=3.0` or use the '
                'simulator profile.')

        self._device   = dai.Device()
        self._pipeline = dai.Pipeline(self._device)

        # ── RGB cam (CAM_A) ──────────────────────────────────────────────
        cam = self._pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A)
        cam_out = cam.requestOutput(rgb_size, dai.ImgFrame.Type.BGR888p)

        # ── Mono cams (CAM_B, CAM_C) ─────────────────────────────────────
        mono_l = self._pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_B)
        mono_r = self._pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_C)

        # ── Stereo depth ─────────────────────────────────────────────────
        stereo = self._pipeline.create(dai.node.StereoDepth)
        preset_map = {
            'FAST_DENSITY':  dai.node.StereoDepth.PresetMode.FAST_DENSITY,
            'HIGH_DETAIL':   dai.node.StereoDepth.PresetMode.HIGH_DETAIL,
            'FAST_ACCURACY': dai.node.StereoDepth.PresetMode.FAST_ACCURACY,
        }
        stereo.setDefaultProfilePreset(
            preset_map.get(stereo_preset,
                           dai.node.StereoDepth.PresetMode.FAST_DENSITY))
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(False)
        stereo.setExtendedDisparity(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(*rgb_size)

        _left_out = mono_l.requestOutput(mono_size, dai.ImgFrame.Type.GRAY8)
        _left_out.link(stereo.left)
        _right_out = mono_r.requestOutput(mono_size, dai.ImgFrame.Type.GRAY8)
        _right_out.link(stereo.right)

        self._q_rgb   = cam_out.createOutputQueue(maxSize=2, blocking=False)
        self._q_depth = stereo.depth.createOutputQueue(maxSize=2, blocking=False)
        self._q_left  = _left_out.createOutputQueue(maxSize=2, blocking=False)
        self._q_right = _right_out.createOutputQueue(maxSize=2, blocking=False)

        self._pipeline.start()

        self._rgb_size  = rgb_size
        self._mono_size = mono_size
        try:
            self._calib = self._device.readCalibration()
        except Exception:
            self._calib = None

        # Latest values, updated by the daemon thread
        self.rgb_frame   = None  # uint8  HxWx3 (BGR)
        self.depth_frame = None  # uint16 HxW   (mm)
        self.left_frame  = None  # uint8  HxW   (GRAY)
        self.right_frame = None  # uint8  HxW   (GRAY)
        self._lock       = threading.Lock()
        self._stopped    = threading.Event()
        self._thread     = threading.Thread(
            target=self._poll_loop, name='oakd_poll', daemon=True)
        self._thread.start()

    def _poll_loop(self):
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
            if in_rgb is not None:
                rgb = in_rgb.getCvFrame()
                with self._lock:
                    self.rgb_frame = rgb
                updated = True
            if in_depth is not None:
                depth = in_depth.getFrame()
                with self._lock:
                    self.depth_frame = depth
                updated = True
            if in_left is not None:
                with self._lock:
                    self.left_frame = in_left.getCvFrame()
                updated = True
            if in_right is not None:
                with self._lock:
                    self.right_frame = in_right.getCvFrame()
                updated = True
            if not updated:
                self._stopped.wait(0.005)  # 200 Hz idle poll

    def stop(self):
        self._stopped.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._device.close()
        except Exception:
            pass


class AutomobileDataPi(Automobile_Data, Node):

    def __init__(self,
                 trig_control:  bool = True,
                 trig_bno:      bool = False,
                 trig_enc:      bool = False,
                 trig_sonar:    bool = False,
                 trig_cam:      bool = False,
                 trig_gps:      bool = False,
                 trig_lidar:    bool = False,
                 trig_tof:      bool = False,
                 trig_cam_oak:  bool = True,
                 ) -> None:

        Automobile_Data.__init__(self)
        Node.__init__(self, 'AutomobileDataPi')

        self.YAW_GLOBAL_OFFSET = -90  # degrees — change before starting
        self.yaw_offset = np.deg2rad(self.YAW_GLOBAL_OFFSET)

        # ── Extra buffers ─────────────────────────────────────────────── #
        self.right_sonar_distance_buffer    = collections.deque(maxlen=SONAR_DEQUE_LENGTH)
        self.left_sonar_distance_buffer     = collections.deque(maxlen=SONAR_DEQUE_LENGTH)
        self.center_sonar_distance          = 3.0
        self.center_sonar_distance_buffer   = collections.deque(maxlen=SONAR_DEQUE_LENGTH)
        self.filtered_center_sonar_distance = 3.0
        self.center_tof_distance_buffer     = collections.deque(maxlen=TOF_DEQUE_LENGTH)
        self.left_tof_distance_buffer       = collections.deque(maxlen=TOF_DEQUE_LENGTH)
        self.encoder_velocity_buffer        = collections.deque(maxlen=SONAR_DEQUE_LENGTH)
        self.reachedPosition                = False
        self.obstacle_buffer                = collections.deque(maxlen=CLASSIFY_DEQUE_LENGTH)
        self.sign_buffer                    = collections.deque(maxlen=CLASSIFY_DEQUE_LENGTH)
        self.is_position_reliable           = True
        self.estimation_last_encoder_distance = 0.0
        self.estimation_last_yaw_est        = 0.0
        self.x_buffer                       = collections.deque(maxlen=5)
        self.y_buffer                       = collections.deque(maxlen=5)
        self.yaw_buffer                     = collections.deque(maxlen=IMU_DEQUE_LENGTH)
        self.lidar_angles                   = 0
        self.lidar_ranges                   = 0
        self.yaw_true                       = 0.0
        self.flag_localisation              = False
        self.frame                          = None   # set by _image_callback (ROS) or by _refresh_oak_frames
        self.depth_frame                    = None   # uint16 HxW (mm), populated only when trig_cam_oak=True
        self.left_frame                     = None   # uint8 HxW (GRAY), populated when trig_cam_oak=True
        self.right_frame                    = None   # uint8 HxW (GRAY), populated when trig_cam_oak=True
        self._oak                           = None   # OakDPipeline instance, see below
        self._bridge                        = None   # CvBridge, only used when trig_cam=True (option B)
        self._oak_bridge                    = None   # CvBridge for publishing oak ROS topics
        self._pub_oak_rgb                   = None
        self._pub_oak_left                  = None
        self._pub_oak_right                 = None
        self._pub_oak_rgb_info              = None
        self._pub_oak_left_info             = None
        self._pub_oak_right_info            = None
        self._cam_info_rgb                  = None
        self._cam_info_left                 = None
        self._cam_info_right                = None

        # ── Publishers & subscribers ──────────────────────────────────── #
        if trig_control:
            self.pub_speed         = self.create_publisher(Float32,     '/automobile/command/speed',    1)
            self.pub_steer         = self.create_publisher(Float32,     '/automobile/command/steer',    1)
            self.pub_stop          = self.create_publisher(Float32,     '/automobile/command/stop',     1)
            self.pub_position      = self.create_publisher(Float32,     '/automobile/command/position', 1)
            self.pub_closest_node  = self.create_publisher(Float32,     '/automobile/closest_node',     1)
            self.pub_next_event    = self.create_publisher(String,      '/automobile/next_event',       1)
            self.pub_prev_event    = self.create_publisher(String,      '/automobile/prev_event',       1)
            self.pub_current_state = self.create_publisher(String,      '/automobile/current_state',    1)
            self.pub_routines      = self.create_publisher(String,      '/automobile/routines',         1)
            self.pub_conditions    = self.create_publisher(Conditions,  '/automobile/conditions',       1)
            self.pub_arena         = self.create_publisher(Bool,        '/automobile/arena',            1)
            self.pub_led           = self.create_publisher(Bool,        '/automobile/led',              1)
            # brain publishes its own estimated position here (NOT the GPS input)
            self.pub_localisation  = self.create_publisher(Localisation, '/automobile/localisation',   1)
            self.sub_position      = self.create_subscription(
                Bool, '/automobile/feedback/position', self.feedback_position_callback, 1)

        if trig_bno:
            self.sub_imu = self.create_subscription(
                IMU, '/oak/imu/data', self.imu_callback, 10)

        if trig_enc:
            self.sub_encSpeed = self.create_subscription(
                Float32, '/automobile/encoder/speed',    self.encoder_velocity_callback, 10)
            self.sub_encDist  = self.create_subscription(
                Float32, '/automobile/encoder/distance', self.encoder_distance_callback, 10)
            self.reset_rel_pose()

        if trig_sonar:
            self.sub_son_center = self.create_subscription(
                Float32, '/automobile/sonar/center', self.center_sonar_callback, 1)
            self.sub_son_right  = self.create_subscription(
                Float32, '/automobile/sonar/right',  self.right_sonar_callback,  1)
            self.sub_son_left   = self.create_subscription(
                Float32, '/automobile/sonar/left',   self.left_sonar_callback,   1)

        if trig_cam:
            # ── Legacy path: subscribe to /oak/rgb/image_raw ─────────────
            # Used by the simulator (Gazebo publishes there) and by the
            # depthai_ros_driver path (option B) if the user later
            # decides to spin one up alongside the brain.
            if not _CV_BRIDGE_AVAILABLE:
                raise RuntimeError('cv_bridge is required when trig_cam=True but is not installed.')
            self._bridge = CvBridge()
            self.create_subscription(
                Image, '/oak/rgb/image_raw', self._image_callback, 1)

        if trig_cam_oak:
            # ── Option A: open the OAK-D Pro directly from the brain ─────
            # Spawns a daemon thread that polls DepthAI queues at the
            # camera rate.  `self.frame` and `self.depth_frame` are
            # refreshed in-place; the brain reads them just like before.
            # The 50 Hz timer also re-publishes the frames as ROS topics
            # (/oak/rgb/image_raw, /oak/left/image_raw, /oak/right/image_raw,
            #  plus matching /camera_info) so that camera_to_socket and other
            # nodes see frames without a separate depthai_ros_driver process.
            try:
                self._oak = OakDPipeline()
                self.create_timer(1.0 / 50.0, self._refresh_oak_frames)
                self.get_logger().info(
                    'OAK-D Pro pipeline opened (option A, direct DepthAI)')
                if _CV_BRIDGE_AVAILABLE:
                    self._oak_bridge         = CvBridge()
                    self._pub_oak_rgb        = self.create_publisher(Image,       '/oak/rgb/image_raw',    1)
                    self._pub_oak_left       = self.create_publisher(Image,       '/oak/left/image_raw',   1)
                    self._pub_oak_right      = self.create_publisher(Image,       '/oak/right/image_raw',  1)
                    self._pub_oak_rgb_info   = self.create_publisher(CameraInfo,  '/oak/rgb/camera_info',  1)
                    self._pub_oak_left_info  = self.create_publisher(CameraInfo,  '/oak/left/camera_info', 1)
                    self._pub_oak_right_info = self.create_publisher(CameraInfo,  '/oak/right/camera_info',1)
                    if self._oak._calib is not None:
                        self._cam_info_rgb   = self._make_cam_info(
                            self._oak._calib, dai.CameraBoardSocket.CAM_A,
                            *self._oak._rgb_size,  'oak_rgb')
                        self._cam_info_left  = self._make_cam_info(
                            self._oak._calib, dai.CameraBoardSocket.CAM_B,
                            *self._oak._mono_size, 'oak_left')
                        self._cam_info_right = self._make_cam_info(
                            self._oak._calib, dai.CameraBoardSocket.CAM_C,
                            *self._oak._mono_size, 'oak_right')
            except Exception as e:
                self.get_logger().error(
                    f'OAK-D Pro init failed: {e}.  '
                    f'Lane keeper and YOLO will not receive frames.')
                self._oak = None

        if trig_gps:
            # GPS comes from the competition bridge on this topic (mm → m already converted)
            self.sub_gps = self.create_subscription(
                Localisation, '/automobile/localisation/gps', self.position_callback, 1)

        if trig_lidar:
            self.sub_lidar = self.create_subscription(
                LaserScan, '/scan', self.lidar_callback, 10)

        if trig_tof:
            self.sub_tof_center = self.create_subscription(
                UInt8, '/automobile/tof/front', self.center_tof_callback, 1)
            self.sub_tof_left   = self.create_subscription(
                UInt8, '/automobile/tof/left',  self.left_tof_callback,   1)

    # ═══════════════════════════════════════════════════════════════════ #
    #  SENSOR CALLBACKS                                                   #
    # ═══════════════════════════════════════════════════════════════════ #

    def center_sonar_callback(self, data: Float32) -> None:
        self.center_sonar_distance = data.data if data.data > 0 else self.center_sonar_distance
        self.center_sonar_distance_buffer.append(self.center_sonar_distance)
        self.filtered_center_sonar_distance = np.median(self.center_sonar_distance_buffer)
        self.sonar_distance          = self.center_sonar_distance
        self.filtered_sonar_distance = self.filtered_center_sonar_distance

    def right_sonar_callback(self, data: Float32) -> None:
        self.right_sonar_distance = data.data if data.data > 0 else self.right_sonar_distance
        self.right_sonar_distance_buffer.append(self.right_sonar_distance)
        self.filtered_right_sonar_distance = np.median(self.right_sonar_distance_buffer)

    def left_sonar_callback(self, data: Float32) -> None:
        self.left_sonar_distance = data.data if data.data > 0 else self.left_sonar_distance
        self.left_sonar_distance_buffer.append(self.left_sonar_distance)
        self.filtered_left_sonar_distance = np.median(self.left_sonar_distance_buffer)

    def center_tof_callback(self, data: UInt8) -> None:
        self.center_tof_distance = data.data if data.data > 0 else self.center_tof_distance
        self.center_tof_distance_buffer.append(self.center_tof_distance / 1000.0)  # mm → m
        self.filtered_center_tof_distance = np.median(self.center_tof_distance_buffer)

    def left_tof_callback(self, data: UInt8) -> None:
        self.left_tof_distance = data.data if data.data > 0 else self.left_tof_distance
        self.left_tof_distance_buffer.append(self.left_tof_distance / 1000.0)  # mm → m
        self.filtered_left_tof_distance = np.median(self.left_tof_distance_buffer)

    def lidar_callback(self, data: LaserScan) -> None:
        self.lidar_angles = np.linspace(data.angle_min, data.angle_max, len(data.ranges))
        self.lidar_ranges = np.array(data.ranges)

    def position_callback(self, data: Localisation) -> None:
        """GPS position from competition bridge.
        data.pos_a / pos_b are already in metres (bridge divides mm by 1000).
        """
        pL     = np.array([data.pos_a, data.pos_b])
        pR     = hf.mL2mR(pL)
        tmp_x  = pR[0] - self.WB / 2 * np.cos(self.yaw)
        tmp_y  = pR[1] - self.WB / 2 * np.sin(self.yaw)
        self.x_buffer.append(tmp_x)
        self.y_buffer.append(tmp_y)
        self.x     = np.mean(self.x_buffer)
        self.y     = np.mean(self.y_buffer)
        self.x_est = self.x
        self.y_est = self.y
        self.x_GPS = self.x
        self.y_GPS = self.y
        self.flag_localisation = True

    def imu_callback(self, data: IMU) -> None:
        self.roll      = float(data.roll)
        self.roll_deg  = np.rad2deg(self.roll)
        self.pitch     = float(data.pitch)
        self.pitch_deg = np.rad2deg(self.pitch)
        self.yaw_true  = float(data.yaw)
        self.yaw       = float(data.yaw) + self.yaw_offset
        self.yaw_deg   = np.rad2deg(self.yaw)

    def encoder_distance_callback(self, data: Float32) -> None:
        self.encoder_distance = data.data
        self.update_rel_position()

    def encoder_velocity_callback(self, data: Float32) -> None:
        self.encoder_velocity = data.data
        self.encoder_velocity_buffer.append(self.encoder_velocity)
        self.filtered_encoder_velocity = np.median(self.encoder_velocity_buffer)

    def obstacle_callback(self, data) -> None:
        self.obstacle = data.data
        self.obstacle_buffer.append(self.obstacle)
        self.filtered_obstacle = np.median(self.obstacle_buffer)

    def sign_callback(self, data) -> None:
        self.sign = data.data
        self.sign_buffer.append(self.sign)
        self.filtered_sign = np.median(self.sign_buffer)

    def feedback_position_callback(self, data: Bool) -> None:
        self.reachedPosition = data.data

    def _make_cam_info(self, calib, socket, w, h, frame_id) -> CameraInfo:
        msg = CameraInfo()
        msg.width  = w
        msg.height = h
        msg.header.frame_id   = frame_id
        msg.distortion_model  = 'plumb_bob'
        try:
            K_raw = calib.getCameraIntrinsics(socket, w, h)  # 3×3 list of lists
            msg.k = [float(v) for row in K_raw for v in row]
            D_raw = calib.getDistortionCoefficients(socket)
            msg.d = [float(v) for v in D_raw]
        except Exception:
            msg.k = [0.0] * 9
            msg.d = []
        msg.r = [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0]
        k = msg.k
        msg.p = [k[0], k[1], k[2], 0.0,
                 k[3], k[4], k[5], 0.0,
                 k[6], k[7], k[8], 0.0]
        return msg

    def _image_callback(self, msg: Image) -> None:
        # Used by the simulator and by option-B (depthai_ros_driver).
        # When the OAK-D direct pipeline is also active (option A), the
        # 50 Hz timer below overwrites self.frame with the DepthAI frame,
        # so this callback effectively becomes a no-op on the real car.
        if self._bridge is None:
            # Defensive: if trig_cam=False this callback was never wired
            # into a subscription, but a pathological caller could still
            # invoke it.  Bail out instead of crashing on AttributeError.
            return
        self.frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _refresh_oak_frames(self) -> None:
        """Periodic copy from the OAK-D daemon thread to brain-visible state."""
        if self._oak is None:
            return
        with self._oak._lock:
            rgb   = self._oak.rgb_frame
            depth = self._oak.depth_frame
            left  = self._oak.left_frame
            right = self._oak.right_frame
        now = self.get_clock().now().to_msg()
        if rgb is not None:
            self.frame = rgb
            if self._pub_oak_rgb is not None:
                try:
                    img_msg = self._oak_bridge.cv2_to_imgmsg(rgb, encoding='bgr8')
                    img_msg.header.stamp = now
                    img_msg.header.frame_id = 'oak_rgb'
                    self._pub_oak_rgb.publish(img_msg)
                    if self._pub_oak_rgb_info is not None and self._cam_info_rgb is not None:
                        self._cam_info_rgb.header.stamp = now
                        self._pub_oak_rgb_info.publish(self._cam_info_rgb)
                except Exception:
                    pass
        if depth is not None:
            self.depth_frame = depth
        if left is not None:
            self.left_frame = left
            if self._pub_oak_left is not None:
                try:
                    img_msg = self._oak_bridge.cv2_to_imgmsg(left, encoding='mono8')
                    img_msg.header.stamp = now
                    img_msg.header.frame_id = 'oak_left'
                    self._pub_oak_left.publish(img_msg)
                    if self._pub_oak_left_info is not None and self._cam_info_left is not None:
                        self._cam_info_left.header.stamp = now
                        self._pub_oak_left_info.publish(self._cam_info_left)
                except Exception:
                    pass
        if right is not None:
            self.right_frame = right
            if self._pub_oak_right is not None:
                try:
                    img_msg = self._oak_bridge.cv2_to_imgmsg(right, encoding='mono8')
                    img_msg.header.stamp = now
                    img_msg.header.frame_id = 'oak_right'
                    self._pub_oak_right.publish(img_msg)
                    if self._pub_oak_right_info is not None and self._cam_info_right is not None:
                        self._cam_info_right.header.stamp = now
                        self._pub_oak_right_info.publish(self._cam_info_right)
                except Exception:
                    pass

    # ═══════════════════════════════════════════════════════════════════ #
    #  COMMAND ACTIONS                                                    #
    # ═══════════════════════════════════════════════════════════════════ #

    def drive_speed(self, speed: float = 0.0) -> None:
        speed      = Automobile_Data.normalizeSpeed(speed)
        self.speed = speed
        self.pub_speed.publish(Float32(data=float(speed)))

    def drive_angle(self, angle: float = 0.0) -> None:
        angle      = Automobile_Data.normalizeSteer(angle)
        self.steer = angle
        self.pub_steer.publish(Float32(data=float(angle)))

    def stop(self, angle: float = 0.0) -> None:
        angle      = Automobile_Data.normalizeSteer(angle)
        self.steer = angle
        self.pub_stop.publish(Float32(data=float(angle)))

    # ═══════════════════════════════════════════════════════════════════ #
    #  ADDITIONAL METHODS                                                 #
    # ═══════════════════════════════════════════════════════════════════ #

    def drive_distance(self, dist: float = 0.0) -> None:
        self.reachedPosition = False
        self.pub_position.publish(Float32(data=float(dist)))

    def publish_closest_node(self, data: float = 0.0) -> None:
        self.pub_closest_node.publish(Float32(data=float(data)))

    def publish_next_event(self, data: str) -> None:
        self.pub_next_event.publish(String(data=str(data)))

    def publish_prev_event(self, data: str) -> None:
        self.pub_prev_event.publish(String(data=str(data)))

    def publish_current_state(self, data: str) -> None:
        self.pub_current_state.publish(String(data=str(data)))

    def publish_arena_flag(self, data: bool) -> None:
        self.pub_arena.publish(Bool(data=bool(data)))

    def publish_routines(self, data: str) -> None:
        self.pub_routines.publish(String(data=str(data)))

    def publish_conditions(self, data: dict) -> None:
        msg = Conditions(
            can_overtake = bool(data['can_overtake']),
            highway      = bool(data['highway']),
            car_on_path  = bool(data['car_on_path']),
            rerouting    = bool(data['rerouting']),
            tunnel       = bool(data['tunnel']),
        )
        self.pub_conditions.publish(msg)

    def publish_led_control(self, data: bool) -> None:
        self.pub_led.publish(Bool(data=bool(data)))

    def publish_localisation(self, x: float, y: float) -> None:
        """Publish the brain's estimated position (NOT the raw GPS input)."""
        msg = Localisation(
            pos_a     = float(x),
            pos_b     = float(y),
            timestamp = 0.0,
            rot_a     = 0.0,
            rot_b     = 0.0,
        )
        self.pub_localisation.publish(msg)

    def destroy_node(self) -> None:
        if self._oak is not None:
            try:
                self._oak.stop()
            except Exception:
                pass
        super().destroy_node()