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
from sensor_msgs.msg import LaserScan, Image
from cv_bridge       import CvBridge
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

# Known sign positions on the map, grouped by class
SIGN_WORLD_POSITIONS = {
    'stop': [
        (0.410312, 11.53633),
        (0.410312,  4.934624),
        (0.814555,  9.982114),
        (1.387690,  7.540225),
        (2.789737,  1.728136),
        (3.768965,  7.540225),
        (3.193981,  6.560997),
        (5.263525,  1.728136),
        (16.86365,  1.321081),
        (17.84570,  4.359640),
    ],
    'priority': [
        (0.405650,  8.115209),
        (0.814555,  6.560997),
        (5.263525,  4.934624),
        (5.667769,  3.382262),
        (4.688541,  0.746096),
        (5.667769,  6.560997),
    ],
    'one_way': [
        (16.86646,  4.934624),
    ],
    'cross_walk': [
        (9.060688,  0.748908),
        (10.43173,  4.364302),
        (16.86646,  3.200937),
        (18.83602,  2.549648),
        (5.689972,  8.115209),
        (1.387690, 10.53490),
        (1.894659, 10.96415),
        (5.263525,  8.622178),
    ],
    'park': [
        (9.950578,  0.748355),
    ],
    'roundabout': [
        (16.55236, 11.40081),
        (17.49554, 12.74824),
        (17.90164, 10.45578),
        (18.84482, 11.80506),
    ],
    'hw_enter': [
        (15.00000, 11.85258),
        (7.366563, 12.94125),
    ],
    'hw_exit': [
        (6.948529, 13.34359),
        (16.19118, 11.40086),
    ],
    'closed_road': [
        (7.309905,  3.789318),
        (16.28937,  0.746574),
    ],
}

# Camera horizontal FOV in radians — calibrate this for your lens
CAMERA_HFOV_RAD = np.deg2rad(62.2)
IMG_WIDTH       = 320  # pixels

# Correction tuning
SIGN_MAX_DIST_M       = 2.5   # ignore detections farther than this
SIGN_MAX_JUMP_M       = 0.8   # reject correction if it would move estimate >this
SIGN_SEARCH_RADIUS_M  = 3.0   # only consider map signs within this radius of x_est
SIGN_ALPHA            = 0.6   # blend weight: 1.0 = hard snap, 0.0 = ignore


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
        │  CAM_A RGB   │ 1280×720  │   self.rgb     │
        └──────────────┘           └────────────────┘
        ┌──────────────┐                    ▲
        │ CAM_B mono   │ 640×400  ┐         │
        ├──────────────┤          │ Stereo  │ 1280×720
        │ CAM_C mono   │ 640×400  ┘ Depth   │ depth aligned
        └──────────────┘            FAST_   │ to CAM_A
                                    DENSITY ▼
                                    ┌──────────────┐
                                    │ self.depth   │
                                    └──────────────┘
    """

    def __init__(self, rgb_size=(1280, 720), mono_size=(640, 400),
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
            'HIGH_ACCURACY': dai.node.StereoDepth.PresetMode.HIGH_ACCURACY,
        }
        stereo.setDefaultProfilePreset(
            preset_map.get(stereo_preset,
                           dai.node.StereoDepth.PresetMode.FAST_DENSITY))
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(False)
        stereo.setExtendedDisparity(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(*rgb_size)

        mono_l.requestOutput(mono_size,
                             dai.ImgFrame.Type.GRAY8).link(stereo.left)
        mono_r.requestOutput(mono_size,
                             dai.ImgFrame.Type.GRAY8).link(stereo.right)

        self._q_rgb   = cam_out.createOutputQueue(maxSize=2, blocking=False)
        self._q_depth = stereo.depth.createOutputQueue(maxSize=2,
                                                       blocking=False)

        self._pipeline.start()

        # Latest values, updated by the daemon thread
        self.rgb_frame   = None  # uint8  HxWx3 (BGR)
        self.depth_frame = None  # uint16 HxW   (mm)
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
            except Exception as e:
                # On device disconnect, sleep a bit and retry
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
        self._oak                           = None   # OakDPipeline instance, see below
        self._bridge                        = None   # CvBridge, only used when trig_cam=True (option B)

        # Tracks which map signs have already been used for a correction
        # key: (sign_class, index_in_list), value: True
        self._used_sign_landmarks = set()
        self._last_corrected_sign  = None  # for logging

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
            self._bridge = CvBridge()
            self.create_subscription(
                Image, '/oak/rgb/image_raw', self._image_callback, 1)

        if trig_cam_oak:
            # ── Option A: open the OAK-D Pro directly from the brain ─────
            # Spawns a daemon thread that polls DepthAI queues at the
            # camera rate.  `self.frame` and `self.depth_frame` are
            # refreshed in-place; the brain reads them just like before.
            try:
                self._oak = OakDPipeline()
                # Replace the static `self.frame` attribute with a
                # property-style refresh: a 50 Hz timer copies the
                # latest values out of the DepthAI thread under a lock.
                self.create_timer(1.0 / 50.0, self._refresh_oak_frames)
                self.get_logger().info(
                    'OAK-D Pro pipeline opened (option A, direct DepthAI)')
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
        if rgb is not None:
            self.frame = rgb
        if depth is not None:
            self.depth_frame = depth

    def correct_position_with_sign(self, detection: dict) -> bool:
        """
        Corrects x_est / y_est using a YOLO sign detection with known distance.

        Parameters
        ----------
        detection : dict
            One entry from self.last_yolo_detections (fields: sign, distance_m,
            x1, x2, conf).

        Returns
        -------
        bool : True if a correction was applied.
        """
        sign_class = detection.get('sign', '')
        distance_m = detection.get('distance_m', -1.0)
        conf       = detection.get('conf', 0.0)
        x1         = detection.get('x1', 0)
        x2         = detection.get('x2', IMG_WIDTH)

        # ── Gate 1: sign class must be in our map ────────────────────────────
        if sign_class not in SIGN_WORLD_POSITIONS:
            return False

        # ── Gate 2: distance must be valid and close enough ──────────────────
        if distance_m <= 0.0 or distance_m > SIGN_MAX_DIST_M:
            return False

        # ── Gate 3: confidence threshold ─────────────────────────────────────
        if conf < 0.60:
            return False

        # ── Find the closest unused map sign of this class ───────────────────
        candidates = SIGN_WORLD_POSITIONS[sign_class]
        best_idx   = None
        best_dist  = float('inf')

        for i, (sx, sy) in enumerate(candidates):
            if (sign_class, i) in self._used_sign_landmarks:
                continue  # already corrected with this landmark
            d_to_estimate = np.hypot(sx - self.x_est, sy - self.y_est)
            if d_to_estimate < SIGN_SEARCH_RADIUS_M and d_to_estimate < best_dist:
                best_dist = d_to_estimate
                best_idx  = i

        if best_idx is None:
            return False  # no unused sign nearby in the map

        sx, sy = candidates[best_idx]

        # ── Compute angle to sign from bbox centre in image ──────────────────
        bbox_cx     = (x1 + x2) / 2.0
        # positive = sign is to the right of centre → car heading rotated right
        angle_offset = ((bbox_cx - IMG_WIDTH / 2.0) / IMG_WIDTH) * CAMERA_HFOV_RAD

        sign_world_angle = self.yaw + angle_offset  # [rad], world frame

        # ── Compute where the car must be ────────────────────────────────────
        corrected_x = sx - distance_m * np.cos(sign_world_angle)
        corrected_y = sy - distance_m * np.sin(sign_world_angle)

        # ── Gate 4: sanity — don't jump more than SIGN_MAX_JUMP_M ───────────
        jump = np.hypot(corrected_x - self.x_est, corrected_y - self.y_est)
        if jump > SIGN_MAX_JUMP_M:
            print(f'[LANDMARK] {sign_class}[{best_idx}] rejected — jump={jump:.2f}m > {SIGN_MAX_JUMP_M}m')
            return False

        # ── Apply soft correction ─────────────────────────────────────────────
        old_x, old_y = self.x_est, self.y_est
        self.x_est = (1.0 - SIGN_ALPHA) * self.x_est + SIGN_ALPHA * corrected_x
        self.y_est = (1.0 - SIGN_ALPHA) * self.y_est + SIGN_ALPHA * corrected_y

        # ── Mark landmark as used ─────────────────────────────────────────────
        self._used_sign_landmarks.add((sign_class, best_idx))
        self._last_corrected_sign = sign_class

        print(f'[LANDMARK] {sign_class}[{best_idx}] @ ({sx:.2f},{sy:.2f}) '
            f'dist={distance_m:.2f}m jump={jump:.2f}m '
            f'({old_x:.2f},{old_y:.2f}) → ({self.x_est:.2f},{self.y_est:.2f})')
        return True

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