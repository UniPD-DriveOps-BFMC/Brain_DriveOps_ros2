#!/usr/bin/python3
"""
automobile_data_simulator.py — ROS2 Jazzy

ROS2 rewrite of the Gazebo simulator interface.
Mirrors the structure of automobile_data_pi.py so Brain can use
either class interchangeably.

Notable differences from automobile_data_pi.py:
  - Commands are JSON strings on /automobile/command (Gazebo bridge format).
  - Encoder distance is computed from ground-truth pose published by the
    simulator (timer-based dead reckoning from x_true/y_true).
  - No OAK-D / DepthAI pipeline; depth_frame is always None.
"""

from automobile_data_interface import Automobile_Data
from std_msgs.msg import String, Float32, Bool
from sensor_msgs.msg import Image, Range, LaserScan
from utils.msg import IMU, Localisation, Conditions

import rclpy
from rclpy.node import Node

import json
import collections
import numpy as np
from time import time
import helper_functions as hf

try:
    from cv_bridge import CvBridge
    _CV_BRIDGE_AVAILABLE = True
except ImportError:
    CvBridge = None
    _CV_BRIDGE_AVAILABLE = False

REALISTIC = False

ENCODER_TIMER = 0.01                                    # [s]
STEER_UPDATE_FREQ = 50.0 if REALISTIC else 150.0        # [Hz]
SERVO_DEAD_TIME_DELAY = 0.15 if REALISTIC else 0.0      # [s]
MAX_SERVO_ANGULAR_VELOCITY = 2.8 if REALISTIC else 15.0 # [rad/s]
DELTA_ANGLE = np.rad2deg(MAX_SERVO_ANGULAR_VELOCITY) / STEER_UPDATE_FREQ
MAX_STEER_COMMAND_FREQ = 50.0
MAX_STEER_SAMPLES = max(int((2 * SERVO_DEAD_TIME_DELAY) * MAX_STEER_COMMAND_FREQ), 10)


class AutomobileDataSimulator(Automobile_Data, Node):
    def __init__(self,
                 trig_control: bool = True,
                 trig_bno:     bool = False,
                 trig_enc:     bool = False,
                 trig_sonar:   bool = False,
                 trig_cam:     bool = False,
                 trig_gps:     bool = False,
                 trig_lidar:   bool = False,
                 trig_tof:     bool = False,
                 ) -> None:
        Automobile_Data.__init__(self)
        Node.__init__(self, 'AutomobileDataSimulator')

        # ── Extra buffers ─────────────────────────────────────────────── #
        self.sonar_distance_buffer       = collections.deque(maxlen=20)
        self.right_sonar_distance_buffer = collections.deque(maxlen=20)
        self.left_sonar_distance_buffer  = collections.deque(maxlen=20)
        self.center_tof_distance_buffer  = collections.deque(maxlen=20)
        self.left_tof_distance_buffer    = collections.deque(maxlen=20)
        self.timestamp      = 0.0
        self.prev_x_true    = self.x_true
        self.prev_y_true    = self.y_true
        self.prev_timestamp = 0.0
        self.velocity_buffer = collections.deque(maxlen=20)
        self.target_steer   = 0.0
        self.curr_steer     = 0.0
        self.steer_deque    = collections.deque(maxlen=MAX_STEER_SAMPLES)
        self.time_last_steer_command = time()
        self.target_dist    = 0.0
        self.arrived_at_dist = True
        self.yaw_true       = 0.0
        self.x_buffer       = collections.deque(maxlen=5)
        self.y_buffer       = collections.deque(maxlen=5)
        self.lidar_angles   = 0
        self.lidar_ranges   = 0
        self.depth_frame    = None   # simulator has no stereo depth

        # ── Publishers & subscribers ──────────────────────────────────── #
        if trig_control:
            self.pub = self.create_publisher(String, '/automobile/command', 1)
            self.create_timer(1.0 / STEER_UPDATE_FREQ, self.steer_update_callback)
            self.create_timer(ENCODER_TIMER, self.drive_distance_callback)
            self.pub_closest_node  = self.create_publisher(Float32, '/automobile/closest_node',  1)
            self.pub_next_event    = self.create_publisher(String,  '/automobile/next_event',    1)
            self.pub_prev_event    = self.create_publisher(String,  '/automobile/prev_event',    1)
            self.pub_current_state = self.create_publisher(String,  '/automobile/current_state', 1)
            self.pub_routines      = self.create_publisher(String,  '/automobile/routines',      1)
            self.pub_conditions    = self.create_publisher(Conditions, '/automobile/conditions', 1)
            self.pub_arena         = self.create_publisher(Bool,    '/automobile/arena',         1)
            self.pub_led           = self.create_publisher(Bool,    '/automobile/led',           1)
            self.pub_localisation  = self.create_publisher(Localisation, '/automobile/localisation', 1)

        if trig_bno:
            self.create_subscription(IMU, '/automobile/IMU', self.imu_callback, 10)

        if trig_enc:
            self.reset_rel_pose()
            self.create_timer(ENCODER_TIMER, self._encoder_timer_callback)

        if trig_sonar:
            self.create_subscription(Range,   '/automobile/sonar1', self.sonar_callback,       10)
            self.create_subscription(Range,   '/automobile/sonar2', self.right_sonar_callback,  10)
            self.create_subscription(Range,   '/automobile/sonar3', self.left_sonar_callback,   10)

        if trig_lidar:
            self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)

        if trig_cam:
            if not _CV_BRIDGE_AVAILABLE:
                raise RuntimeError('cv_bridge is required when trig_cam=True but is not installed.')
            self._bridge = CvBridge()
            self.create_subscription(Image, '/automobile/image_raw', self.camera_callback, 1)

        if trig_gps:
            self.create_subscription(
                Localisation, '/automobile/localisation', self.position_callback, 10)

        if trig_tof:
            self.create_subscription(Float32, '/automobile/tof/right', self.right_tof_callback, 10)
            self.create_subscription(Float32, '/automobile/tof/left',  self.left_tof_callback,  10)

    # ═══════════════════════════════════════════════════════════════════ #
    #  SENSOR CALLBACKS                                                   #
    # ═══════════════════════════════════════════════════════════════════ #

    def camera_callback(self, data: Image) -> None:
        self.frame = self._bridge.imgmsg_to_cv2(data, 'bgr8')

    def sonar_callback(self, data: Range) -> None:
        self.sonar_distance = data.range
        self.sonar_distance_buffer.append(self.sonar_distance)
        self.filtered_sonar_distance = np.median(self.sonar_distance_buffer)

    def right_sonar_callback(self, data: Range) -> None:
        self.right_sonar_distance = data.range
        self.right_sonar_distance_buffer.append(self.right_sonar_distance)
        self.filtered_right_sonar_distance = np.median(self.right_sonar_distance_buffer)

    def left_sonar_callback(self, data: Range) -> None:
        self.left_sonar_distance = data.range
        self.left_sonar_distance_buffer.append(self.left_sonar_distance)
        self.filtered_left_sonar_distance = np.median(self.left_sonar_distance_buffer)

    def center_tof_callback(self, data: Range) -> None:
        self.center_tof_distance = data.range
        self.center_tof_distance_buffer.append(self.center_tof_distance)
        self.filtered_center_tof_distance = np.median(self.center_tof_distance_buffer)

    def right_tof_callback(self, data: Float32) -> None:
        self.right_sonar_distance = data.data

    def left_tof_callback(self, data: Float32) -> None:
        self.left_tof_distance = data.data
        self.left_tof_distance_buffer.append(self.left_tof_distance)
        self.filtered_left_tof_distance = np.median(self.left_tof_distance_buffer)

    def lidar_callback(self, data: LaserScan) -> None:
        self.lidar_angles = np.linspace(data.angle_min, data.angle_max, len(data.ranges))
        self.lidar_ranges = np.array(data.ranges)

    def position_callback(self, data: Localisation) -> None:
        pL = np.array([data.pos_a, data.pos_b])   # fixed: was pos_a, pos_a
        pR = hf.mL2mR(pL)
        tmp_x = pR[0] - self.WB / 2 * np.cos(self.yaw)
        tmp_y = pR[1] - self.WB / 2 * np.sin(self.yaw)
        self.x_buffer.append(tmp_x)
        self.y_buffer.append(tmp_y)
        self.x     = np.mean(self.x_buffer)
        self.y     = np.mean(self.y_buffer)
        self.x_est = self.x
        self.y_est = self.y

    def imu_callback(self, data: IMU) -> None:
        self.roll      = float(data.roll)
        self.roll_deg  = np.rad2deg(self.roll)
        self.pitch     = float(data.pitch)
        self.pitch_deg = np.rad2deg(self.pitch)
        self.yaw_true  = float(data.yaw)
        self.yaw       = float(data.yaw) + self.yaw_offset
        self.yaw_deg   = np.rad2deg(self.yaw)
        true_posL = np.array([data.posx, data.posy])
        true_posR = hf.mL2mR(true_posL)
        self.x_true   = true_posR[0] - self.WB / 2 * np.cos(self.yaw_true)
        self.y_true   = true_posR[1] - self.WB / 2 * np.sin(self.yaw_true)
        self.timestamp = float(data.timestamp)

    def _encoder_timer_callback(self) -> None:
        """Dead-reckoning encoder from ground-truth pose (replaces rospy.Timer)."""
        curr_x    = self.x_true
        curr_y    = self.y_true
        prev_x    = self.prev_x_true
        prev_y    = self.prev_y_true
        curr_time = self.timestamp
        prev_time = self.prev_timestamp
        delta = np.hypot(curr_x - prev_x, curr_y - prev_y)
        motion_yaw    = np.arctan2(curr_y - prev_y, curr_x - prev_x)
        abs_yaw_diff  = np.abs(hf.diff_angle(motion_yaw, self.yaw_true))
        sign = 1.0 if abs_yaw_diff < np.pi / 2 else -1.0
        dt = curr_time - prev_time
        if dt > 0.0:
            velocity = (delta * sign) / dt
            self.encoder_velocity = velocity
            self.velocity_buffer.append(velocity)
            self.filtered_encoder_velocity = np.median(self.velocity_buffer)
            self.encoder_distance += sign * delta
            self.prev_x_true    = curr_x
            self.prev_y_true    = curr_y
            self.prev_timestamp = curr_time
            self.update_rel_position()

    # ═══════════════════════════════════════════════════════════════════ #
    #  TIMERS                                                             #
    # ═══════════════════════════════════════════════════════════════════ #

    def steer_update_callback(self) -> None:
        if len(self.steer_deque) > 0:
            curr_time = time()
            angle, t = self.steer_deque.popleft()
            if curr_time - t < SERVO_DEAD_TIME_DELAY:
                self.steer_deque.appendleft((angle, t))
            else:
                self.target_steer = angle
        diff = self.target_steer - self.curr_steer
        if diff > 0.0:
            incr = min(diff, DELTA_ANGLE)
        elif diff < 0.0:
            incr = max(diff, -DELTA_ANGLE)
        else:
            return
        self.curr_steer += incr
        self._pub_steer_json(self.curr_steer)

    def drive_distance_callback(self) -> None:
        Kp = 0.5
        max_speed = 0.2
        if not self.arrived_at_dist:
            dist_error = self.target_dist - self.encoder_distance
            self._pub_speed_json(min(Kp * dist_error, max_speed))
            if np.abs(dist_error) < 0.01:
                self.arrived_at_dist = True
                self.drive_speed(0.0)

    # ═══════════════════════════════════════════════════════════════════ #
    #  COMMAND ACTIONS                                                    #
    # ═══════════════════════════════════════════════════════════════════ #

    def drive_speed(self, speed: float = 0.0) -> None:
        self.arrived_at_dist = True
        self._pub_speed_json(speed)

    def drive_angle(self, angle: float = 0.0, direct: bool = False) -> None:
        angle = Automobile_Data.normalizeSteer(angle)
        curr_time = time()
        if curr_time - self.time_last_steer_command > 1.0 / MAX_STEER_COMMAND_FREQ:
            self.time_last_steer_command = curr_time
            self.steer_deque.append((angle, curr_time))
        else:
            print('Missed steer command...')

    def drive_distance(self, dist: float = 0.0) -> None:
        self.target_dist     = self.encoder_distance + dist
        self.arrived_at_dist = False

    def stop(self, angle: float = 0.0) -> None:
        self.steer_deque.append((angle, time()))
        self.speed = 0.0
        data = {'action': '3', 'steerAngle': float(angle)}
        self.pub.publish(String(data=json.dumps(data)))

    # ═══════════════════════════════════════════════════════════════════ #
    #  INTERNAL PUBLISHERS                                                #
    # ═══════════════════════════════════════════════════════════════════ #

    def _pub_steer_json(self, angle: float) -> None:
        data = {'action': '2', 'steerAngle': float(angle)}
        self.pub.publish(String(data=json.dumps(data)))

    def _pub_speed_json(self, speed: float) -> None:
        speed = Automobile_Data.normalizeSpeed(speed)
        self.speed = speed
        data = {'action': '1', 'speed': float(speed)}
        self.pub.publish(String(data=json.dumps(data)))

    # ═══════════════════════════════════════════════════════════════════ #
    #  DASHBOARD / TELEMETRY PUBLISHERS                                   #
    # ═══════════════════════════════════════════════════════════════════ #

    def publish_closest_node(self, data: float = 0.0) -> None:
        self.pub_closest_node.publish(Float32(data=float(data)))

    def publish_next_event(self, data: str) -> None:
        self.pub_next_event.publish(String(data=str(data)))

    def publish_prev_event(self, data: str) -> None:
        self.pub_prev_event.publish(String(data=str(data)))

    def publish_current_state(self, data: str) -> None:
        self.pub_current_state.publish(String(data=str(data)))

    def publish_routines(self, data: str) -> None:
        self.pub_routines.publish(String(data=str(data)))

    def publish_arena_flag(self, data: bool) -> None:
        self.pub_arena.publish(Bool(data=bool(data)))

    def publish_led_control(self, data: bool) -> None:
        self.pub_led.publish(Bool(data=bool(data)))

    def publish_conditions(self, data: dict) -> None:
        msg = Conditions(
            can_overtake = bool(data['can_overtake']),
            highway      = bool(data['highway']),
            car_on_path  = bool(data['car_on_path']),
            rerouting    = bool(data['rerouting']),
            tunnel       = bool(data['tunnel']),
        )
        self.pub_conditions.publish(msg)

    def publish_localisation(self, x: float, y: float) -> None:
        msg = Localisation(pos_a=float(x), pos_b=float(y), timestamp=0.0, rot_a=0.0, rot_b=0.0)
        self.pub_localisation.publish(msg)
