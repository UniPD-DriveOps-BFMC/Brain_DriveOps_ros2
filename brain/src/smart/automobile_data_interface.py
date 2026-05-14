#!/usr/bin/python3

import names_and_constants as nac

# Functional libraries
# import rospy
import numpy as np
from automobile_ekf import AutomobileEKF
import time
from collections import deque

YAW_GLOBAL_OFFSET = np.deg2rad(0)

START_X = 0.2
START_Y = 14.8

GPS_DELAY = 0.45  # [s] delay for gps message to arrive
ENCODER_POS_FREQ = 100.0  # [Hz] frequency of encoder position messages
GPS_FREQ = 10.0  # [Hz] frequency of gps messages
BUFFER_PAST_MEASUREMENTS_LENGTH = int(round(GPS_DELAY * ENCODER_POS_FREQ))

# Vehicle driving parameters
MIN_SPEED = -0.3             # [m/s]     minimum speed
MAX_SPEED = 2.5              # [m/s]     maximum speed
MAX_ACCEL = 5.5              # [m/ss]    maximum accel
MAX_STEER = 28.0             # [deg]     maximum steering angle

# Vehicle parameters
LENGTH = 0.45  			     # [m]       car body length
WIDTH = 0.18   			     # [m]       car body width
BACKTOWHEEL = 0.10  	     # [m]       distance of the wheel and the car body
WHEEL_LEN = 0.03  		     # [m]       wheel raduis
WHEEL_WIDTH = 0.03  	     # [m]       wheel thickness
WB = 0.26  			         # [m]       wheelbase

# Camera parameters — Luxonis OAK-D Pro (IMX378 RGB, 1280×960)
FRAME_WIDTH  = 1280                      # [pix]  RGB stream width
FRAME_HEIGHT = 960                       # [pix]  RGB stream height
# extrinsics: position and orientation of camera wrt car frame
# TODO: update CAM_Z and CAM_PITCH after running calibration/step2_extrinsics.py
CAM_X     = 0.0                          # [m]
CAM_Y     = 0.0                          # [m]
CAM_Z     = 0.2                          # [m]   lens height above ground reference
CAM_ROLL  = 0.0                          # [rad]
CAM_PITCH = np.deg2rad(20)               # [rad] downward tilt
CAM_YAW   = 0.0                          # [rad]
# intrinsics: from factory calibration (step1_intrinsics.py)
CAM_FOV = np.deg2rad(63.8326)            # [rad] HFOV
CAM_FX  = 1027.5515                      # [px]
CAM_FY  = 1027.5515                      # [px]
CAM_CX  = 634.0629                       # [px]
CAM_CY  = 454.3752                       # [px]
CAM_K = np.array([[CAM_FX, 0.0,    CAM_CX],
                  [0.0,    CAM_FY, CAM_CY],
                  [0.0,    0.0,    1.0   ]])
# Estimator parameters
EST_INIT_X = 3.0               # [m]
EST_INIT_Y = 3.0               # [m]
EST_INIT_YAW = 0.0             # [rad]


EKF_STEPS_BEFORE_TRUST = 10  # 10 is fine, 15 is safe


class Automobile_Data():
    def __init__(self,
                 trig_control=True,
                 trig_bno=False,
                 trig_enc=False,
                 trig_sonar=False,
                 trig_cam=False,
                 trig_gps=False,
                 trig_lidar=False, 
                 trig_tof=False
                 ) -> None:
        """Manage flow of data with the car

        :param trig_control: trigger on commands, defaults to True
        :type trig_control: bool, optional

        :param trig_cam: trigger on camera, defaults to False
        :type trig_cam: bool, optional

        :param trig_gps: trigger on GPS, defaults to False
        :type trig_gps: bool, optional

        :param trig_bno: trigger on IMU, defaults to False
        :type trig_bno: bool, optional

        :param trig_stateEst: trigger on state estimation, defaults to False
        :type trig_stateEst: bool, optional

        :param trig_enc: trigger on encoder, defaults to False
        :type trig_enc: bool, optional

        :param speed_buff_len: length of the speed buffer
        :type speed_buff_len: int, optional
        """

        # CAR POSITION
        self.x_true = START_X  # [m] true:x coordinate (used in sim and SPARCS)
        self.y_true = START_Y  # [m] true:y coordinate (used in sim and SPARCS)
        self.x = 0.0           # [m] GPS:x global coordinate
        self.y = 0.0           # [m] GPS:y global coordinate
        #self.closest_node = [0.0, 0.0]   
        # IMU
        self.yaw_offset = YAW_GLOBAL_OFFSET  # [rad]   yaw offset
        self.roll = 0.0                      # [rad]   roll angle
        self.roll_deg = 0.0                  # [deg]   roll angle
        self.pitch = 0.0                     # [rad]   pitch angle
        self.pitch_deg = 0.0                 # [deg]   pitch angle
        self.yaw = 0.0                       # [rad]   yaw angle
        self.yaw_deg = 0.0                   # [deg]   yaw angle
        self.yaw_true = 0.0                  # [deg]   true yaw angle
        self.accel_x = 0.0                   # [m/ss]  accelx angle
        self.accel_y = 0.0                   # [m/ss]  accely angle
        self.accel_z = 0.0                   # [m/ss]  accelz angle
        self.gyrox = 0.0                     # [rad/s] gyrox angular vel
        self.gyroy = 0.0                     # [rad/s] gyroy angular vel
        self.gyroz = 0.0                     # [rad/s] gyroz angular vel
        self.filtered_yaw = 0.0              # [deg]   filtered yaw angles
        self.yaw_random_start = 0.0
        self.IMU_yaw = 0.0
        # ENCODER
        self.encoder_velocity = 0.0           # [m/s] ENC:speed measure
        self.filtered_encoder_velocity = 0.0  # [m/s] ENC:filtered speed
        self.encoder_distance = 0.0           # [m] total abs dist -never reset
        self.prev_dist = 0.0                  # [m] previous distance
        self.prev_gps_dist = 0.0
        # CAR POSE ESTIMATION
        self.x_est = 0.0                # [m] EST:x EKF estimated global coord
        self.y_est = 0.0                # [m] EST:y EKF estimated global coord
        self.yaw_est = self.yaw_offset  # [rad] EST:yaw EKF estimated
        self.gps_cnt = 0
        self.trust_gps = True  # [bool] EST:var is true if the EKF trusts GPS
        self.buffer_gps_positions_still_car = []
        # LOCAL POSITION
        self.x_loc = 0.0       # [m] local:x local coord
        self.y_loc = 0.0       # [m] local:y local coord
        self.yaw_loc = 0.0     # [rad] local:yaw local
        self.yaw_loc_o = 0.0   # [rad] local:yaw origin wrt to global IMU yaw
        self.dist_loc = 0.0    # [m] local:abs dist, len of local traj
        self.dist_loc_o = 0.0  # [m] local:abs dist origin, wrt global enc dist
        self.last_gps_sample_time = time.time()
        self.new_gps_sample_arrived = True
        # SONARs
        self.sonar_distance = 3.0  # [m] SONAR: unfilt dist from front
        self.filtered_sonar_distance = 3.0  # [m] SONAR: filt dist from front
        self.right_sonar_distance = 3.0  # [m] SONAR: unfilt dist from lat
        self.filtered_right_sonar_distance = 3.0  # [m] SONAR:filt dist lat
        self.left_sonar_distance = 3.0  # [m] SONAR: unfilt dist from lat
        self.filtered_left_sonar_distance = 3.0  # [m] SONAR:filt dist lat
        # TOFs
        self.center_tof_distance = 0.21  # [m] TOF: unfilt dist from lat
        self.filtered_center_tof_distance = 0.21  # [m] TOF:filt dist lat
        self.left_tof_distance = 0.21 # [m] TOF: unfilt dist from lat
        self.filtered_left_tof_distance = 0.21  # [m] TOF:filt dist lat
        # ESP32 CAMERA
        self.obstacle = 0.0            # ESP32: confidence level for obstacle classification
        self.filtered_obstacle = 0.0   # ESP32: filtered confidence level for obstacle classification
        self.sign = 0.0                # ESP32: confidence level for sign classification
        self.filtered_sign = 0.0       # ESP32: filtered confidence level for sign classification  
        # CAMERA
        # [ndarray] CAM:image of the camera
        self.frame = np.zeros((FRAME_WIDTH, FRAME_HEIGHT, 3), np.uint8)
        # CONTROL ACTION
        self.speed = 0.0            # [m/s]     MOTOR:speed
        self.steer = 0.0            # [rad]     SERVO:steering angle
        # CONSTANT PARAMETERS
        self.MIN_SPEED = MIN_SPEED        # [m/s] minimum speed
        self.MAX_SPEED = MAX_SPEED        # [m/s] maximum speed
        self.MAX_STEER = MAX_STEER        # [deg] maximum steering angle
        # VEHICLE PARAMETERS
        self.LENGTH = LENGTH			 # [m] car body length
        self.WIDTH = WIDTH  			 # [m] car body width
        self.BACKTOWHEEL = BACKTOWHEEL	 # [m] dist wheel -- car body
        self.WHEEL_LEN = WHEEL_LEN  	 # [m] wheel raduis
        self.WHEEL_WIDTH = WHEEL_WIDTH   # [m] wheel thickness
        self.WB = WB  			         # [m] wheelbase
        # CAMERA PARAMETERS
        self.FRAME_WIDTH = FRAME_WIDTH    # [pix]     frame width
        self.FRAME_HEIGHT = FRAME_HEIGHT  # [pix]     frame height
        self.CAM_X = CAM_X
        self.CAM_Y = CAM_Y
        self.CAM_Z = CAM_Z
        self.CAM_ROLL = CAM_ROLL
        self.CAM_PITCH = CAM_PITCH
        self.CAM_YAW = CAM_YAW
        self.CAM_FOV = CAM_FOV
        self.CAM_K = CAM_K
        # ESTIMATION PARAMETERS
        self.last_estimation_callback_time = None
        self.est_init_state = np.array([EST_INIT_X, EST_INIT_Y]).reshape(-1, 1)
        self.ekf = AutomobileEKF(x0=self.est_init_state, WB=self.WB)

        self.past_encoder_distances = deque(maxlen=BUFFER_PAST_MEASUREMENTS_LENGTH)
        self.past_yaws = deque(maxlen=BUFFER_PAST_MEASUREMENTS_LENGTH)
        self.yaws_between_updates = deque(maxlen=int(round(ENCODER_POS_FREQ/GPS_FREQ)))

        self.STARTED_WITH_IMU = False

        # I/O interface
        # rospy.init_node('automobile_data', anonymous=False)

        # SUBSCRIBERS AND PUBLISHERS
        # to be implemented in the specific class
        # they need to refer to the specific callbacks
        pass


    # COMMAND ACTIONS
    def drive(self, speed=0.0, angle=0.0) -> None:
        """Command a speed and steer angle to the car
        :param speed: [m/s] desired speed, defaults to 0.0
        :param angle: [deg] desired angle, defaults to 0.0
        """
        self.drive_speed(speed)
        self.drive_angle(angle)

        
    def update_rel_position(self) -> None:
        """Update relative pose of the car
        right-hand frame of reference with x aligned with the direction of
        motion
        """
        self.yaw_loc = self.yaw - self.yaw_loc_o
        curr_dist = self.encoder_distance

        # add curr_dist to distance buffer
        self.past_encoder_distances.append(curr_dist)
        # update yaw buffer
        if len(self.past_yaws) > BUFFER_PAST_MEASUREMENTS_LENGTH-1:
            self.yaws_between_updates.append(self.past_yaws.popleft())
        self.past_yaws.append(self.yaw)

        self.dist_loc = np.abs(curr_dist - self.dist_loc_o)
        signed_L = curr_dist - self.prev_dist
        L = np.abs(signed_L)
        dx = L * np.cos(self.yaw_loc)
        dy = L * np.sin(self.yaw_loc)
        self.x_loc += dx
        self.y_loc += dy
        self.prev_dist = curr_dist
        # update gps estimation filler
        if self.new_gps_sample_arrived:
            self.last_gps_sample_time = time.time()
            self.new_gps_sample_arrived = False


    def update_estimated_state(self):  # <++> !!!!!!!!!!!
        return
        ''' Updates estimated state according to EKF '''
        # Sampling time of the estimator
        callback_time = time.time()
        if self.last_estimation_callback_time is None:
            self.last_estimation_callback_time = callback_time
            return
        else:
            DT = callback_time - self.last_estimation_callback_time
            self.last_estimation_callback_time = callback_time

        if DT > 0 and \
                len(self.past_encoder_distances) > \
                BUFFER_PAST_MEASUREMENTS_LENGTH-1:
            self.new_gps_sample_arrived = True
            # calculate velocity using the encoder for maximum precision
            # note: we are using the encoder distance delayed of GPS delay
            # get the last element of the buffer
            # delayed distance
            curr_gps_dist = self.past_encoder_distances.popleft()
            # delayed yaw
            curr_yaw = np.median(np.array(self.yaws_between_updates))
            dist = curr_gps_dist - self.prev_gps_dist
            velocity = dist / DT
            self.prev_gps_dist = curr_gps_dist

            # INPUT: [SPEED, STEER]
            u0 = velocity
            u1 = curr_yaw
            u = np.array([u0, u1]).reshape(-1, 1)
            # OUTPUT: [GPS:x, GPS:y, IMU:yaw]
            zx = self.x
            zy = self.y
            z = np.array([zx, zy]).reshape(-1, 1)
            # PREDICT and UPDATE STEPS
            print("CALCULATING EKF ESTIMATION")
            x_est, y_est = self.ekf.estimate_state(sampling_time=DT, input=u,
                                                   output=z)
            # check if x_est and y_est are valid
            p_ekf = np.array([x_est, y_est])  # estimate of the kalmann filter
            p_gps = np.array([self.x, self.y])       # current gps position
            # distances
            ekf_gps_diff = np.linalg.norm(p_ekf - p_gps)
            # checks
            if ekf_gps_diff > 0.9:  # 0.5:
                # -> use gps
                self.ekf.x[0, 0] = p_gps[0]
                self.ekf.x[1, 0] = p_gps[1]
                self.trust_gps = False
                self.gps_cnt = 0
            else:  # fairly accurate
                self.gps_cnt += 1
                if self.gps_cnt > EKF_STEPS_BEFORE_TRUST:
                    self.trust_gps = True
            # update the estimated state, if gps is trusted
            if self.trust_gps:
                self.x_est = x_est  # + tot_delta_x
                self.y_est = y_est  # + tot_delta_y\

# USED ONLY IN 2024 
    def decide_yaw_start(self) -> None:
        heading = self.yaw_true
        #heading = np.rad2deg(self.yaw) % 360
        print(f"Heading: {heading}")
        if heading is None:
            print("Heading not available!")
            return
        if 45 <= heading < 135:
            self.yaw_random_start = np.deg2rad(90)
            print("Orientation: North")
        elif 135 <= heading < 225:
            self.yaw_random_start = np.deg2rad(180)          
            print("Orientation: West")
        elif 225 <= heading < 315:
            self.yaw_random_start = np.deg2rad(-90)
            print("Orientation: South")
        else:
            self.yaw_random_start = np.deg2rad(0)    
            print("Orientation: Est")


    def reset_rel_pose(self) -> None:
        """Set origin of the local frame to the actual pose
        """
        self.x_loc = 0.0
        self.y_loc = 0.0
        self.yaw_loc_o = self.yaw
        self.prev_yaw = self.yaw
        self.yaw_loc = 0.0
        self.prev_yaw_loc = 0.0
        self.dist_loc = 0.0
        self.dist_loc_o = self.encoder_distance

    # STATIC METHODS
    def normalizeSpeed(val):
        """Clamp speed value

        :param val: speed to clamp
        :type val: double
        :return: clamped speed value
        :rtype: double
        """
        if val < MIN_SPEED:
            val = MIN_SPEED
        elif val > MAX_SPEED:
            val = MAX_SPEED
        return val

    def normalizeSteer(val):
        """Clamp steer value

        :param val: steer to clamp
        :type val: double
        :return: clamped steer value
        :rtype: double
        """
        if val < -MAX_STEER:
            val = -MAX_STEER
        elif val > MAX_STEER:
            val = MAX_STEER
        return val

    def __str__(self):
        description = '''
{:#^65s}
(x,y):\t\t\t\t({:.2f},{:.2f})\t\t[m]
{:#^65s}
(x_est,y_est,yaw_est):\t\t({:.2f},{:.2f},{:.2f})\t[m,m,deg]
{:#^65s}
(x_loc,y_loc,yaw_loc):\t\t({:.2f},{:.2f},{:.2f})\t[m,m,deg]
dist_loc:\t\t\t{:.2f}\t\t\t[m]
{:#^65s}
roll, pitch, yaw:\t\t{:.2f}, {:.2f}, {:.2f}\t[deg]
ax, ay, az:\t\t\t{:.2f}, {:.2f}, {:.2f}\t[m/s^2]
wx, wy, wz:\t\t\t{:.2f}, {:.2f}, {:.2f}\t[rad/s]
{:#^65s}
encoder_distance:\t\t{:.3f}\t\t\t[m]
encoder_velocity (filtered):\t{:.2f} ({:.2f})\t\t[m/s]
{:#^65s}
sonar_distance (filtered):\t{:.3f} ({:.3f})\t\t[m]
'''
        return description.format(' POSITION ',
                                  self.x, self.y,
                                  ' ESTIMATION ',
                                  self.x_est, self.y_est,
                                  np.rad2deg(self.yaw_est),
                                  ' LOCAL POSITION ',
                                  self.x_loc, self.y_loc,
                                  np.rad2deg(self.yaw_loc), self.dist_loc,
                                  ' IMU ',
                                  self.roll_deg, self.pitch_deg,
                                  self.yaw_deg, self.accel_x, self.accel_y,
                                  self.accel_z, self.gyrox, self.gyroy,
                                  self.gyroz,
                                  ' ENCODER ',
                                  self.encoder_distance,
                                  self.encoder_velocity,
                                  self.filtered_encoder_velocity,
                                  ' SONAR ',
                                  self.sonar_distance,
                                  self.filtered_sonar_distance)
