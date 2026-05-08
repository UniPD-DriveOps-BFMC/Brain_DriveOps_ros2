#!/usr/bin/python3
import names_and_constants as nac
import os
import signal
import cv2 as cv
# import rospy                # type: ignore #suppress warning
import rclpy, threading
import numpy as np
from time import sleep, time
# from unix_socket_camera import UnixSocketCamera

import csv
from datetime import datetime


import json
with open('data/events_config.json', 'r') as file:
    events_config = json.load(file)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--random', action='store_true')
parser.add_argument('--rc', action='store_true')
parser.add_argument('--sim', action='store_true')
parser.add_argument('--show', action='store_true')
parser.add_argument('--arena', action='store_true')
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()
# If we don't call, these will be False
nac.RANDOM_START = args.random
nac.RC_MODE = args.rc
nac.SIMULATOR_FLAG = args.sim
nac.SHOW_IMGS = args.show
nac.ARENA = args.arena
nac.RESUME = args.resume

if not nac.SIMULATOR_FLAG:  # PI
    from automobile_data_pi import AutomobileDataPi
else: 
    from automobile_data_simulator import AutomobileDataSimulator
    
import helper_functions as hf
from path_planning4_mod import PathPlanning
from controller3 import Controller
from controllerSP import ControllerSpeed
from controllerAG import ControllerSpeed as ControllerBL
from detection import Detection
from brain import Brain
from rc_brain import RC_Brain
from environmental_data_simulator import EnvironmentalData





os.system('clear')
print('Main brain starting...')

track = cv.imread('data/2024_VerySmall.png')

# PARAMETERS
TARGET_FPS = 30.0 # target fps of the main loop


DESIRED_SPEED = 0.35  # [m/s]
SP_SPEED = 0.35  # [m/s]
CURVE_SPEED = 0.25  # [m/s]
BL_SP_SPEED = 0.8
BL_CURVE_SPEED = 0.35

# CONTROLLER
k1 = 0.0  # 0.0 gain error parallel to direction (speed)
k2 = 0.0  # 0.0 perpenddicular error gain
# pure paralllel k2 = 10 is very good at staying in the center of the lane
# k3 = 0.7  # 1.0 yaw error gain .8 with ff #BFMC_2023
k3_NL = 1.3 # for no_lane part #BFMC_2024
k3 = 1.0 # rest of the map #BFMC_2024
k3D = 0.08  # 0.08 derivative gain of yaw error
dist_point_ahead = 0.35  # distance of the point ahead in m

# dt_ahead = 0.5 # [s] how far into the future the curvature is estimated,
# feedforwarded to yaw controller
# ff_curvature = 1.0  # feedforward gain #BFMC_2023
ff_curvature = 1.0  # feedforward gain

x_orig = 0.0
y_orig = 0.0

# load camera with opencv
#if not nac.SIMULATOR_FLAG:  # <++>
#    cap = cv.VideoCapture(0)
#    cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
#    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
#    cap.set(cv.CAP_PROP_FPS, 30)
# if not nac.SIMULATOR_FLAG:
    # cap = UnixSocketCamera(socket_addr="/tmp/bfmc_camera_brain.sock", frame_size=(320, 240))


# stop the car with ctrl+c
def handler(signum, frame):
    print("Exiting ...")
    try:
        car.stop()
    except Exception:
        pass
    if nac.SIMULATOR_FLAG:
        os.system('ros2 service call /gazebo/pause_physics std_srvs/srv/Empty')
    cv.destroyAllWindows()
    sleep(.5)
    # Cleanly close the OAK-D Pro pipeline (option A).  Without this the
    # DepthAI daemon thread keeps the USB device locked and the next
    # main_brain.py launch fails with "Device already in use".
    try:
        car.destroy_node()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass
    exit(0)


if __name__ == '__main__':

    hf.create_frames(nac.SHOW_IMGS)
    rclpy.init()

    # init the car data
    if nac.SIMULATOR_FLAG:
        # os.system('rosservice call /gazebo/reset_simulation')
        # os.system('rosservice call gazebo/unpause_physics')
        car = AutomobileDataSimulator(trig_cam=True,
                                      trig_gps=True, 
                                      trig_bno=True, 
                                      trig_enc=True,
                                      trig_control=True,
                                      trig_sonar=True,
                                      trig_lidar=True,
                                      trig_tof=True)  # <++>
    else:
        # Option A: the brain opens the OAK-D Pro pipeline directly (no
        # depthai_ros_driver / no /oak/rgb/image_raw topic needed at
        # runtime).  The legacy ROS-image subscriber is therefore left
        # disabled (trig_cam=False).  If you want to switch to option B
        # (run depthai_ros_driver as a separate process and let the
        # brain subscribe to its topics), set trig_cam=True and
        # trig_cam_oak=False.
        car = AutomobileDataPi(trig_cam=False,
                               trig_cam_oak=True,
                               trig_gps=True,
                               trig_bno=True,
                               trig_enc=True,
                               trig_control=True,
                               trig_sonar=True,
                               trig_lidar=True,
                               trig_tof=True)
    sleep(1.5)

    # Start spinning immediately so ROS2 callbacks (encoder, GPS, IMU)
    # are processed during the heavy ONNX model loading that follows.
    spin_thread = threading.Thread(target=rclpy.spin, args=(car,), daemon=True)
    spin_thread.start()

    signal.signal(signal.SIGINT, handler)

    # init trajectory
    path_planner = PathPlanning(track)

    # init env
    env = EnvironmentalData(car=car, trig_v2v=True, trig_v2x=True, trig_semaphore=True) 
    # init controller
    controller = Controller(k1=k1, k2=k2, k3=k3, k3_NL=k3_NL, k3D=k3D,
                            dist_point_ahead=dist_point_ahead,
                            ff=ff_curvature)
    # The following controllers are not used in 2025
    controller_sp = ControllerSpeed(desired_speed=SP_SPEED,
                                    curve_speed=CURVE_SPEED)
    controller_ag = ControllerBL(straight_speed=BL_SP_SPEED,
                                 curve_speed=BL_CURVE_SPEED,
                                 lookahead=0.8)

    # initiliaze all the neural networks for detection and lane following
    detect = Detection()

    if nac.RC_MODE:
        brain = RC_Brain(car=car, controller=controller, detection=detect, max_speed=DESIRED_SPEED)
    else:
        brain = Brain(car=car, controller=controller, controller_sp=controller_sp,
                        controller_ag=controller_ag,
                        detection=detect, env=env, path_planner=path_planner,
                        desired_speed=DESIRED_SPEED)
    
    
    hf.show_track(track, car, nac.SHOW_IMGS)


    try:
        car.stop()
        fps_avg = 0.0
        fps_cnt = 0

        # Setup logging
        
        log_filename = f'logs/yaw_distance_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        log_file = open(log_filename, mode='w', newline='')
        log_writer = csv.writer(log_file)
        log_writer.writerow(['timestamp', 'encoder_distance','yaw_true'])  # CSV header
        

        while rclpy.ok():

            loop_start_time = time()
            # clear the screen
            print("\033c")

            # MAKE ME A LOG FILE HERE
            log_writer.writerow([time(), car.encoder_distance, car.yaw_true])
            #print("Car yaw: ", car.yaw_true)
            #print("Distance: ", car.encoder_distance)

            
            # <++++++++++++++++++++>
            #hf.show_car(track, car, nac.SHOW_IMGS)

            if not nac.SIMULATOR_FLAG:
                if car.frame is None:
                    print("Waiting for camera frame...")
                    continue
                
            brain.run()

            # DEBUG INFO
            # print(f'Lane detection time = {detect.avg_lane_detection_time:.1f} [ms]')
            # print(f'Sign detection time = {detect.avg_sign_detection_time:.1f} [ms]')
            # print(f'FPS = {fps_avg:.1f}, loop_cnt = {fps_cnt}, capped at {TARGET_FPS}')
            # print('VALIDATION: ', brain.start_node_validated)
            # print('PATH: ', brain.path_planner.route_graph.nodes)
            # print(f'POSITION: X_EST = {car.x_est}, Y_EST = {car.y_est}')
            # print(f'POSITION: X_DIST = {car.x_est - x_orig}, Y_DIST = {car.y_est - y_orig}')
            # if np.abs(x_orig - car.x_est) > 0.32 or np.abs(y_orig - car.y_est) > 0.32:
            #     car.drive_speed(0.0)
            #     car.drive_angle(0.0)
            #     raise

            #hf.show_camera(car, nac.SHOW_IMGS)

            if nac.SHOW_IMGS:
                if cv.waitKey(1) == 27:
                    cv.destroyAllWindows()
                    break

            loop_time = time() - loop_start_time
            fps_avg = (fps_avg * fps_cnt + 1.0 / loop_time) / (fps_cnt + 1)
            fps_cnt += 1
            if loop_time < 1.0 / TARGET_FPS:
                sleep(1.0 / TARGET_FPS - loop_time)

    except KeyboardInterrupt:
        print("Shutting down")
        car.stop()
        
        log_file.close()
        
        sleep(.5)
        cv.destroyAllWindows()
        car.destroy_node()
        rclpy.shutdown()
        exit(0)        
        pass
