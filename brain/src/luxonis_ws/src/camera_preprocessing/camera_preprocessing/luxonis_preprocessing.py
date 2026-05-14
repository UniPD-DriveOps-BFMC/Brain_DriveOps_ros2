import os
import json
import numpy as np
import cv2
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge

from camera_preprocessing.controller3 import Controller

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODEL_PATH = os.path.join(_PACKAGE_DIR, 'lane_keeper_small.onnx')


class LaneInferenceNode(Node):

    def __init__(self):
        super().__init__('lane_inference_node')

        self.declare_parameter('model_path',    _DEFAULT_MODEL_PATH)
        self.declare_parameter('input_topic',   '/oak/rgb/image_raw')
        self.declare_parameter('command_topic', '/automobile/command')
        self.declare_parameter('desired_speed', 0.007)
        self.declare_parameter('debug_view',    True)
        self.declare_parameter('faster',        False)
        self.declare_parameter('SIMULATOR_FLAG', True)

        model_path         = self.get_parameter('model_path').value
        input_topic        = self.get_parameter('input_topic').value
        command_topic      = self.get_parameter('command_topic').value
        self.desired_speed = self.get_parameter('desired_speed').value
        self.debug_view    = self.get_parameter('debug_view').value
        self.faster        = self.get_parameter('faster').value
        self.simulator_flag = self.get_parameter('SIMULATOR_FLAG').value

        # On the real car the STM firmware treats the speed field as a
        # normalized motor command (~PWM duty), not m/s. 0.3 is way too
        # high; 0.006 is right above the deadband. Override the default
        # only if the user did NOT pass desired_speed explicitly.
        if not self.simulator_flag and \
           self.get_parameter('desired_speed').value == 0.3:
            self.desired_speed = 0.006
            self.get_logger().info('Real-car mode: desired_speed adjusted to 0.006')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image, input_topic, self.image_callback, 10)

        # Publishers depend on whether we are talking to the simulator
        # (single String topic, JSON payload) or to the real STM
        # firmware (separate Float32 topics for steer/speed/stop).
        if self.simulator_flag:
            self.cmd_pub = self.create_publisher(String, command_topic, 10)
        else:
            self.steer_pub = self.create_publisher(
                Float32, '/automobile/command/steer', 10)
            self.speed_pub = self.create_publisher(
                Float32, '/automobile/command/speed', 10)
            self.stop_pub = self.create_publisher(
                Float32, '/automobile/command/stop', 10)
        
        if not os.path.isfile(model_path):
            self.get_logger().error(f'Modello ONNX non trovato: {model_path}')
            raise FileNotFoundError(f'ONNX model not found: {model_path}')

        self.session    = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        self.controller = Controller(
            k1=1.0, k2=1.0, k3=1.0, k3D=0.08,
            dist_point_ahead=0.35, ff=1.0,
        )

        self.get_logger().info(
            f'LaneInferenceNode avviato\n'
            f'  model   → {model_path}\n'
            f'  input   → {input_topic}\n'
            f'  command → {command_topic}\n'
            f'  speed   → {self.desired_speed}\n'
            f'  debug   → {self.debug_view}\n'
            f'  faster  → {self.faster}'
        )

    def preprocess_frame(self, cv_image: np.ndarray):
        """
        Preprocessing identico a detect_lane():
        gray → crop bottom 2/3 → resize 64x64 → Canny → blur → resize 32x32
        Restituisce (blob per ONNX, frame_debug per visualizzazione).
        """
        IMG_SIZE = (32, 32)

        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        img = img[int(img.shape[0] / 3):int(img.shape[0] * 0.85), :]               # taglia top 1/3
        img = cv2.resize(img, (2 * IMG_SIZE[0], 2 * IMG_SIZE[1]))
        img = cv2.Canny(img, 100, 200)
        img = cv2.blur(img, (3, 3))
        img = cv2.resize(img, IMG_SIZE)

        # Frame debug: ingrandito 10x per renderlo visibile
        debug_frame = cv2.resize(img, (320, 320), interpolation=cv2.INTER_NEAREST)

        if self.faster:
            # (1, 1, 32, 32) — mirrors blobFromImage(scale=1.0)
            blob = img.astype(np.float32)[np.newaxis, np.newaxis, ...]
        else:
            img_flip = cv2.flip(img, 1)
            # (2, 1, 32, 32) — mirrors blobFromImages(scale=1.0)
            blob = np.stack((img, img_flip), axis=0).astype(np.float32)[:, np.newaxis, ...]

        return blob, debug_frame

    def publish_command(self, action: str, value: float) -> None:
        """
        action '1' -> speed,  action '2' -> steerAngle.

        SIMULATOR_FLAG=True  -> JSON String on /automobile/command
        SIMULATOR_FLAG=False -> Float32 on /automobile/command/{steer,speed}
        """
        if self.simulator_flag:
            if action == '1':
                payload = json.dumps({'action': '1', 'speed': float(value)})
            else:
                payload = json.dumps({'action': '2', 'steerAngle': float(value)})
            msg = String()
            msg.data = payload
            self.cmd_pub.publish(msg)
        else:
            msg = Float32()
            msg.data = float(value)
            if action == '1':
                self.speed_pub.publish(msg)
            else:
                self.steer_pub.publish(msg)

    def image_callback(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return

        blob, debug_frame = self.preprocess_frame(cv_image)


        # Mostra frame preprocessato (320x320 invece di 32x32)
        if self.debug_view:
            cv2.imshow('preprocessed 32x32 (ingrandito 10x)', debug_frame)
            cv2.imshow('raw', cv2.resize(cv_image, (320, 240)))
            cv2.waitKey(1)

        if self.faster:
            # blob shape: (1,1,32,32) — single inference
            out = -self.session.run(None, {self.input_name: blob})[0]
            e2 = float(out[0][0])
            e3 = float(out[0][1])
        else:
            # blob shape: (2,1,32,32) — split and run twice
            blob_orig = blob[0:1, ...]   # (1,1,32,32)
            blob_flip = blob[1:2, ...]   # (1,1,32,32)

            out_orig = -self.session.run(None, {self.input_name: blob_orig})[0]
            out_flip = -self.session.run(None, {self.input_name: blob_flip})[0]

            e2 = (float(out_orig[0][0]) - float(out_flip[0][0])) / 2
            e3 = (float(out_orig[0][1]) - float(out_flip[0][1])) / 2

        # Controller
        _, angle_rad = self.controller.get_control(
            e2=e2, e3=e3, curv=0.0,
            desired_speed=self.desired_speed,
        )
        #angle_deg = float(np.clip(-np.rad2deg(angle_rad), -23.0, 23.0))  # segno invertito
        angle_deg = float(-np.rad2deg(angle_rad))  # segno invertito

        # Pubblica
        self.publish_command('2', angle_deg)
        self.publish_command('1', self.desired_speed)

        self.get_logger().info(
            f'e2={e2:.4f}  e3={e3:.4f}  steer={angle_deg:.2f}°'
        )


    def emergency_stop(self) -> None:
        """
        Send a stop command on shutdown so the motor disarms cleanly.
        Real-car only. Publishes 0.0 on /automobile/command/stop and
        also on /automobile/command/speed as a belt-and-braces fallback
        in case the firmware does not action /stop.
        """
        if self.simulator_flag:
            return
        try:
            zero = Float32(); zero.data = 0.0
            self.stop_pub.publish(zero)
            self.speed_pub.publish(zero)
            self.get_logger().info('Emergency stop sent on /automobile/command/stop')
        except Exception as e:
            self.get_logger().error(f'Emergency stop failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = LaneInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.emergency_stop()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
