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


# ─────────────────────────────────────────────────────────────────────────────
# Kalman filter for (e2, e3)
# State  : [e2, e3, e2_dot, e3_dot]  (position + velocity)
# Measure: [e2, e3]
# ─────────────────────────────────────────────────────────────────────────────
class LaneKalmanFilter:
    """
    Simple constant-velocity Kalman filter that tracks the lateral offset (e2)
    and heading error (e3) produced by the CNN.

    The filter serves two purposes:
      1. Smooth noisy CNN outputs.
      2. Provide a Mahalanobis-distance gate so that implausible jumps
         (caused by the CNN locking onto the wrong pair of lines) are
         rejected and the last valid prediction is used instead.
    """

    def __init__(self, dt: float = 1 / 30,
                 process_noise: float = 0.01,
                 measurement_noise: float = 0.05,
                 gate_threshold: float = 9.0):
        """
        Parameters
        ----------
        dt               : nominal time step (seconds) between frames
        process_noise    : Q diagonal variance — how much the true state can
                           drift per frame (increase for faster dynamics)
        measurement_noise: R diagonal variance — how noisy the CNN output is
        gate_threshold   : Mahalanobis² threshold above which a measurement
                           is considered an outlier (chi² dof=2, 99% ≈ 9.21)
        """
        self.dt = dt
        self.gate_threshold = gate_threshold

        # State: [e2, e3, e2_dot, e3_dot]
        self.x = np.zeros((4, 1), dtype=np.float64)

        # Transition matrix (constant velocity)
        self.F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=np.float64)

        # Measurement matrix (we observe e2, e3 only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process-noise covariance
        q = process_noise
        self.Q = np.diag([q, q, q * 10, q * 10])

        # Measurement-noise covariance
        r = measurement_noise
        self.R = np.diag([r, r])

        # Initial state covariance (large → trust first measurement)
        self.P = np.eye(4) * 1.0

        self.initialized = False

    # ------------------------------------------------------------------
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    # ------------------------------------------------------------------
    def update(self, z: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Run one Kalman update step.

        Returns
        -------
        accepted : True if z passed the Mahalanobis gate
        state    : best estimate of [e2, e3] (filtered if accepted,
                   predicted if rejected)
        """
        z = z.reshape(2, 1)

        if not self.initialized:
            self.x[0, 0] = z[0, 0]
            self.x[1, 0] = z[1, 0]
            self.initialized = True
            return True, z.flatten()

        # Innovation and its covariance
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R

        # Mahalanobis² gate
        maha2 = float(y.T @ np.linalg.inv(S) @ y)
        accepted = maha2 < self.gate_threshold

        if accepted:
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(4) - K @ self.H) @ self.P
        # else: keep predicted state, do NOT update P with bad data

        return accepted, self.x[:2].flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Hough-based lane validator / confidence scorer
# ─────────────────────────────────────────────────────────────────────────────
class HoughLaneValidator:
    """
    Runs a lightweight traditional-CV pipeline on the (already cropped)
    grayscale frame to:

      * detect line segments with Hough
      * classify them as left / right candidates by position and angle
      * score overall confidence (0.0 – 1.0)
      * build a binary lane mask that can be AND-ed with the CNN input so
        that spurious lines outside the expected lane area are suppressed

    The confidence score is used by the inference node to decide how much
    to trust the CNN output vs the Kalman prediction.
    """

    # Lines whose absolute angle (from vertical) exceeds this are horizontal
    # road markings / shadows — ignore them.
    MAX_ANGLE_FROM_VERTICAL_DEG = 45.0

    def __init__(self,
                 img_width: int = 640,
                 img_height: int = 320,
                 conf_low: float = 0.25,
                 conf_high: float = 0.65):
        self.img_width  = img_width
        self.img_height = img_height
        self.conf_low   = conf_low   # below → ignore CNN, use Kalman
        self.conf_high  = conf_high  # above → fully trust CNN
        self._last_left_x  = img_width * 0.25
        self._last_right_x = img_width * 0.75

    # ------------------------------------------------------------------
    def _angle_from_vertical(self, line) -> float:
        """Return the absolute angle in degrees from vertical for a segment."""
        x1, y1, x2, y2 = line[0]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if abs(dy) < 1e-3:
            return 90.0  # perfectly horizontal
        return abs(np.degrees(np.arctan2(abs(dx), abs(dy))))

    # ------------------------------------------------------------------
    def analyze(self, gray_crop: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Parameters
        ----------
        gray_crop : grayscale image already cropped to the same vertical
                    slice used by preprocess_frame (bottom 2/3 of frame).

        Returns
        -------
        confidence : float in [0, 1]
        lane_mask  : uint8 binary mask of the same size as gray_crop,
                     white where plausible lane lines were found.
        """
        h, w = gray_crop.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # ── edge detection ──────────────────────────────────────────────
        blurred = cv2.GaussianBlur(gray_crop, (5, 5), 0)
        edges   = cv2.Canny(blurred, 50, 150)

        # ── Hough lines (probabilistic) ─────────────────────────────────
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=max(h // 8, 20),
            maxLineGap=max(h // 6, 15),
        )

        if lines is None:
            return 0.0, mask

        left_xs, right_xs = [], []
        cx = w / 2

        for line in lines:
            angle = self._angle_from_vertical(line)
            if angle > self.MAX_ANGLE_FROM_VERTICAL_DEG:
                continue  # skip near-horizontal

            x1, y1, x2, y2 = line[0]
            mid_x = (x1 + x2) / 2.0

            # Draw the valid segment onto the mask
            cv2.line(mask, (x1, y1), (x2, y2), 255, 2)

            if mid_x < cx:
                left_xs.append(mid_x)
            else:
                right_xs.append(mid_x)

        # ── confidence scoring ──────────────────────────────────────────
        has_left  = len(left_xs)  > 0
        has_right = len(right_xs) > 0

        if has_left and has_right:
            # Both sides visible: check that their separation is plausible
            mean_left  = np.mean(left_xs)
            mean_right = np.mean(right_xs)
            separation = (mean_right - mean_left) / w  # normalised [0,1]
            # Ideal lane width is roughly 30–70 % of image width
            sep_score = 1.0 if 0.20 <= separation <= 0.80 else 0.4
            confidence = sep_score

            # Update memory for adaptive ROI
            self._last_left_x  = mean_left
            self._last_right_x = mean_right
        elif has_left or has_right:
            # Only one side — lower confidence
            confidence = 0.40
        else:
            confidence = 0.0

        return float(np.clip(confidence, 0.0, 1.0)), mask

    # ------------------------------------------------------------------
    def build_adaptive_roi_mask(self, shape: tuple, margin: float = 0.20) -> np.ndarray:
        """
        Build a trapezoidal mask centred on the last known lane position.
        Pixels outside the expected lane corridor are set to 0.

        Parameters
        ----------
        shape  : (height, width) of the image to mask
        margin : extra fraction of image width added on each side of the
                 expected lane — increase for more tolerance.
        """
        h, w = shape
        roi_mask = np.zeros((h, w), dtype=np.uint8)

        left_x  = max(0, int(self._last_left_x  - margin * w))
        right_x = min(w, int(self._last_right_x + margin * w))

        # Trapezoid: narrower at the top (vanishing point), wider at the bottom
        top_shrink = int((right_x - left_x) * 0.15)
        pts = np.array([
            [left_x  + top_shrink, 0],
            [right_x - top_shrink, 0],
            [right_x,              h],
            [left_x,               h],
        ], dtype=np.int32)
        cv2.fillPoly(roi_mask, [pts], 255)

        return roi_mask


# ─────────────────────────────────────────────────────────────────────────────
# Main inference node
# ─────────────────────────────────────────────────────────────────────────────
class LaneInferenceNode(Node):

    def __init__(self):
        super().__init__('lane_inference_node')

        # ── parameters ──────────────────────────────────────────────────
        self.declare_parameter('model_path',        _DEFAULT_MODEL_PATH)
        self.declare_parameter('input_topic',        '/oak/rgb/image_raw')
        self.declare_parameter('command_topic',      '/automobile/command')
        self.declare_parameter('desired_speed',      0.007)
        self.declare_parameter('debug_view',         True)
        self.declare_parameter('faster',             False)
        self.declare_parameter('SIMULATOR_FLAG',     True)
        # Kalman gate threshold (Mahalanobis² — chi² dof=2 at 99% ≈ 9.21)
        self.declare_parameter('kalman_gate',        9.21)
        # Max physical steering change per frame (degrees) — hard gate
        self.declare_parameter('max_steer_rate_deg', 8.0)
        # Whether to apply the Hough-based adaptive ROI mask
        self.declare_parameter('use_adaptive_roi',   True)

        model_path          = self.get_parameter('model_path').value
        input_topic         = self.get_parameter('input_topic').value
        command_topic       = self.get_parameter('command_topic').value
        self.desired_speed  = self.get_parameter('desired_speed').value
        self.debug_view     = self.get_parameter('debug_view').value
        self.faster         = self.get_parameter('faster').value
        self.simulator_flag = self.get_parameter('SIMULATOR_FLAG').value
        kalman_gate         = self.get_parameter('kalman_gate').value
        self.max_steer_rate = self.get_parameter('max_steer_rate_deg').value
        self.use_adaptive_roi = self.get_parameter('use_adaptive_roi').value

        if not self.simulator_flag and \
           self.get_parameter('desired_speed').value == 0.3:
            self.desired_speed = 0.006
            self.get_logger().info('Real-car mode: desired_speed adjusted to 0.006')

        # ── ROS I/O ─────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, input_topic, self.image_callback, 10)

        if self.simulator_flag:
            self.cmd_pub = self.create_publisher(String, command_topic, 10)
        else:
            self.steer_pub = self.create_publisher(
                Float32, '/automobile/command/steer', 10)
            self.speed_pub = self.create_publisher(
                Float32, '/automobile/command/speed', 10)
            self.stop_pub  = self.create_publisher(
                Float32, '/automobile/command/stop',  10)

        # ── ONNX model ──────────────────────────────────────────────────
        if not os.path.isfile(model_path):
            self.get_logger().error(f'Modello ONNX non trovato: {model_path}')
            raise FileNotFoundError(f'ONNX model not found: {model_path}')

        self.session    = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        # ── controller ──────────────────────────────────────────────────
        self.controller = Controller(
            k1=1.0, k2=1.0, k3=1.0, k3D=0.08,
            dist_point_ahead=0.35, ff=1.0,
        )

        # ── robustness components ────────────────────────────────────────
        self.kalman    = LaneKalmanFilter(gate_threshold=kalman_gate)
        self.validator = HoughLaneValidator()

        self._prev_angle_deg: float | None = None   # for rate-of-change gate
        self._frame_count: int = 0

        self.get_logger().info(
            f'LaneInferenceNode (robust) avviato\n'
            f'  model          → {model_path}\n'
            f'  input          → {input_topic}\n'
            f'  command        → {command_topic}\n'
            f'  speed          → {self.desired_speed}\n'
            f'  debug          → {self.debug_view}\n'
            f'  faster         → {self.faster}\n'
            f'  kalman_gate    → {kalman_gate}\n'
            f'  max_steer_rate → {self.max_steer_rate}°/frame\n'
            f'  adaptive_roi   → {self.use_adaptive_roi}'
        )

    # ──────────────────────────────────────────────────────────────────
    # Pre-processing
    # ──────────────────────────────────────────────────────────────────
    def preprocess_frame(self, cv_image: np.ndarray):
        """
        gray → crop bottom 2/3 → (optional adaptive-ROI mask) →
        resize 64x64 → Canny → blur → resize 32x32

        Returns
        -------
        blob        : numpy array ready for ONNX
        debug_frame : enlarged 320x320 uint8 for cv2.imshow
        gray_crop   : full-res cropped grayscale (for Hough validator)
        """
        IMG_SIZE = (32, 32)

        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # Crop: keep the bottom 2/3 up to 85 % of the frame height
        crop_top = int(img.shape[0] / 3)
        crop_bot = int(img.shape[0] * 0.85)
        gray_crop = img[crop_top:crop_bot, :]

        # ── adaptive ROI mask ────────────────────────────────────────────
        if self.use_adaptive_roi and self.kalman.initialized:
            roi_mask = self.validator.build_adaptive_roi_mask(gray_crop.shape)
            masked = cv2.bitwise_and(gray_crop, gray_crop, mask=roi_mask)
        else:
            masked = gray_crop

        # ── standard CNN pre-processing ──────────────────────────────────
        small = cv2.resize(masked, (2 * IMG_SIZE[0], 2 * IMG_SIZE[1]))
        edges = cv2.Canny(small, 100, 200)
        blurd = cv2.blur(edges, (3, 3))
        final = cv2.resize(blurd, IMG_SIZE)

        debug_frame = cv2.resize(final, (320, 320),
                                 interpolation=cv2.INTER_NEAREST)

        if self.faster:
            blob = final.astype(np.float32)[np.newaxis, np.newaxis, ...]
        else:
            flipped = cv2.flip(final, 1)
            blob = np.stack((final, flipped), axis=0) \
                       .astype(np.float32)[:, np.newaxis, ...]

        return blob, debug_frame, gray_crop

    # ──────────────────────────────────────────────────────────────────
    # Steering rate-of-change gate
    # ──────────────────────────────────────────────────────────────────
    def _rate_gate(self, angle_deg: float) -> float:
        """
        Clamp the steering output so it cannot change by more than
        `max_steer_rate` degrees in a single frame.  This prevents
        instantaneous lane switches from slamming the wheels.
        """
        if self._prev_angle_deg is None:
            return angle_deg
        delta = angle_deg - self._prev_angle_deg
        clamped = self._prev_angle_deg + np.clip(
            delta, -self.max_steer_rate, self.max_steer_rate)
        return float(clamped)

    # ──────────────────────────────────────────────────────────────────
    # Publishing
    # ──────────────────────────────────────────────────────────────────
    def publish_command(self, action: str, value: float) -> None:
        if self.simulator_flag:
            if action == '1':
                payload = json.dumps({'action': '1', 'speed': float(value)})
            else:
                payload = json.dumps({'action': '2', 'steerAngle': float(value)})
            msg = String(); msg.data = payload
            self.cmd_pub.publish(msg)
        else:
            msg = Float32(); msg.data = float(value)
            if action == '1':
                self.speed_pub.publish(msg)
            else:
                self.steer_pub.publish(msg)

    # ──────────────────────────────────────────────────────────────────
    # Main callback
    # ──────────────────────────────────────────────────────────────────
    def image_callback(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return

        self._frame_count += 1

        # ── 1. Pre-process ───────────────────────────────────────────────
        blob, debug_frame, gray_crop = self.preprocess_frame(cv_image)

        # ── 2. Hough validator (runs on full-res crop, not on 32x32) ─────
        confidence, hough_mask = self.validator.analyze(gray_crop)

        # ── 3. CNN inference ─────────────────────────────────────────────
        if self.faster:
            out  = -self.session.run(None, {self.input_name: blob})[0]
            e2_raw = float(out[0][0])
            e3_raw = float(out[0][1])
        else:
            blob_orig = blob[0:1, ...]
            blob_flip = blob[1:2, ...]
            out_orig  = -self.session.run(None, {self.input_name: blob_orig})[0]
            out_flip  = -self.session.run(None, {self.input_name: blob_flip})[0]
            e2_raw    = (float(out_orig[0][0]) - float(out_flip[0][0])) / 2
            e3_raw    = (float(out_orig[0][1]) - float(out_flip[0][1])) / 2

        # ── 4. Kalman predict + gate ─────────────────────────────────────
        self.kalman.predict()

        # Weight the measurement by the Hough confidence:
        #   confidence < conf_low  → skip the CNN output entirely (use prediction)
        #   confidence > conf_high → fully trust CNN
        #   in between             → blend CNN and Kalman prediction
        if confidence < self.validator.conf_low:
            # Very low confidence: trust only the Kalman prediction
            e2_used, e3_used = float(self.kalman.x[0, 0]), float(self.kalman.x[1, 0])
            accepted = False
            blend    = 0.0
        else:
            # Try to update the Kalman filter with the CNN reading
            accepted, filtered = self.kalman.update(
                np.array([e2_raw, e3_raw], dtype=np.float64))

            if accepted:
                if confidence > self.validator.conf_high:
                    blend = 1.0   # fully trust CNN (filtered)
                else:
                    # Linear blend in the mid-confidence range
                    blend = (confidence - self.validator.conf_low) / \
                            (self.validator.conf_high - self.validator.conf_low)
                pred_e2 = float(self.kalman.x[0, 0])
                pred_e3 = float(self.kalman.x[1, 0])
                e2_used = blend * filtered[0] + (1 - blend) * pred_e2
                e3_used = blend * filtered[1] + (1 - blend) * pred_e3
            else:
                # Mahalanobis gate rejected: fall back to Kalman prediction
                e2_used = float(self.kalman.x[0, 0])
                e3_used = float(self.kalman.x[1, 0])
                blend   = 0.0

        # ── 5. Controller ────────────────────────────────────────────────
        _, angle_rad = self.controller.get_control(
            e2=e2_used, e3=e3_used, curv=0.0,
            desired_speed=self.desired_speed,
        )
        angle_deg_raw = float(-np.rad2deg(angle_rad))

        # ── 6. Rate-of-change gate ───────────────────────────────────────
        angle_deg = self._rate_gate(angle_deg_raw)
        self._prev_angle_deg = angle_deg

        # ── 7. Publish ───────────────────────────────────────────────────
        self.publish_command('2', angle_deg)
        self.publish_command('1', self.desired_speed)

        # ── 8. Debug display ─────────────────────────────────────────────
        if self.debug_view:
            # Annotate the Hough mask
            hough_bgr = cv2.cvtColor(hough_mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(hough_bgr,
                        f'conf={confidence:.2f} blend={blend:.2f}',
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if accepted else (0, 0, 255), 1)
            cv2.imshow('hough validator', hough_bgr)
            cv2.imshow('preprocessed 32x32 (x10)', debug_frame)
            cv2.imshow('raw', cv2.resize(cv_image, (320, 240)))
            cv2.waitKey(1)

        self.get_logger().info(
            f'[{self._frame_count:05d}] '
            f'e2_raw={e2_raw:.4f}  e3_raw={e3_raw:.4f}  '
            f'e2={e2_used:.4f}  e3={e3_used:.4f}  '
            f'conf={confidence:.2f}  blend={blend:.2f}  '
            f'gate={"OK" if accepted else "REJECTED"}  '
            f'steer={angle_deg:.2f}° (raw={angle_deg_raw:.2f}°)'
        )

    # ──────────────────────────────────────────────────────────────────
    def emergency_stop(self) -> None:
        if self.simulator_flag:
            return
        try:
            zero = Float32(); zero.data = 0.0
            self.stop_pub.publish(zero)
            self.speed_pub.publish(zero)
            self.get_logger().info('Emergency stop sent.')
        except Exception as e:
            self.get_logger().error(f'Emergency stop failed: {e}')


# ─────────────────────────────────────────────────────────────────────────────
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
