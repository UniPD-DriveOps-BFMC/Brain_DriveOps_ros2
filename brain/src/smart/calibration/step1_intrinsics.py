#!/usr/bin/env python3
"""
STEP 1 — Read factory intrinsics from the OAK-D Pro.

The camera stores per-unit calibration values burned in at the factory.
This script reads them at the resolution we actually use (1280×960) and
prints the numbers ready to paste into automobile_data_interface.py.

Usage:
    python3 calibration/step1_intrinsics.py

The camera must be plugged in via USB. No checkerboard needed.
"""

import sys
import numpy as np

try:
    import depthai as dai
except ImportError:
    sys.exit("depthai not found — run:  pip install depthai")

W, H = 1280, 960
SOCKET = dai.CameraBoardSocket.CAM_A   # RGB camera


def main():
    print("Connecting to OAK-D Pro …")
    with dai.Device() as device:
        calib = device.readCalibration()

        K_raw = calib.getCameraIntrinsics(SOCKET, W, H)   # 3×3 list of lists
        D_raw = calib.getDistortionCoefficients(SOCKET)    # up to 14 coefficients

    K = np.array(K_raw)
    D = np.array(D_raw)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    hfov_deg = np.degrees(2 * np.arctan2(W / 2.0, fx))
    vfov_deg = np.degrees(2 * np.arctan2(H / 2.0, fy))

    print()
    print("=" * 60)
    print(f"  Intrinsic matrix K  ({W}×{H})")
    print("=" * 60)
    print(f"  fx = {fx:.4f} px")
    print(f"  fy = {fy:.4f} px")
    print(f"  cx = {cx:.4f} px")
    print(f"  cy = {cy:.4f} px")
    print(f"  HFOV = {hfov_deg:.2f}°   VFOV = {vfov_deg:.2f}°")
    print()
    print(f"  Distortion D = {np.round(D, 6).tolist()}")
    print()
    print("=" * 60)
    print("  Paste this block into automobile_data_interface.py")
    print("  (replace the current CAM_FX / CAM_FY / CAM_CX / CAM_CY lines)")
    print("=" * 60)
    print(f"""
CAM_FOV = np.deg2rad({hfov_deg:.4f})    # [rad] HFOV from factory calibration
CAM_FX  = {fx:.4f}                      # [px]
CAM_FY  = {fy:.4f}                      # [px]
CAM_CX  = {cx:.4f}                      # [px]
CAM_CY  = {cy:.4f}                      # [px]
CAM_K   = np.array([[CAM_FX, 0.0,    CAM_CX],
                    [0.0,    CAM_FY, CAM_CY],
                    [0.0,    0.0,    1.0   ]])
""")


if __name__ == "__main__":
    main()
