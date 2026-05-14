#!/usr/bin/env python3
"""
STEP 2 — Measure camera extrinsics (height + pitch) using the built-in
stereo depth sensor.  No ruler, no tape, no typing needed.

Physical setup:
  1. Park the car on flat ground.
  2. Plug in the OAK-D Pro via USB.
  3. Make sure nothing blocks the stereo cameras (remove any lens cap).

How it works:
  - A live RGB frame is shown.
  - LEFT-CLICK at least 4 ground points visible in the image.
    Click on the centre column of the image (left/right does not matter much).
    Use points at varied distances — close to far.
  - The depth sensor measures the 3D position of each click automatically.
  - Click "Fit & Exit" — CAM_Z and CAM_PITCH are computed from a
    ground-plane fit and printed ready to paste.

Update FY, CY, FX, CX below with the values from step1_intrinsics.py.
"""

import os
import sys
import time
import numpy as np
import tkinter as tk

try:
    import depthai as dai
except ImportError:
    sys.exit("depthai not found — run:  pip install depthai")

try:
    import cv2 as cv
except ImportError:
    sys.exit("opencv-python not found")

try:
    from PIL import Image, ImageTk, ImageDraw
except ImportError:
    sys.exit("Pillow not found — run:  pip install Pillow")

try:
    from scipy.optimize import curve_fit
except ImportError:
    sys.exit("scipy not found — run:  pip install scipy")

# ── UPDATE with step1_intrinsics.py values ────────────────────────────────────
FX = 1027.5515
FY = 1027.5515
CX = 634.0629
CY = 454.3752
# ─────────────────────────────────────────────────────────────────────────────

W, H = 1280, 960
DISPLAY_SCALE = 0.75
DW = int(W * DISPLAY_SCALE)
DH = int(H * DISPLAY_SCALE)

DEPTH_PATCH = 7    # median over NxN pixels around the click (reduces noise)
MIN_POINTS  = 4

# Physically measured tilt angle — used as initial guess for the fit
MEASURED_PITCH_DEG = 23.0


# ── Ground-plane model ───────────────────────────────────────────────────────

def ground_plane_model(vd_pairs, pitch, cam_z):
    """
    Correct ground-plane constraint derived from the depth sensor.

    The depth sensor returns Z_d (metres) along the optical axis, which is
    tilted downward by pitch angle P.  A pixel at row v on the ground satisfies:

        Z_d * (sin(P) + cos(P) * (v - cy) / fy) = CAM_Z

    Derivation: the camera-frame z-coordinate (up, pre-pitch) of a measured
    point is  z_car = -sin(P)*Z_d + cos(P)*(cy-v)/fy*Z_d.  For a ground
    point z_car = -CAM_Z, which rearranges to the formula above.

    Returns the residual (should be 0 for all ground points).
    """
    v_arr, z_d_arr = vd_pairs
    return z_d_arr * (np.sin(pitch) + np.cos(pitch) * (v_arr - CY) / FY) - cam_z


# ── DepthAI pipeline ──────────────────────────────────────────────────────────

def open_device():
    """
    Open the OAK-D and return (device, rgb_queue, depth_queue).
    Device is kept alive — call os._exit(0) at the end to avoid teardown crash.
    """
    print("Connecting to OAK-D Pro …")
    device   = dai.Device()
    pipeline = dai.Pipeline(device)

    # RGB
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgb_q = cam.requestOutput((W, H), dai.ImgFrame.Type.BGR888p)\
                .createOutputQueue(maxSize=4, blocking=False)

    # Stereo depth aligned to RGB
    mono_l = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    mono_r = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo  = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DETAIL)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(W, H)

    mono_l.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8).link(stereo.left)
    mono_r.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8).link(stereo.right)

    depth_q = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

    pipeline.start()
    print("Pipeline started.")
    return device, pipeline, rgb_q, depth_q  # pipeline must stay alive or queues close


def grab_frames(rgb_q, depth_q, warmup=10):
    """Poll until we have a synchronised RGB + depth pair."""
    rgb_frame   = None
    depth_frame = None
    deadline    = time.time() + 15.0
    count       = 0

    while (rgb_frame is None or depth_frame is None or count < warmup) \
          and time.time() < deadline:
        r = rgb_q.tryGet()
        d = depth_q.tryGet()
        if r is not None:
            rgb_frame = r.getCvFrame()
            count += 1
        if d is not None:
            depth_frame = d.getFrame()   # uint16, mm
        time.sleep(0.02)

    if rgb_frame is None or depth_frame is None:
        raise RuntimeError("Timed out waiting for frames. Check USB connection.")
    return rgb_frame, depth_frame


def sample_depth(depth_frame, u, v, patch=DEPTH_PATCH):
    """Median depth in a patch around (u, v), ignoring zeros."""
    h, w  = depth_frame.shape
    r     = patch // 2
    y0, y1 = max(0, v - r), min(h, v + r + 1)
    x0, x1 = max(0, u - r), min(w, u + r + 1)
    patch_vals = depth_frame[y0:y1, x0:x1].flatten()
    valid = patch_vals[patch_vals > 0]
    if len(valid) == 0:
        return 0
    return int(np.median(valid))


# ── GUI ───────────────────────────────────────────────────────────────────────

def run_gui(rgb_q, depth_q):
    """
    Show the live RGB frame.  User clicks ground points; depth is read
    automatically.  Returns list of (v_pixel_full_res, depth_metres).
    """
    # initial frame for display
    rgb_frame, _ = grab_frames(rgb_q, depth_q)
    cam_points = []   # list of (v_full_res, depth_m)

    pil_img = Image.fromarray(cv.cvtColor(rgb_frame, cv.COLOR_BGR2RGB))\
                   .resize((DW, DH), Image.LANCZOS)
    draw    = ImageDraw.Draw(pil_img)

    root   = tk.Tk()
    root.title("Step 2 — click ground points")
    root.resizable(False, False)

    info = tk.Label(root,
        text="LEFT-CLICK ground points (centre column, varied distances) — depth is read automatically",
        font=("Helvetica", 10), pady=4)
    info.pack()

    tk_img = ImageTk.PhotoImage(pil_img)
    canvas = tk.Canvas(root, width=DW, height=DH, cursor="crosshair")
    canvas.pack()
    img_id = canvas.create_image(0, 0, anchor="nw", image=tk_img)

    status = tk.Label(root, text="0 points collected (need ≥ 4)",
                      font=("Helvetica", 10), fg="gray40", pady=2)
    status.pack()

    def refresh():
        nonlocal tk_img
        tk_img = ImageTk.PhotoImage(pil_img)
        canvas.itemconfig(img_id, image=tk_img)
        canvas.image = tk_img

    def on_click(event):
        # map display coords → full-res pixel
        u = int(event.x / DISPLAY_SCALE)
        v = int(event.y / DISPLAY_SCALE)

        # grab a fresh depth frame
        _, depth_now = grab_frames(rgb_q, depth_q, warmup=0)
        d_mm = sample_depth(depth_now, u, v)

        if d_mm <= 0:
            status.config(text="No depth at that pixel — try another spot",
                          fg="red")
            return

        depth_m = d_mm / 1000.0
        cam_points.append((float(v), depth_m))
        n = len(cam_points)
        print(f"  Point {n}: pixel ({u}, {v})  depth {d_mm} mm")

        # draw marker on image
        ex, ey = event.x, event.y
        r = 7
        draw.ellipse([ex-r, ey-r, ex+r, ey+r], fill="lime", outline="lime")
        draw.text((ex+10, ey-10), f"{d_mm/1000:.2f}m", fill="lime")
        refresh()

        if n < MIN_POINTS:
            status.config(
                text=f"{n} point(s) — need at least {MIN_POINTS}",
                fg="gray40")
        else:
            status.config(
                text=f"{n} points — click more or press 'Fit & Exit'",
                fg="green")

    canvas.bind("<Button-1>", on_click)

    btn = tk.Button(root, text="Fit & Exit", command=root.destroy,
                    font=("Helvetica", 11, "bold"),
                    bg="#4caf50", fg="white", padx=12, pady=4)
    btn.pack(pady=6)

    root.mainloop()
    return cam_points


# ── Fit ───────────────────────────────────────────────────────────────────────

def fit_and_print(cam_points):
    if len(cam_points) < 2:
        print("Need at least 2 points. Exiting.")
        return

    v_arr   = np.array([p[0] for p in cam_points])
    z_d_arr = np.array([p[1] for p in cam_points])

    print(f"\nFitting ground plane with {len(cam_points)} points …\n")
    try:
        (pitch, cam_z), cov = curve_fit(
            ground_plane_model,
            (v_arr, z_d_arr),
            np.zeros(len(cam_points)),
            p0=[np.deg2rad(MEASURED_PITCH_DEG), 0.20],
            bounds=([np.deg2rad(MEASURED_PITCH_DEG - 10), 0.03],
                    [np.deg2rad(MEASURED_PITCH_DEG + 10), 0.80]))
    except RuntimeError as e:
        print(f"Fit failed: {e}")
        print("Try clicking points at more varied distances on flat ground.")
        return

    std       = np.sqrt(np.diag(cov))
    residuals = ground_plane_model((v_arr, z_d_arr), pitch, cam_z)
    rmse      = np.sqrt(np.mean(residuals**2))

    print("=" * 60)
    print("  Ground-plane fit result")
    print("=" * 60)
    print(f"  CAM_Z     = {cam_z:.4f} m    (±{std[1]*2:.4f} m, 2σ)")
    print(f"  CAM_PITCH = {np.degrees(pitch):.3f}°  ({pitch:.6f} rad)"
          f"    (±{np.degrees(std[0])*2:.3f}°, 2σ)")
    print(f"  RMSE      = {rmse*1000:.2f} mm")
    if rmse > 0.02:
        print("\n  WARNING: RMSE > 20 mm — ground may not be flat, or depth "
              "readings are noisy.")
        print("  Prefer points < 1 m from camera where stereo depth is most accurate.")
    print("\n  Per-point residuals:")
    for i, (v, zd, r) in enumerate(zip(v_arr, z_d_arr, residuals)):
        print(f"    {i+1}: row={v:.0f} px  depth={zd*1000:.0f} mm  residual={r*1000:+.1f} mm")
    print()
    print("=" * 60)
    print("  Paste into automobile_data_interface.py")
    print("=" * 60)
    print(f"""
CAM_Z     = {cam_z:.4f}                  # [m]  lens height above ground
CAM_PITCH = np.deg2rad({np.degrees(pitch):.3f})        # [rad] downward tilt
""")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _keep_alive = open_device()   # device+pipeline must stay in scope until os._exit
    *_, rgb_q, depth_q = _keep_alive
    cam_points = run_gui(rgb_q, depth_q)
    fit_and_print(cam_points)
    os._exit(0)   # bypass DepthAI teardown crash (known v3 bug)
