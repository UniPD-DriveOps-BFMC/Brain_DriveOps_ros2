"""
collect_images.py — Acquisizione frame dalla camera OAK-D Pro
Usa la risoluzione nativa MAX del sensore IMX378 (4056x3040) per il full FOV,
poi salva a 1920x1080 (ridimensionata, non croppata).
"""

import depthai as dai
import cv2
import time
import argparse
import os
from pathlib import Path
from datetime import datetime

CLASS_NAMES = [
    'stop', 'parking', 'priority', 'roundabout',
    'one_way', 'no_entry', 'exit_highway',
    'entrance_highway', 'crosswalk',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class', dest='cls', required=True,
                        choices=CLASS_NAMES)
    parser.add_argument('--output',
                        default=os.path.expanduser('~/dataset_bfmc'))
    parser.add_argument('--auto-interval', type=float, default=0.5)
    parser.add_argument('--max', type=int, default=200)
    parser.add_argument('--save-width', type=int, default=1920,
                        help='Larghezza file salvato (default: 1920)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Risoluzione nativa massima del sensore → FULL FOV
    SENSOR_W, SENSOR_H = 4056, 3040

    # Dimensione file salvato (mantiene aspect ratio del sensore)
    save_w = args.save_width
    save_h = int(save_w * SENSOR_H / SENSOR_W)

    out_dir = Path(args.output) / args.cls
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob('*.jpg')))

    print(f"\n📁 Output: {out_dir}")
    print(f"📊 Esistenti: {existing}")
    print(f"🎯 Classe: {args.cls}")
    print(f"📐 Sensor FULL FOV: {SENSOR_W}x{SENSOR_H}")
    print(f"💾 Salvataggio:     {save_w}x{save_h} (downscale, no crop)\n")

    device   = dai.Device()
    pipeline = dai.Pipeline(device)

    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    # Richiedi la risoluzione massima del sensore → full FOV
    cam_out = cam.requestOutput((SENSOR_W, SENSOR_H),
                                dai.ImgFrame.Type.BGR888p)
    q = cam_out.createOutputQueue(maxSize=4, blocking=False)

    pipeline.start()

    window_name = f'OAK Dataset Collector — {args.cls}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    print("Controlli:")
    print("  SPACE → salva frame")
    print("  A     → toggle auto-save")
    print("  Q/ESC → esci\n")

    saved     = 0
    auto_save = False
    last_save = 0

    while True:
        frame_msg = q.tryGet()
        if frame_msg is None:
            time.sleep(0.01)
            continue

        # Frame a risoluzione nativa (full FOV del sensore)
        frame_full = frame_msg.getCvFrame()

        # Ridimensiona per il salvataggio (mantiene FOV, solo risoluzione minore)
        frame_save = cv2.resize(frame_full, (save_w, save_h))

        # Display ridimensionato per visualizzazione
        display = frame_save.copy()
        H, W = display.shape[:2]

        color = (0, 255, 0) if auto_save else (200, 200, 200)

        cv2.putText(display, f'Class: {args.cls}',
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        cv2.putText(display, f'Saved: {saved + existing}/{args.max}',
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display,
                    f'AUTO-SAVE ON ({args.auto_interval}s)' if auto_save
                    else 'SPACE=save  A=auto  Q=quit',
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, f'FOV: {SENSOR_W}x{SENSOR_H} (full)',
                    (20, H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Mirino centrale
        cv2.drawMarker(display, (W // 2, H // 2), (0, 255, 0),
                       cv2.MARKER_CROSS, 40, 2)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        now = time.time()

        save_now = False
        if key == ord(' '):
            save_now = True
        elif key in (ord('a'), ord('A')):
            auto_save = not auto_save
            print(f"Auto-save: {'ON' if auto_save else 'OFF'}")
        elif key in (ord('q'), ord('Q'), 27):
            break

        if auto_save and (now - last_save) >= args.auto_interval:
            save_now = True

        if save_now and saved < args.max:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = out_dir / f'{args.cls}_{ts}.jpg'
            cv2.imwrite(str(filename), frame_save)  # salva SENZA overlay
            saved += 1
            last_save = now
            print(f'✅ {filename.name}  ({saved + existing} totale)')

            if saved >= args.max:
                print(f'\n🎉 Raggiunti {args.max} frame')

    pipeline.stop()
    device.close()
    cv2.destroyAllWindows()

    print(f'\n📊 Riepilogo:')
    print(f'  Classe:          {args.cls}')
    print(f'  Nuovi frame:     {saved}')
    print(f'  Totale cartella: {saved + existing}')


if __name__ == '__main__':
    main()
