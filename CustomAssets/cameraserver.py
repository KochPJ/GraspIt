import cv2
import time
from pathlib import Path

from utils import mask_images

def main():
    device = "/dev/video0"
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)         
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError(f"failed to open camera: {device}")
    
    outdir = Path("input_images")
    outdir.mkdir(exist_ok=True)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("failed to grab frame")
            break

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == 32:
            filename = outdir / f"capture_{idx}.png"
            cv2.imwrite(str(filename), frame)
            idx += 1
    
    cap.release()
    cv2.destroyAllWindows()

    mask_images()

if __name__ == "__main__":
    main()