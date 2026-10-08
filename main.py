import numpy as np
import cv2
import time
import math

WIDTH, HEIGHT = 960, 540

PI = math.pi
PHI = (1.0 + math.sqrt(5.0)) / 2.0
DSTAR = math.log(PI) / math.log(PHI)
TIME_FLOW = PHI - 1.0

def main():
    cv2.namedWindow("Project Phi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Project Phi", WIDTH, HEIGHT)

    running = True
    playing = True
    sim_time = 0.0
    previous = time.perf_counter()

    while running:
        now = time.perf_counter()
        dt = min(now - previous, 0.1) # clamp dt to avoid large jumps when frame stalls
        previous = now
        if playing:
            sim_time += dt * TIME_FLOW

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # esc
            running = False
        elif key == 32:  # space
            playing = not playing

        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)   

        cv2.imshow("Project Phi", frame)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
        