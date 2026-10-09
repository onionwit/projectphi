import numpy as np
import cv2
import time
import math

WIDTH, HEIGHT = 960, 540

PI = math.pi
TAU = 2.0 * PI
PHI = (1.0 + math.sqrt(5.0)) / 2.0
REQ = math.sqrt(PHI / PI)
DSTAR = math.log(PI) / math.log(PHI)
GA = TAU / PHI**2
LNPHI = math.log(PHI)
TIME_FLOW = PHI - 1.0

def numpy_preview(width, height, sim_time, zoom_c, theta, phi, pan):
    """Dependency-free fallback. Not a replacement for the OpenCL kernel;
    it just keeps the display pipeline testable without a GPU."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32) # create a grid of pixel coordinates
    nx = (2.0 * x - width) / max(height, 1) # normalize x coordinates to [-1, 1] range
    ny = (2.0 * y - height) / max(height, 1) # normalize y coordinates to [-1, 1] range
    rr = np.sqrt(nx**2 + ny**2) # compute the radial distance from the center

    angle = ( # compute the angle of each pixel in polar
        np.arctan2(ny, nx)
        + theta
    )  
    
    phase = ( # compute the phase for the simulation based on time and angle
        sim_time * TIME_FLOW 
        + PHI * np.sin(angle) 
        + np.cos(angle) 
        + theta
    ) 

    # compute the disk intensity based on radial distance
    disk = np.exp(-((rr - 0.34) / 0.075) ** 2) 

    # compute the ring intensity based on radial distance
    ring = np.exp(-((rr - 0.52) / 0.22) ** 2)

    # compute the core intensity based on radial distance
    core = np.exp(-(rr / 0.16) ** 4) 

    # compute the warp effect based on phase and radial distance
    warp = (
        np.sin(
            phase * 3.0 
            + rr * 16.0 
            + phi) * 0.5 
        + 0.5
    )

    # compute the hue based on angle, warp, time, and zoom
    hue = (angle / TAU + 0.15 * warp + sim_time * 0.015 + zoom_c * 0.02) % 1.0 

    # convert hue to RGB using HSV to RGB conversion
    h6 = hue * 6.0
    c = np.ones_like(h6)
    xcol = 1.0 - np.abs(h6 % 2.0 - 1.0)
    z = np.zeros_like(h6)
    sector = np.floor(h6).astype(np.int32) % 6
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    masks = [sector == i for i in range(6)]
    rgb[masks[0]] = np.stack([c[masks[0]], xcol[masks[0]], z[masks[0]]], axis=1)
    rgb[masks[1]] = np.stack([xcol[masks[1]], c[masks[1]], z[masks[1]]], axis=1)
    rgb[masks[2]] = np.stack([z[masks[2]], c[masks[2]], xcol[masks[2]]], axis=1)
    rgb[masks[3]] = np.stack([z[masks[3]], xcol[masks[3]], c[masks[3]]], axis=1)
    rgb[masks[4]] = np.stack([xcol[masks[4]], z[masks[4]], c[masks[4]]], axis=1)
    rgb[masks[5]] = np.stack([c[masks[5]], z[masks[5]], xcol[masks[5]]], axis=1) 

    # compute the final intensity based on the ring, disk, and warp
    intensity = (0.12 * ring + 1.2 * disk) * (0.45 + 0.55 * warp)
    rgb *= intensity[..., None]
    rgb += (core * 0.015)[..., None]
    rgb = np.clip(rgb, 0.0, 1.0)
    # OpenCV expects BGR.
    return (rgb[:, :, ::-1] * 255.0).astype(np.uint8)





def main():
    cv2.namedWindow("Project Phi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Project Phi", WIDTH, HEIGHT)

    zoom_c = 0.0
    theta = phi = 0.0
    pan = np.zeros(3, dtype=np.float32)
    running = True
    playing = True
    sim_time = 0.0
    previous = time.perf_counter()

    while running:
        now = time.perf_counter()
        dt = min(now - previous, 0.1)
        previous = now
        if playing:
            sim_time += dt * TIME_FLOW

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # esc
            running = False
        elif key == 32:  # space
            playing = not playing

        frame = numpy_preview(WIDTH, HEIGHT, sim_time, zoom_c, theta, phi, pan)

        cv2.imshow("Project Phi", frame)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
        