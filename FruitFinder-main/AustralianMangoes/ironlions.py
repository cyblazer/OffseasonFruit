import numpy as np
import cv2, math, random

fg_scale = 0.7
spread_iter = 10

last_was_detected = False

def get_contour_center(contour):
    M = cv2.moments(contour)
    
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    else:
        return None

def rectify_by_camera_angle_3d(image, angle_degrees, plane_distance=1.0, focal_length=None):
    """
    Simulate a 3D camera rotation (around the X-axis) and warp the image accordingly,
    keeping the result centered in the original frame.

    Parameters
    ----------
    image : ndarray
        Input BGR image.
    angle_degrees : float
        Tilt angle (in degrees) between the camera's optical axis and the Z-axis.
        Positive = tilt down; negative = tilt up.
    plane_distance : float
        Distance from camera center to the planar scene (in arbitrary units).
        Controls the strength of perspective distortion.
    focal_length : float or None
        Focal length in pixels. If None, uses 0.8 * max(width, height).

    Returns
    -------
    warped : ndarray
        The perspective-warped image, same size as input.
    H : ndarray, shape (3,3)
        The homography matrix that was applied.
    """
    h, w = image.shape[:2]
    cx, cy = w/2.0, h/2.0

    # 1) Camera intrinsics
    if focal_length is None:
        f = 0.8 * max(w, h)
    else:
        f = focal_length
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float32)

    # 2) Define the four corners of the image plane at Z = plane_distance
    corners = np.array([
        [0,   0],
        [w,   0],
        [w,   h],
        [0,   h],
    ], dtype=np.float32)

    # 3) Lift corners into 3D
    Z = plane_distance
    pts3d = []
    for u, v in corners:
        X = (u - cx) * Z / f
        Y = (v - cy) * Z / f
        pts3d.append([X, Y, Z])
    pts3d = np.array(pts3d, dtype=np.float32)  # shape (4,3)

    # 4) Rotate in 3D around the X-axis by the given angle
    theta = np.deg2rad(angle_degrees)
    Rx = np.array([
        [1,           0,            0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)],
    ], dtype=np.float32)
    pts3d_rot = (Rx @ pts3d.T).T  # shape (4,3)

    # 5) Project back into the image plane
    proj = []
    for X, Y, Zp in pts3d_rot:
        u2 = (f * X / Zp) + cx
        v2 = (f * Y / Zp) + cy
        proj.append([u2, v2])
    proj = np.array(proj, dtype=np.float32)  # shape (4,2)

    # 6) Compute a translation to recenter the warped image
    #    so that the centroid of the projected corners
    #    lands back at the image center.
    orig_center = np.array([cx, cy], dtype=np.float32)
    proj_center = proj.mean(axis=0)
    offset = orig_center - proj_center
    proj_trans = proj + offset

    # 7) Build the homography and warp
    H = cv2.getPerspectiveTransform(corners, proj_trans)
    warped = cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_LINEAR)

    return warped, H

def runPipeline(image, b):
    image,_ = rectify_by_camera_angle_3d(image, 30)
    global fg_scale
    global spread_iter
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([10, 230, 130])
    upper_bound = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=spread_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=spread_iter)
    masked = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 200,400)

    edges = cv2.dilate(edges, kernel, iterations=5)
    edges = cv2.erode(edges, kernel, iterations=10)
    sure_bg = cv2.dilate(mask, kernel, iterations=5)

    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, fg_scale * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    image_watershed = image.copy()
    markers = cv2.watershed(image_watershed, markers)

    segments = np.zeros_like(image)
    largest_contour = np.array([[]])
    largest_area = 0

    for marker_id in np.unique(markers):
        if marker_id <= 1: continue
        mask = markers == marker_id
        if np.count_nonzero(mask) < 400 or np.count_nonzero(mask) > 8000: continue
        segments[mask] = [random.randint(0, 255) for _ in range(3)]

        mask = np.uint8(markers == marker_id) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        for contour in contours:
            center = get_contour_center(contour)
            cv2.circle(image, center, 10, (255, 0, 0), 10)
            if not largest_contour.any():
                largest_contour = contour
                continue
            area = cv2.contourArea(contour)
            if area > largest_area:
                largest_contour = contour
                largest_area = area

    image_watershed[markers == -1] = [0, 0, 255]

    final_result = cv2.addWeighted(image_watershed, 0, segments, 1, 0)
    return largest_contour, image, [0, 0, 0]