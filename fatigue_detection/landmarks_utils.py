import numpy as np
def landmarks_to_points(landmarks, indices, image_w, image_h):
    """
    Convert MediaPipe landmark indices to pixel coordinates.
    """
    return [(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in indices]

def calculate_ear(points):
    """
    Calculate Eye Aspect Ratio (EAR) from 6 landmark points.
    p1, p4: Outer and inner eye corners (horizontal endpoints)

    p2, p6: Upper and lower vertical landmarks (first vertical pair)

    p3, p5: Second vertical pair
    """
    p1, p2, p3, p4, p5, p6 = points

    # Vertical distances
    vertical_1 = np.linalg.norm(np.array(p2) - np.array(p6))
    vertical_2 = np.linalg.norm(np.array(p3) - np.array(p5))
    # Horizontal distance
    horizontal = np.linalg.norm(np.array(p1) - np.array(p4))

    # EAR formula
    if horizontal == 0:
        return 0  # prevent divide by zero
    return (vertical_1 + vertical_2) / (2.0 * horizontal)



