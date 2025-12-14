"""
OmnidirectionalViewer用ヘルパー関数テンプレート
"""

OMNI_HELPER = '''
def _create_omni_map(pitch, yaw, roll, imagepoint, output_width=960, output_height=540):
    """Calculate remap coordinates for omnidirectional image"""
    sensor_size = 0.561
    sensor_width = sensor_size
    sensor_height = sensor_size * (output_height / output_width)
    viewpoint = -1.0

    # Generate rotation matrix
    roll_rad = roll * np.pi / 180
    pitch_rad = pitch * np.pi / 180
    yaw_rad = yaw * np.pi / 180

    m1 = np.array([[1, 0, 0], [0, np.cos(roll_rad), np.sin(roll_rad)], [0, -np.sin(roll_rad), np.cos(roll_rad)]])
    m2 = np.array([[np.cos(pitch_rad), 0, -np.sin(pitch_rad)], [0, 1, 0], [np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
    m3 = np.array([[np.cos(yaw_rad), np.sin(yaw_rad), 0], [-np.sin(yaw_rad), np.cos(yaw_rad), 0], [0, 0, 1]])
    rotation_matrix = np.dot(m3, np.dot(m2, m1))

    # Calculate phi, theta coordinates
    width = np.arange(-sensor_width, sensor_width, sensor_width * 2 / output_width)
    height = np.arange(-sensor_height, sensor_height, sensor_height * 2 / output_height)
    ww, hh = np.meshgrid(width, height)

    point_distance = imagepoint - viewpoint
    if point_distance == 0:
        point_distance = 0.1

    a1 = ww / point_distance
    a2 = hh / point_distance
    b1 = -a1 * viewpoint
    b2 = -a2 * viewpoint

    a = 1 + (a1 ** 2) + (a2 ** 2)
    b = 2 * ((a1 * b1) + (a2 * b2))
    c = (b1 ** 2) + (b2 ** 2) - 1
    d = ((b ** 2) - (4 * a * c)) ** 0.5

    x = (-b + d) / (2 * a)
    y = (a1 * x) + b1
    z = (a2 * x) + b2

    xd = rotation_matrix[0][0] * x + rotation_matrix[0][1] * y + rotation_matrix[0][2] * z
    yd = rotation_matrix[1][0] * x + rotation_matrix[1][1] * y + rotation_matrix[1][2] * z
    zd = rotation_matrix[2][0] * x + rotation_matrix[2][1] * y + rotation_matrix[2][2] * z

    phi = np.arcsin(zd)
    theta = np.arcsin(yd / np.cos(phi))

    xd_mask = np.where(xd > 0, 0, 1)
    yd_offset = np.where(yd > 0, np.pi, -np.pi)
    offset = yd_offset * xd_mask
    gain = -2 * xd_mask + 1
    theta = gain * theta + offset

    return phi, theta


def _remap_omni(image, phi, theta):
    """Remap omnidirectional image"""
    h, w = image.shape[:2]
    phi_map = (phi * h / np.pi + h / 2).astype(np.float32)
    theta_map = (theta * w / (2 * np.pi) + w / 2).astype(np.float32)
    return cv2.remap(image, theta_map, phi_map, cv2.INTER_CUBIC)
'''
