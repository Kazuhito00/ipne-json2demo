"""
Multi-Object Tracking ヘルパー関数テンプレート
"""

MOT_HELPER = '''
class MOTTracker:
    """Multi-Object Tracker with sequential ID assignment"""
    def __init__(self, dt=1/30, max_staleness=5):
        self.tracker = MultiObjectTracker(dt=dt, tracker_kwargs={'max_staleness': max_staleness})
        self.id_map = {}  # motpy ID -> sequential ID
        self.next_id = 1

    def step(self, detections):
        self.tracker.step(detections=detections)

    def active_tracks(self):
        tracks = self.tracker.active_tracks()
        result = []
        for track in tracks:
            orig_id = track.id
            if orig_id not in self.id_map:
                self.id_map[orig_id] = self.next_id
                self.next_id += 1
            result.append({'id': self.id_map[orig_id], 'box': track.box})
        return result

def _mot_get_track_color(track_id):
    """Generate unique color based on track ID"""
    temp_index = abs(int(track_id + 35)) * 3
    return ((29 * temp_index) % 255, (17 * temp_index) % 255, (37 * temp_index) % 255)

def _mot_draw_tracks(image, tracks, thickness=2):
    """Draw tracking results on image"""
    debug_image = image.copy()
    for track in tracks:
        track_id = track['id']
        box = track['box']
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        color = _mot_get_track_color(track_id)
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, thickness=thickness)
        label = f"ID:{track_id}"
        cv2.putText(debug_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness + 1)
        cv2.putText(debug_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)
    return debug_image
'''
