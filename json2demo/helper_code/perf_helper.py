"""
パフォーマンス描画ヘルパー関数テンプレート
"""

PERF_HELPER = '''
def draw_perf_info(image, perf_times, perf_keys, total_ms, fps):
    """Draw performance info on image"""
    result = image.copy()
    y = 30
    for key in perf_keys:
        if key in perf_times:
            text = f'{key}: {perf_times[key]:.1f}ms'
            cv2.putText(result, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            y += 25
    total_text = f'Total: {total_ms:.1f}ms ({fps:.1f}FPS)'
    cv2.putText(result, total_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result, total_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 1)
    return result
'''
