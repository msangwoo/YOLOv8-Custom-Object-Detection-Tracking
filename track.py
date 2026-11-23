import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# 1. 학습된 best.pt 모델 및 테스트 영상 로드
model = YOLO("runs/detect/runs/detect/custom_yolo-2/weights/best.pt")
video_path = "test_10sec.mov"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

# 출력 비디오 설정
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("tracked_result_10sec.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# 객체별 이동 궤적(Trajectory) 저장용 딕셔너리
track_history = defaultdict(lambda: [])

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # persist=True로 프레임 간 객체 ID 유지 (ByteTrack 사용)
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.4)
    
    # 바운딩 박스, 클래스명, 추적 ID가 표시된 프레임 생성
    annotated_frame = results[0].plot()

    # 객체 중심점 이동 궤적(Trajectory) 시각화
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, _, _ = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:  # 최근 30프레임 궤적 유지
                track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 0), thickness=3)

    out.write(annotated_frame)

cap.release()
out.release()
print("추적 동영상 생성 완료: tracked_result_10sec.mp4")
