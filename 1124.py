#1124.py
import mysql.connector
import json
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
import time  # 시간 측정을 위한 모듈

# MySQL 데이터베이스 연결
def connect_database():
    return mysql.connector.connect(
        host="localhost",
        user="face",
        password="1234",
        database="my_face"
    )

# FaceNet 모델 정의
class FaceNet(torch.nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        self.model = InceptionResnetV1(pretrained='vggface2').eval()  # Pretrained FaceNet 모델

    def forward(self, x):
        return self.model(x)


# 얼굴 탐지 및 임베딩 생성 클래스
class FaceRecognition:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.facenet = FaceNet().to(self.device)
        self.facenet.eval()

    def detect_faces(self, frame):
        boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
        return boxes, probs, landmarks

    def get_embedding(self, face):
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        face_tensor = transform(face).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.facenet(face_tensor).cpu().numpy().flatten().tolist()
        return embedding


# 데이터베이스에서 가장 유사한 얼굴 찾기
def find_closest_match(embedding_vector, threshold=0.35):
    conn = connect_database()
    cursor = conn.cursor(dictionary=True)

    # 모든 사용자 데이터 가져오기
    cursor.execute("SELECT user_id, name, embedding_vector FROM users")
    rows = cursor.fetchall()

    closest_name = None
    min_distance = float('inf')

    for row in rows:
        db_embedding = np.array(json.loads(row["embedding_vector"])).flatten()
        distance = cosine(embedding_vector, db_embedding)
        if distance < threshold and distance < min_distance:
            closest_name = row["name"]
            min_distance = distance

    cursor.close()
    conn.close()

    return closest_name


# 얼굴 임베딩 DB에 추가
def add_to_database(name, embedding_vector):
    conn = connect_database()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO users (name, embedding_vector) VALUES (%s,%s)
    """, (name, json.dumps(embedding_vector)))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Added {name} to the database.")


# 가장 큰 얼굴 바운딩 박스를 선택
def get_largest_face(boxes):
    if boxes is None:
        return None
    largest_box = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    return largest_box


# 실시간 얼굴 감지 및 DB 추가/조회
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    face_recognition = FaceRecognition(device)

    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    previous_name = None  # 이전에 인식된 이름
    current_name = None  # 현재 이름
    recognition_start_time = None  # 현재 이름이 언제 시작되었는지 저장
    countdown_display_time = 0  # 화면에 카운트다운을 표시할 시간

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to capture video.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = face_recognition.detect_faces(frame_rgb)

        if boxes is not None and len(boxes) > 0:
            largest_box = get_largest_face(boxes)
            x1, y1, x2, y2 = map(int, largest_box)
            face_crop = frame_rgb[y1:y2, x1:x2]

            # 얼굴 임베딩 생성
            embedding = face_recognition.get_embedding(Image.fromarray(face_crop))

            # 데이터베이스에서 가장 가까운 얼굴 찾기
            closest_name = find_closest_match(embedding)

            # 화면에 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            display_name = closest_name if closest_name else "Unknown"
            cv2.putText(frame, display_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 이름 변화 감지 및 타이머 초기화
            if closest_name != previous_name:
                previous_name = closest_name
                recognition_start_time = time.time()  # 새 이름으로 타이머 시작

            # 5초 이상 같은 이름이 감지된 경우
            if closest_name == previous_name and closest_name is not None:
                elapsed_time = time.time() - recognition_start_time
                remaining_time = max(0, 5 - elapsed_time)

                if elapsed_time >= 5 and closest_name != current_name:
                    current_name = closest_name
                    print(f"currentName은 {current_name}입니다")

                # 카운트다운을 화면에 표시
                cv2.putText(frame, f"Recognizing: {int(remaining_time)}s",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                recognition_start_time = time.time()  # 이름이 바뀌면 타이머 초기화
        else:
            #print("No face detected.")  # 얼굴이 감지되지 않은 경우 메시지 출력
            previous_name = None  # 얼굴이 감지되지 않으면 이름 초기화
            recognition_start_time = None

        # 화면에 currentName 표시
        if current_name:
            cv2.putText(frame, f"currentName: {current_name}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 키
            break
        elif key == ord('n') and closest_name is None:  # 'n' 키가 눌렸을 때 새로운 사용자 추가
            name = input("Enter name for the detected face: ")
            add_to_database(name, embedding)
            print(f"{name} has been added to the database.")

    video_capture.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
