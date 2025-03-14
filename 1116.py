import mysql.connector
import json
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine

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
def find_closest_match(embedding_vector, threshold=0.6):
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
    INSERT INTO users (name, age, embedding_vector) VALUES (%s, %s, %s)
    """, (name, 30, json.dumps(embedding_vector)))

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

            if closest_name:
                print(f"Recognized: {closest_name}")
            else:
                # 사용자 입력 받기
                name = input("Enter name for the detected face: ")
                add_to_database(name, embedding)
                closest_name = name

            # 화면에 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, closest_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            print("No face detected.")  # 얼굴이 감지되지 않은 경우 메시지 출력


        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 키
            break

    video_capture.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
