from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import json
from scipy.spatial.distance import cosine
import mysql.connector
import time
from torchvision import transforms
import uvicorn


app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        #모든 도메인에서 요청을 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터베이스 연결 함수
def connect_database():
    return mysql.connector.connect(
        host="localhost",
        user="face",
        password="1234",
        database="my_face"
    )

# 얼굴 인식 클래스 정의
class FaceRecognition:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.mtcnn = MTCNN(keep_all=True, device=self.device)  # return_landmarks 제거
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.previous_name = None
        self.current_name = None
        self.recognition_start_time = None

    def detect_faces(self, frame):
        # MTCNN의 detect 메서드: boxes와 probs만 반환
        boxes, probs = self.mtcnn.detect(frame)
        return boxes, probs

    def get_embedding(self, face_image):
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        face_tensor = transform(face_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.facenet(face_tensor).cpu().numpy().flatten()

    def find_closest_match(self, embedding_vector, threshold=0.35):
        conn = connect_database()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT name, embedding_vector FROM users")
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

    def process_frame(self, image_rgb):
        boxes, _ = self.detect_faces(image_rgb)
        if boxes is not None and len(boxes) > 0:
            largest_box = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
            x1, y1, x2, y2 = map(int, largest_box)
            face_crop = image_rgb[y1:y2, x1:x2]
            face_image = Image.fromarray(face_crop)

            embedding = self.get_embedding(face_image)
            closest_name = self.find_closest_match(embedding)

            # 이름 변경 시 타이머 초기화
            if closest_name != self.previous_name:
                self.previous_name = closest_name
                self.recognition_start_time = time.time()

            # 동일한 이름이 5초 동안 감지된 경우
            if closest_name == self.previous_name and closest_name is not None:
                elapsed_time = time.time() - self.recognition_start_time
                remaining_time = max(0, 5 - elapsed_time)

                if elapsed_time >= 5:
                    self.current_name = closest_name
                return {"currentName": self.current_name or "Unknown", "remainingTime": int(remaining_time)}

        # 얼굴 감지 실패 시 초기화
        self.previous_name = None
        self.recognition_start_time = None
        return {"currentName": "", "remainingTime": 5}


recognition = FaceRecognition()

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 감지 및 처리
    result = recognition.process_frame(image_rgb)

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)