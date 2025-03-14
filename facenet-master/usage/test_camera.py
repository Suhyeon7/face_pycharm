# import cv2
#
# cap = cv2.VideoCapture(0)  # 카메라 인덱스 0
# # AVFoundation 사용
# #cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
# if not cap.isOpened():
#     print("Camera access failed. Check permissions.")
# else:
#     print("Camera accessed successfully!")
#     cap.release()
import cv2

# 웹캠 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
        break

cap.release()
cv2.destroyAllWindows()
