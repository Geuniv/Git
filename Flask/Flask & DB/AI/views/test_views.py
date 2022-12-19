# # 라이브러리
# import time
# from flask import Blueprint , render_template, Response , url_for , flash ,request
# import  os
#
# import tensorflow as rf
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from PIL import Image
#
# import torchvision
# from torchvision import datasets, models, transforms
#
# import cv2
# import mediapipe as mp
# import datetime
# import scipy
# from ..AI_model import cataract_predict as cp
#
# bp = Blueprint('test' , __name__ , url_prefix='/test')
#
# model = models.resnet34(pretrained=True)
# num_features = model.fc.in_features
# model.fc = nn.Linear(num_features, 3)
# model.load_state_dict(torch.load("C:/Users/user/Desktop/pythonProject/pythonProject/Flask & DB/AI/AI_model/model_dict.pth", map_location ='cpu'), strict=False)
# model.eval()
# print('모델 로드 완료')
# transforms_test = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
#
# model = model.to('cpu')
#
# mp_face_detection = mp.solutions.face_detection  # 얼굴 검출
# mp_drawing = mp.solutions.drawing_utils  # 얼굴 특징 표시
#
# cap = cv2.VideoCapture(0)
#
# # 메인 서버
# @bp.route("/")
# def test():
#     return render_template("main.html")
#
#
# # 데이터 예측 처리
# @bp.route('/predict', methods=['POST'])
# def make_prediction():
#     if request.method == 'POST':
#
#         # 업로드 파일 처리 분기
#         file = request.files['image']
#         if not file: return render_template('index.html', label="No Files")
#
#         # 이미지 픽셀 정보 읽기
#         # 알파 채널 값 제거 후 1차원 Reshape
#         img = misc.imread(file)
#         img = img[:, :, :3]
#         img = img.reshape(1, -1)
#
#         # 입력 받은 이미지 예측
#         prediction = model.predict(img)
#
#         # 예측 값을 1차원 배열로부터 확인 가능한 문자열로 변환
#         label = str(np.squeeze(prediction))
#
#         # 숫자가 10일 경우 0으로 처리
#         if label == '10': label = '0'
#
#         # 결과 리턴
#         return render_template('index.html', label=label)
#
# if __name__ == '__main__':
#     # 모델 로드
#     # ml/model.py 선 실행 후 생성
#     model = joblib.load('./model/model.pkl')
#     # Flask 서비스 스타트
#     app.run
#
# if __name__ == '__main__':
#     # 모델 로드
#     # ml/model.py 선 실행 후 생성
#     model = joblib.load('./model/model.pkl')
#     # Flask 서비스 스타트
#     return
#
# # 메인카메라 서버
# @bp.route("/Camera")
# def camera():
#     return render_template("camera.html")
#
#
# @bp.route("/image_page")
# def first():
#     return render_template("image_page.html")
#
#
# def gen(cap):
#     # model_selection=0 -> 2m 내, model_selection=1 -> 5m 내
#     with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection:
#         while True:
#             cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#             cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
#             success, image = cap.read()
#
#             image.flags.writeable = False
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = face_detection.process(image)
#
#             # Draw the face detection annotations on the image.
#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#             if results.detections:
#                 for detection in results.detections:
#                     keypoints = detection.location_data.relative_keypoints
#                     right_eye = keypoints[0]  # 왼쪽눈
#                     left_eye = keypoints[1]  # 오른쪽눈
#
#                     h, w, _ = image.shape
#                     right = image[int(right_eye.y * h - 40):int(right_eye.y * h + 40),
#                             int(right_eye.x * w - 60):int(right_eye.x * w + 40)]
#                     left = image[int(left_eye.y * h - 40):int(left_eye.y * h + 40),
#                            int(left_eye.x * w - 40):int(left_eye.x * w + 60)]
#
#             now = datetime.datetime.now().strftime("%d_%H-%M-%S")
#             right_name = './image_/' + 'right_image' + str(now) + ".jpg"
#             left_name = './image_/' + 'left_image' + str(now) + ".jpg"
#             eye_name = './image_/' + 'eye_image' + str(now) + ".jpg"
#             cv2.imshow("Image", image)
#             key = cv2.waitKey(1)
#
#             if key == ord('q'):
#                 break
#
#             elif key == ord('c'):
#                 print("Right_eye")
#                 cv2.imwrite(right_name, right)
#                 cp.image_test(right_name)
#                 print("Left_eye")
#                 cv2.imwrite(left_name, left)
#                 cp.image_test(left_name)
#
#             elif key == ord('r'):
#                 print("Right_eye")
#                 cv2.imwrite(right_name, right)
#                 cp.image_test(right_name)
#
#             elif key == ord('l'):
#                 print("Left_eye")
#                 cv2.imwrite(left_name, left)
#                 cp.image_test(left_name)
#
#     cv2.destroyAllWindows()
#     result = cp.image_test('./image_/eye_image18_14-29-55.jpg')
#     return result
#
#
# # 카메라 서버
# @bp.route('/video_feed')
# def video_feed():
#     global cap
#     if Response(gen(cap), mimetype='multipart/x-mixed-replace; boundary=frame'):
#         return render_template("camera.html")  # 윈도우창이 출력시 카메라 페이지로 다시 돌아간다
