from flask import render_template,Response,Blueprint
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import pandas as pd
from ..AI_model import cataract_predict as cp
import datetime
import time

bp = Blueprint('cataract' , __name__ , url_prefix='/test')

@bp.route('/test/cataract')
def camera():
    return render_template('test/cataract_camera.html')

userID = '000000001'

now = datetime.datetime.now()
nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

mp_face_detection = mp.solutions.face_detection  # 얼굴 검출
mp_drawing = mp.solutions.drawing_utils  # 얼굴 특징 표시
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
width = 1260
height = 720
font = 'C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/fonts/H2GSRB.TTF'
logo = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/assets/img/button/eye.png', cv2.IMREAD_UNCHANGED), (80, 80))
face = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/assets/img/button/face.png', cv2.IMREAD_UNCHANGED), (80, 80))
background = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/assets/img/button/background.jpg'), (1000, 630))

def overlay(image, x, y, w, h, overlay_image):  # 대상 이미지 (3채널), x, y 좌표, width, height, 덮어씌울 이미지 (4채널:투명도를 가짐)
    alpha = overlay_image[:, :, 3]  # BGR
    image_alpha = alpha / 255  # 0 ~ 255 -> 255 로 나누면 0 ~ 1 사이의 값 (1: 불투명, 0: 완전투명)
    for c in range(3):  # channel BGR
        image[y - h:y + h, x - w:x + w, c] = (overlay_image[:, :, c] * image_alpha) + (
                    image[y - h:y + h, x - w:x + w, c] * (1 - image_alpha))

def img_size(image,def_img, img_x, img_y):
    h, w, _ = def_img.shape
    image[int(img_y - h / 2):int(img_y + h / 2), int(img_x - w / 2):int(img_x + w / 2)] = def_img

testEnd = False
List = []

def gen(cap):
    global testEnd
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection:
        while True:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            success, image = cap.read()
            image = cv2.flip(image, 1)

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            draw.text(xy=(100, 35),text= f'손바닥이 보이게 손을 든 후 정면을 바라보고 주먹을 쥐어주세요',font=ImageFont.truetype(font, 40), fill=(255, 255, 255))
            image = np.array(image)

            if testEnd is False:
                if results.detections:
                    for detection in results.detections:
                        keypoints = detection.location_data.relative_keypoints
                        right_eye = keypoints[0]  # 왼쪽눈
                        left_eye = keypoints[1]  # 오른쪽눈

                        h, w, _ = image.shape
                        right = image[int(right_eye.y * h - 40):int(right_eye.y * h + 40),
                                int(right_eye.x * w - 60):int(right_eye.x * w + 40)]
                        left = image[int(left_eye.y * h - 40):int(left_eye.y * h + 40),
                               int(left_eye.x * w - 40):int(left_eye.x * w + 40)]

                now = datetime.datetime.now().strftime("%d_%H-%M-%S")
                right_name = 'C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/assets/image/' + 'right_image' + str(now) + ".jpg"
                left_name = 'C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/assets/image/' + 'left_image' + str(now) + ".jpg"

                with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5) as hands:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = hands.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            finger1 = int(hand_landmarks.landmark[8].y * 100)
                            finger2 = int(hand_landmarks.landmark[5].y * 100)
                            hand = int(hand_landmarks.landmark[0].y * 100)
                            dist = abs(int(finger1 - hand))
                            dist2 = abs(int(finger2 - hand))

                            overlay(image, *(450, 130), 40, 40, face)

                            image = Image.fromarray(image)
                            draw = ImageDraw.Draw(image)
                            draw.text(xy=(495, 115), text= f'정면을 보고 주먹을 쥐어주세요', font=ImageFont.truetype(font,40), fill=(0, 225, 225))
                            image = np.array(image)

                            if dist < dist2:
                                print("왼쪽 눈")
                                cv2.imwrite(right_name, right)
                                class_name, score = cp.image_test(right_name)
                                score_str = str(round(score)) + '%'
                                List.append({
                                    'ID': userID,
                                    '시간': nowDatetime,
                                    '여부': class_name,
                                    '확률': score_str,
                                    '이미지': right_name
                                })
                                print(class_name, score_str)
                                print("오른쪽 눈")

                                cv2.imwrite(left_name, left)
                                class_name, score = cp.image_test(left_name)
                                score_str = str(round(score)) + '%'
                                List.append({
                                    'ID': userID,
                                    '시간': nowDatetime,
                                    '여부': class_name,
                                    '확률': score_str,
                                    '이미지': left_name
                                })
                                print(class_name, score_str)
                                print('-' * 50)
                                df = pd.DataFrame(List)
                                df.to_csv(f'C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/csv/cataract.csv')
                                timeStart = time.time()
                                testEnd = True

            else:
                img_size(image, background, 630, 360)
                image = Image.fromarray(image)
                draw = ImageDraw.Draw(image)
                draw.text(xy=(320, 220), text=f"백내장 테스트 결과 ", font=ImageFont.truetype(font, 70),fill=(0, 0, 0))
                draw.text(xy=(400, 360), text=f" 왼 쪽 :          {score_str} {class_name}", font=ImageFont.truetype(font,40), fill=(0, 0, 0))
                draw.text(xy=(400, 480), text=f"오른쪽 :         {score_str} {class_name}", font=ImageFont.truetype(font,40), fill=(0, 0, 0))
                draw.text(xy=(410, 590), text=f'{int(11 - (time.time() - timeStart))}초후 테스트 종료합니다.',font=ImageFont.truetype(font,40), fill=(0, 0, 0))
                image = np.array(image)
                if int(11 - (time.time() - timeStart)) == 0:
                    break

                right_img = cv2.resize(cv2.imread(right_name, cv2.IMREAD_UNCHANGED), (80, 80))
                left_img = cv2.resize(cv2.imread(left_name, cv2.IMREAD_UNCHANGED), (80, 80))
                img_size(image, right_img, 600, 380)
                img_size(image, left_img, 600, 500)

            overlay(image, *(50, 50), 40, 40, logo)
            cv2.imshow("Cataract", image)
            key = cv2.waitKey(1)
            if key == ord('r'):
                testEnd = False

            elif key == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    print(List)
    print('저장 완료됬습니다.')

@bp.route('/test/cataract_camera')
def video_feed():
    global cap
    if Response(gen(cap),mimetype='multipart/x-mixed-replace; boundary=frame'):
        return render_template("test/cataract_main.html")    # 윈도우창이 출력시 카메라 페이지로 다시 돌아간다