from flask import Flask,render_template,Response,request,Blueprint
from AI.views.auth_views import login_required
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import pandas as pd
from ..AI_model import eyeTest as et
import datetime
import time

bp = Blueprint('glaucoma' , __name__ , url_prefix='/test')

@bp.route('/glaucoma')
@login_required # @login_required 데코레이터
def glaucoma():
    return render_template('test/glaucoma.html')

d_start = 30  # 시작거리
d_end = 50  # 끝나는 거리
disease_name = '녹내장'
userID = '000000001'

now = datetime.datetime.now()
nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

detector = FaceMeshDetector(maxFaces=1)
mp_hands = mp.solutions.hands

width = 1260
height = 720
testEnd = False
another = False
counter = 0
selectionSpeed = 8
btn_size = 40
List = []
eye = '오른쪽눈'

logo = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/assets/img/button/eye.png', cv2.IMREAD_UNCHANGED), (80, 80))
test = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/assets/img/button/test.png', cv2.IMREAD_UNCHANGED), (300, 210))
font = ImageFont.truetype('C:Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/fonts/H2GSRB.TTF', 40)
orgin_font = 'C:Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/fonts/H2GSRB.TTF'
background = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/assets/img/button/background.jpg'), (1000, 630))
true = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/assets/img/button/Yes.png', cv2.IMREAD_UNCHANGED), (80, 80))
false = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/assets/img/button/No.png', cv2.IMREAD_UNCHANGED), (80, 80))
disease = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/image/glaucoma/glaucoma.jpg'), (740, 440))
cap = cv2.VideoCapture(0)

def text_def(xy, text_name, fontstyle, text_color):
    global image
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    draw.text(xy=xy, text=text_name, font=fontstyle, fill=text_color)
    image = np.array(image)


def image_def(image_name):
    global image
    h, w, _ = image_name.shape
    image[int(360 - h / 2):int(360 + h / 2), int(630 - w / 2):int(630 + w / 2)] = image_name

def gen(cap):
    global image, testEnd, another, eye, counter
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            success, image = cap.read()

            image = cv2.flip(image, 1)
            image, faces = detector.findFaceMesh(image, draw=False)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_def(disease)

            if testEnd is False:
                text_def((100, 35), f'한쪽눈을 가리고 얼굴이 정면을 바라본 상태에서 그림같은 자세를 취하고 해당방향의 \n손가락이 보이면 V, 안보이면 X에 손가락을 대주세요.', font,
                         (255, 255, 255))
                if another is False:
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            finger = hand_landmarks.landmark[8]
                            h, w, _ = image.shape
                            finger = (int(finger.x * w), int(finger.y * h))
                            if (abs(finger[0] - 260) < btn_size) & (abs(finger[1] - 360) < btn_size):
                                counter += 1
                                cv2.ellipse(image, (260, 360), (btn_size, btn_size), 0, 0, counter * selectionSpeed,
                                            (255, 0, 255), 20)
                                if counter * selectionSpeed > 360:
                                    counter = 0
                                    answer = '정상'
                                    List.append({
                                        'ID': userID,
                                        '시간': nowDatetime,
                                        '눈': eye,
                                        '여부': answer
                                    })
                                    df = pd.DataFrame(List)
                                    df.to_csv(f'C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/csv/{disease_name}.csv')
                                    print(f'{answer}으로 의심됩니다.')
                                    timeStart = time.time()
                                    another = True

                            elif (abs(finger[0] - 1000) < btn_size) & (abs(finger[1] - 360) < btn_size):
                                counter += 1
                                cv2.ellipse(image, (1000, 360), (btn_size, btn_size), 0, 0, counter * selectionSpeed,
                                            (255, 0, 255), 20)
                                if counter * selectionSpeed > 360:
                                    counter = 0
                                    answer = disease_name
                                    List.append({
                                        'ID': userID,
                                        '시간': nowDatetime,
                                        '눈': eye,
                                        '여부': answer
                                    })
                                    df = pd.DataFrame(List)
                                    df.to_csv(f'C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/csv/{disease_name}.csv')
                                    print(f'{answer}입니다.')
                                    timeStart = time.time()
                                    another = True
                            else:
                                pass

                            cv2.circle(image, finger, 20, (255, 0, 0), 2, cv2.LINE_AA)  # 파란색

                    if len(List) == 2:
                        timeStart = time.time()
                        testEnd = True

                    et.overlay(image, *(260, 360), 40, 40, true)
                    et.overlay(image, *(1000, 360), 40, 40, false)
                    et.overlay(image, *(150, 630), 150, 120, test)  # 횟수창
                    text_def((50, 607), f'{eye}', ImageFont.truetype('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/fonts/H2GSRB.TTF', 30), (0, 0, 0))

                else:
                    image_def(background)
                    eye = '왼쪽눈'
                    text_def((370, 250), '테스트 계속하기', ImageFont.truetype('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/fonts/H2GSRB.TTF', 70), (0, 0, 0))
                    text_def((350, 380), f'{int(6 - (time.time() - timeStart))}초후 {eye} 테스트 시작합니다.',
                             ImageFont.truetype('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/fonts/H2GSRB.TTF', 40), (0, 0, 0))
                    if int(6 - (time.time() - timeStart)) == 0:
                        another = False
            else:
                image_def(background)
                text_def((300, 220), f"{disease_name}테스트 검사결과 ", ImageFont.truetype('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/fonts/H2GSRB.TTF', 60),
                         (0, 0, 0))
                text_def((400, 360), f"{List[0]['눈']}: {List[0]['여부']}입니다 ", font, (0, 0, 0))
                text_def((400, 450), f"{List[1]['눈']}: {List[1]['여부']}입니다 ", font, (0, 0, 0))
                text_def((350, 580), f'{int(11 - (time.time() - timeStart))}초후 테스트 종료합니다.',
                         ImageFont.truetype('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/fonts/H2GSRB.TTF', 40), (0, 0, 0))
                if int(11 - (time.time() - timeStart)) == 0:
                    break

            et.overlay(image, *(50, 50), 40, 40, logo)

            cv2.imshow("glaucoma", image)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    print(List)
    print('저장 완료됬습니다.')

    cv2.destroyAllWindows()
    cap.release()


@bp.route('/glaucoma_camera')
def video_feed():
    global cap
    if Response(gen(cap),mimetype='multipart/x-mixed-replace; boundary=frame'):
        return render_template("test/glaucoma.html")    # 윈도우창이 출력시 카메라 페이지로 다시 돌아간다