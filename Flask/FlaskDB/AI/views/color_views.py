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

bp = Blueprint('color' , __name__ , url_prefix='/test')

@bp.route('/color')
@login_required # @login_required 데코레이터
def color():
    return render_template('test/color.html')

userID = '000000001'
d_start = 50    # 시작거리
d_end = 75      # 끝나는 거리
disease_name = '색각'

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
num = []
eye = '오른쪽눈'
timeStart = time.time()

logo = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/assets/img/button/eye.png', cv2.IMREAD_UNCHANGED), (80, 80))
test = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/assets/img/button/test.png', cv2.IMREAD_UNCHANGED), (300, 210))
font = 'C:Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/fonts/H2GSRB.TTF'
background = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/assets/img/button/background.jpg'), (1000, 630))
true = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/assets/img/button/Yes.png', cv2.IMREAD_UNCHANGED), (80, 80))
false = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/assets/img/button/No.png', cv2.IMREAD_UNCHANGED), (80, 80))
disease = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/image/color/colortest.jpg'),(1000,370))

two = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/image/color/2.png', cv2.IMREAD_UNCHANGED),(80,80))
twentyone = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/image/color/21.png', cv2.IMREAD_UNCHANGED),(80,80))
twentysix = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/image/color/26.png', cv2.IMREAD_UNCHANGED),(80,80))
seventyfour = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/image/color/74.png', cv2.IMREAD_UNCHANGED),(80,80))
nintyseven = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/image/color/97.png', cv2.IMREAD_UNCHANGED),(80,80))

def image_def(img_x,img_y,image_name):
    global image
    h, w, _ = image_name.shape
    image[int(img_y-h/2):int(img_y+h/2),int(img_x-w/2):int(img_x+w/2)]=image_name

def text_def(xy,text_name,fontstyle,text_color):
    global image
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    draw.text(xy=xy,  text=text_name, font=fontstyle, fill= text_color)
    image = np.array(image)


cap = cv2.VideoCapture(0)
def gen(cap):
    global image, testEnd, another, eye, counter, num, image
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

            et.overlay(image, *(210, 550), 40, 40, two)
            et.overlay(image, *(423, 550), 40, 40, twentyone)
            et.overlay(image, *(636, 550), 40, 40, twentysix)
            et.overlay(image, *(849, 550), 40, 40, seventyfour)
            et.overlay(image, *(1060, 550), 40, 40, nintyseven)

            if testEnd is False:
                if another is False:

                    text_def((100, 35), f'그림을 보고 맞는 숫자들을 손가락으로 3초이상 가르켜주세요.', ImageFont.truetype(font,40), (255, 255, 255))
                    image_def(630, 300, disease)
                    et.overlay(image, *(50, 50), 40, 40, logo)
                    text_def((150, 130), f"① ", ImageFont.truetype(font, 70), (0, 0, 0))
                    text_def((500, 130), f"② ", ImageFont.truetype(font, 70), (0, 0, 0))
                    text_def((850, 130), f"③ ", ImageFont.truetype(font, 70), (0, 0, 0))

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            finger = hand_landmarks.landmark[8]
                            h, w, _ = image.shape
                            finger = (int(finger.x * w), int(finger.y * h))
                            if len(num) < 3:
                                if (abs(finger[0] - 210) < btn_size) & (abs(finger[1] - 550) < btn_size):
                                    counter += 1
                                    cv2.ellipse(image, (210, 550), (btn_size, btn_size), 0, 0, counter * selectionSpeed,
                                                (255, 0, 255), 10)
                                    if counter * selectionSpeed > 360:
                                        counter = 0
                                        answer = 2
                                        num.append(answer)

                                elif (abs(finger[0] - 423) < btn_size) & (abs(finger[1] - 550) < btn_size):
                                    counter += 1
                                    cv2.ellipse(image, (423, 550), (btn_size, btn_size), 0, 0, counter * selectionSpeed,
                                                (255, 0, 255), 10)
                                    if counter * selectionSpeed > 360:
                                        counter = 0
                                        answer = 21
                                        num.append(answer)

                                elif (abs(finger[0] - 636) < btn_size) & (abs(finger[1] - 550) < btn_size):
                                    counter += 1
                                    cv2.ellipse(image, (636, 550), (btn_size, btn_size), 0, 0, counter * selectionSpeed,
                                                (255, 0, 255), 10)
                                    if counter * selectionSpeed > 360:
                                        counter = 0
                                        answer = 26
                                        num.append(answer)

                                elif (abs(finger[0] - 849) < btn_size) & (abs(finger[1] - 550) < btn_size):
                                    counter += 1
                                    cv2.ellipse(image, (849, 550), (btn_size, btn_size), 0, 0, counter * selectionSpeed,
                                                (255, 0, 255), 10)
                                    if counter * selectionSpeed > 360:
                                        counter = 0
                                        answer = 74
                                        num.append(answer)

                                elif (abs(finger[0] - 1060) < btn_size) & (abs(finger[1] - 550) < btn_size):
                                    counter += 1
                                    cv2.ellipse(image, (1060, 550), (btn_size, btn_size), 0, 0,
                                                counter * selectionSpeed, (255, 0, 255), 10)
                                    if counter * selectionSpeed > 360:
                                        counter = 0
                                        answer = 97
                                        num.append(answer)
                                else:
                                    pass

                            elif len(num) == 3:
                                if (num[0] == 97) & (num[1] == 74) & (num[2] == 26):
                                    final = '정상'
                                elif (num[1] == 21) & (num[2] != 2):
                                    final = '적녹색맹'
                                elif ((num[1] == 21) & (num[2] == 2)) | (num[2] == 2):
                                    final = '녹색맹'
                                else:
                                    final = '색맹'
                                List.append({
                                    'ID': userID,
                                    '시간': nowDatetime,
                                    '눈': eye,
                                    '여부': final
                                })

                                print(f'{final}입니다')
                                df = pd.DataFrame(List)
                                df.to_csv(f'C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/csv/{disease_name}.csv')

                                if eye == '오른쪽눈':
                                    timeStart = time.time()
                                    another = True
                                else:
                                    timeStart = time.time()
                                    testEnd = True

                            cv2.circle(image, finger, 20, (255, 0, 0), 2, cv2.LINE_AA)  # 파란색
                    et.overlay(image, *(150, 630), 150, 120, test)  # 횟수창
                    text_def((50, 607), f'{eye}', ImageFont.truetype(font, 30), (0, 0, 0))
                else:
                    image_def(630, 360, background)
                    eye = '왼쪽눈'
                    text_def((390, 250), '테스트 계속하기', ImageFont.truetype(font, 70), (0, 0, 0))
                    text_def((350, 380), f'{int(6 - (time.time() - timeStart))}초후 {eye} 테스트 시작합니다.',
                             ImageFont.truetype(font, 40), (0, 0, 0))
                    num = []
                    if int(6 - (time.time() - timeStart)) == 0:
                        another = False

            else:
                image_def(630, 360, background)
                text_def((370, 220), f"{disease_name}테스트 검사결과 ", ImageFont.truetype(font, 60),
                         (0, 0, 0))
                text_def((430, 360), f"{List[0]['눈']}: {List[0]['여부']}입니다 ", ImageFont.truetype(font, 40), (0, 0, 0))
                text_def((430, 450), f"{List[1]['눈']}: {List[1]['여부']}입니다 ", ImageFont.truetype(font, 40), (0, 0, 0))
                text_def((350, 580), f'{int(11 - (time.time() - timeStart))}초후 {eye} 테스트 시작합니다.',
                         ImageFont.truetype(font, 40), (0, 0, 0))
                if int(11 - (time.time() - timeStart)) == 0:
                    break

            cv2.imshow("color", image)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    print(List)
    print('저장 완료됬습니다.')
    cap.release()
    cv2.destroyAllWindows()

@bp.route('/color_camera')
def video_feed():
    global cap
    if Response(gen(cap),mimetype='multipart/x-mixed-replace; boundary=frame'):
        return render_template("test/color.html")    # 윈도우창이 출력시 카메라 페이지로 다시 돌아간다