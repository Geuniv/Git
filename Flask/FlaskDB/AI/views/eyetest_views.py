from flask import Flask,render_template,Response,request,Blueprint
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import pandas as pd
import random
from ..AI_model import eyeTest as et
import datetime
import time
from glob import glob

bp = Blueprint('eyetest' , __name__ , url_prefix='/test')

userID = '000000001'
now = datetime.datetime.now()
nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

mp_face_detection = mp.solutions.face_detection  # 얼굴 검출
mp_drawing = mp.solutions.drawing_utils  # 얼굴 특징 표시
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
width = 1260
height = 720

d_start = 30  # 시작거리
d_end = 60  # 끝나는 거리
userID = '000000001'

dataList = []
list = []
folders = glob('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/image/eyetest/*')
for folder in folders:
    imgs = glob(folder + '/*jpg')
    list.append(random.choice(imgs))
for img in list:
    name = img.split('\\')[-1].replace('.jpg', '')
    category = img.split('\\')[-2].split('/')[-1]
    dataList.append({
        '방향': name,
        '시력': category,
        '경로': img
    })
df = pd.DataFrame(dataList)
df.to_csv('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/csv/eyetest.csv')
print(df)
testEnd = False
another = False
List = []
finalList = []
eye = '오른쪽눈'
width = 1260
height = 720
counter = 0
selectionSpeed = 8
btn_size = 40
idx = 0

now = datetime.datetime.now()
nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

detector = FaceMeshDetector(maxFaces=1)
mp_hands = mp.solutions.hands

btn_right = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/button/lightblue.png', cv2.IMREAD_UNCHANGED), (80, 80))
btn_left = cv2.flip(btn_right, 1)
btn_up = cv2.rotate(btn_right, cv2.ROTATE_90_CLOCKWISE)
btn_down = cv2.flip(btn_up, -1)
logo = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/button/eye.png', cv2.IMREAD_UNCHANGED), (80, 80))
test = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/button/test.png', cv2.IMREAD_UNCHANGED), (300, 210))
font = ImageFont.truetype('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/fonts/H2GSRB.TTF', 40)
background = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/button/background.jpg'), (1000, 630))

def eyeTes_def(idx):
    df = pd.read_csv('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/csv/eyetest.csv')
    eyeImg = cv2.imread(df['경로'][idx])
    eyeName = df['방향'][idx]
    eyeText = df['시력'][idx]
    return eyeText, eyeImg, eyeName

def image_def(image_name):
    global image
    h, w, _ = image_name.shape
    image[int(360 - h / 2):int(360 + h / 2), int(630 - w / 2):int(630 + w / 2)] = image_name


def text_def(xy, text_name, fontstyle, text_color):
    global image
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    draw.text(xy=xy, text=text_name, font=fontstyle, fill=text_color)
    image = np.array(image)


def true_false(x_finger, y_finger):
    answer_idx = ''
    global counter, selectionSpeed, btn_size, eyeName, userID, nowDatetime, eye, eyeText, idx, List
    if (abs(x_finger - 410) < btn_size) & (abs(y_finger - 360) < btn_size):
        counter += 1
        cv2.ellipse(image, (410, 360), (btn_size, btn_size), 0, 0, counter * selectionSpeed, (255, 0, 255), 20)
        if counter * selectionSpeed > 360:
            answer = 'left'
            counter = 0
            if eyeName == answer:  # 이름이 틀렸을때 똑같은 이미지 테스트
                answer_idx = 1
                List.append({'ID': userID, '시간': nowDatetime, '눈': eye, '시력': eyeText, '여부': answer_idx})
                print('true', eyeText)
            elif eyeName != answer:  # 이름이 틀렸을때 똑같은 이미지 테스트
                answer_idx = 0
                List.append({'ID': userID, '시간': nowDatetime, '눈': eye, '시력': eyeText, '여부': answer_idx})
                print('false', eyeText)
    elif (abs(x_finger - 850) < btn_size) & (abs(y_finger - 360) < btn_size):
        counter += 1
        cv2.ellipse(image, (850, 360), (btn_size, btn_size), 0, 0, counter * selectionSpeed, (255, 0, 255), 20)
        if counter * selectionSpeed > 360:
            answer = 'right'
            counter = 0
            if eyeName == answer:  # 이름이 틀렸을때 똑같은 이미지 테스트
                answer_idx = 1
                List.append({'ID': userID, '시간': nowDatetime, '눈': eye, '시력': eyeText, '여부': answer_idx})
                print('true', eyeText)
            elif eyeName != answer:  # 이름이 틀렸을때 똑같은 이미지 테스트
                answer_idx = 0
                List.append({'ID': userID, '시간': nowDatetime, '눈': eye, '시력': eyeText, '여부': answer_idx})
                print('false', eyeText)
    elif (abs(x_finger - 630) < btn_size) & (abs(y_finger - 135) < btn_size):
        counter += 1
        cv2.ellipse(image, (630, 135), (btn_size, btn_size), 0, 0, counter * selectionSpeed, (255, 0, 255), 20)
        if counter * selectionSpeed > 360:
            answer = 'up'
            counter = 0
            if eyeName == answer:  # 이름이 틀렸을때 똑같은 이미지 테스트
                answer_idx = 1
                List.append({'ID': userID, '시간': nowDatetime, '눈': eye, '시력': eyeText, '여부': answer_idx})
                print('true', eyeText)
            elif eyeName != answer:  # 이름이 틀렸을때 똑같은 이미지 테스트
                answer_idx = 0
                List.append({'ID': userID, '시간': nowDatetime, '눈': eye, '시력': eyeText, '여부': answer_idx})
                print('false', eyeText)
    elif (abs(x_finger - 630) < btn_size) & (abs(y_finger - 585) < btn_size):
        counter += 1
        cv2.ellipse(image, (630, 585), (btn_size, btn_size), 0, 0, counter * selectionSpeed, (255, 0, 255), 20)
        if counter * selectionSpeed > 360:
            answer = 'down'
            counter = 0
            if eyeName == answer:  # 이름이 틀렸을때 똑같은 이미지 테스트
                answer_idx = 1
                List.append({'ID': userID, '시간': nowDatetime, '눈': eye, '시력': eyeText, '여부': answer_idx})
                print('true', eyeText)
            elif eyeName != answer:  # 이름이 틀렸을때 똑같은 이미지 테스트
                answer_idx = 0
                List.append({'ID': userID, '시간': nowDatetime, '눈': eye, '시력': eyeText, '여부': answer_idx})
                print('false', eyeText)
    else:
        pass
    return answer_idx


def final_answer():
    global idx, another, testEnd, answer_idx
    if answer_idx == 1:
        if idx == 13:
            if List[-1]['눈'] == '오른쪽눈':
                final_List = List[-1]
                finalList.append(final_List)
                another = True
            elif List[-1]['눈'] == '왼쪽눈':
                final_List = List[-1]
                finalList.append(final_List)
                testEnd = True
        if len(List) < 13:
            idx += 1
            if idx > 13:
                idx = 13
            elif len(List) >= 2:
                if (List[-1]['여부'] == 1) & (List[-2]['여부'] == 1):
                    idx += 1
                if idx > 13:
                    idx = 13

        elif len(List) == 13:
            if List[-1]['눈'] == '오른쪽눈':
                final_List = List[-1]
                finalList.append(final_List)
                another = True
            elif List[-1]['눈'] == '왼쪽눈':
                final_List = List[-1]
                finalList.append(final_List)
                testEnd = True
    elif answer_idx == 0:
        idx -= 1
        if idx < 0:
            idx = 0
        if len(List) >= 3:
            if (List[-1]['여부'] == 0) & (List[-2]['여부'] == 0) & (List[-3]['여부'] == 0):
                if List[-1]['눈'] == '오른쪽눈':
                    final_List = List[-1]
                    finalList.append(final_List)
                    another = True
                elif List[-1]['눈'] == '왼쪽눈':
                    final_List = List[-1]
                    finalList.append(final_List)
                    testEnd = True
        elif len(List) >= 2:  # 마지막 2개가 틀렸을대 1개 전단계 이미지를 띄운다.
            if (List[-1]['여부'] == 0) & (List[-2]['여부'] == 0):
                idx -= 1
                if idx < 0:  # 시력 0.1 3번 틀렸을때 바로 출력
                    idx = 0
        elif len(List) == 13:
            if List[-1]['눈'] == '오른쪽눈':
                final_List = List[-1]
                finalList.append(final_List)
                another = True
            elif List[-1]['눈'] == '왼쪽눈':
                final_List = List[-1]
                finalList.append(final_List)
                testEnd = True
    return idx, another, testEnd

@bp.route('/eyetest')
def camera():
    return render_template('test/eyetest.html')

def gen(cap):
    global testEnd, another, image, idx, eye, List, answer_idx, eyeName, eyeText
    cap = cv2.VideoCapture(0)
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
            et.overlay(image, *(50, 50), 40, 40, logo)
            if testEnd is False:
                if faces:
                    face = faces[0]
                    # 거리측정코드
                    pointLeft = face[145]
                    pointRight = face[374]
                    w, _ = detector.findDistance(pointLeft, pointRight)
                    W = 6.3
                    f = 840
                    d = (W * f) / w
                    if another is False:
                        text_def((100, 35), f'{d_start}~{d_end}cm 사이에서 시력테스트 하세요.', font, (0, 0, 0))

                        # 60~70 중코드가 보인다
                        if d_start < int(d) <= d_end:
                            singleHeight = 50
                            d_color = (255, 0, 255)
                            eyeText, eyeImg, eyeName = eyeTes_def(idx)
                            image_def(eyeImg)

                            if results.multi_hand_landmarks:
                                for hand_landmarks in results.multi_hand_landmarks:
                                    finger = hand_landmarks.landmark[8]
                                    h, w, _ = image.shape
                                    x_finger, y_finger = int(finger.x * w), int(finger.y * h)
                                    cv2.circle(image, (x_finger, y_finger), 20, (255, 0, 0), 2, cv2.LINE_AA)  # 파란색
                                    answer_idx = true_false(x_finger, y_finger)
                                    timeStart = time.time()
                                    idx, another, testEnd = final_answer()
                        else:
                            d_color = (200, 200, 200)
                        cvzone.putTextRect(image, f' {int(d)}cm ', (230, 149), scale=2, colorR=d_color)
                        et.overlay(image, *(410, 360), 40, 40, btn_right)  # 왼쪽
                        et.overlay(image, *(850, 360), 40, 40, btn_left)  # 오른쪽
                        et.overlay(image, *(630, 135), 40, 40, btn_up)  # 위쪽
                        et.overlay(image, *(630, 585), 40, 40, btn_down)  # 아래쪽

                        et.overlay(image, *(150, 630), 150, 120, test)  # 횟수창
                        text_def((35, 607), f'{eye}: {len(List)}', ImageFont.truetype('../static/fonts/H2GSRB.TTF', 30),
                                 (0, 0, 0))
                    else:
                        eyeText, eyeImg, eyeName = eyeTes_def(idx)
                        print(eyeName)
                        image_def(background)
                        eye = '왼쪽눈'
                        text_def((370, 250), '테스트 계속하기', ImageFont.truetype('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/fonts/H2GSRB.TTF', 70),
                                 (0, 0, 0))
                        text_def((350, 380), f'{int(6 - (time.time() - timeStart))}초후 {eye} 테스트 시작합니다.',
                                 ImageFont.truetype('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/fonts/H2GSRB.TTF', 40), (0, 0, 0))
                        List = []
                        if int(6 - (time.time() - timeStart)) == 0:
                            another = False

            else:
                image_def(background)
                text_def((320, 220), "시력테스트 검사결과 ", ImageFont.truetype('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/fonts/H2GSRB.TTF', 70), (0, 0, 0))
                text_def((350, 360), f"{finalList[0]['눈']}의 시력은 {finalList[0]['시력']} 입니다 ", font, (0, 0, 0))
                text_def((350, 450), f"{finalList[1]['눈']}의 시력은 {finalList[1]['시력']} 입니다 ", font, (0, 0, 0))
                text_def((410, 590), f'{int(11 - (time.time() - timeStart))}초후 테스트 종료합니다.',
                ImageFont.truetype('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/fonts/H2GSRB.TTF', 40), (0, 0, 0))
                if int(11 - (time.time() - timeStart)) == 0:
                    break

            et.overlay(image, *(50, 50), 40, 40, logo)

            cv2.imshow("Eyetest", image)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    df = pd.DataFrame(finalList)
    df.to_csv('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/csv/시력테스트.csv')
    print(eyeName)
    print(finalList)
    cv2.destroyAllWindows()
    cap.release()

@bp.route('/video_feed')
def video_feed():
    global cap
    if Response(gen(cap),mimetype='multipart/x-mixed-replace; boundary=frame'):
        return render_template("test/eyetest.html")    # 윈도우창이 출력시 카메라 페이지로 다시 돌아간다