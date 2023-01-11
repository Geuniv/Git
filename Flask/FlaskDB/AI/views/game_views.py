from flask import render_template,Response,Blueprint , g
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
import random
import os

bp = Blueprint('game' , __name__ , url_prefix='/game')

@bp.route('/game')
def game():
    return render_template('test/game.html')

totalTime = 30  # 총 게임 시간은 20초로 지정
userID = '000000001'
eye_dist = 33

now = datetime.datetime.now()
nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

font = ImageFont.truetype('../AI/static/fonts/H2GSRB.TTF', 35)
small_font = ImageFont.truetype('../AI/static/fonts/H2GSRB.TTF', 10)
logo = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/assets/img/button/eye.png', cv2.IMREAD_UNCHANGED), (80, 80))

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = FaceMeshDetector(maxFaces=1)
mp_hands = mp.solutions.hands

detector = FaceMeshDetector(maxFaces=1)
idList = [0, 17, 78, 292]

# import images
folderBirds = 'C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/image/game'
ListBirds = os.listdir(folderBirds)
catch = []
for object in ListBirds:
    catch.append(cv2.resize(cv2.imread(f'{folderBirds}/{object}', cv2.IMREAD_UNCHANGED), (80, 80)))
background = cv2.resize(cv2.imread('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/assets/img/button/background.jpg'), (1000, 630))

currentObject = catch[0]
pos = [300, 128]
speed = random.randint(5, 10)
count = 0
global birds
birds = True
gameOver = False

ratioList = []
blinkCounter = 0
counter = 0
List = []
finalList = []
def_counter = 0
selectionSpeed = 8
btn_size = 40

timeStart = time.time()  # 시작 시간은 웹캠이 열리는 시점의 현재시간

def text_def(xy, text_name, fontstyle, text_color):
    global image
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    draw.text(xy=xy, text=text_name, font=fontstyle, fill=text_color)
    image = np.array(image)


def resetObject():
    global birds
    pos[0] = random.randint(0, 1180)
    pos[1] = 0
    randNo = random.randint(0, 1000000)  # change the ratio of eatables/ non-eatables
    currentObject = catch[random.randint(0, 3)]
    birds = True
    return currentObject

def gen(cap):
    global image, eye_dist, blinkCounter, timeStart, totalTime, gameOver, currentObject, eye_dist, counter, count,birds
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            success, image = cap.read()
            image = cv2.flip(image, 1)
            results = hands.process(image)
            et.overlay(image, *(50, 50), 40, 40, logo)
            if time.time() - timeStart < totalTime:
                if gameOver is False:
                    text_def((100, 35), '눈을 깜빡하면서 새를 잡아보세요.', font, (255, 255, 255))
                    image, faces = detector.findFaceMesh(image, draw=False)
                    image = cvzone.overlayPNG(image, currentObject, pos)
                    pos[1] += speed

                    if pos[1] > 520:
                        currentObject = resetObject()

                    if faces:
                        face = faces[0]
                        up = face[159]  # Lefteye
                        down = face[23]
                        left = face[130]
                        right = face[243]

                        upDown, _ = detector.findDistance(up, down)
                        leftRight, _ = detector.findDistance(left, right)

                        cx, cy = (up[0] + down[0]) // 2, (up[1] + down[1]) // 2
                        dist, _ = detector.findDistance((cx, cy), (pos[0] + 50, pos[1] + 50))

                        ratio = int((upDown / leftRight) * 100)
                        ratioList.append(ratio)
                        if len(ratioList) > 3:
                            ratioList.pop(0)
                        ratioAvg = sum(ratioList) / len(ratioList)
                        if ratioAvg < eye_dist and counter == 0:
                            eyeStatus = "Open"
                            blinkCounter += 1
                            color = (0, 200, 0)
                            counter = 1
                        if counter != 0:
                            eyeStatus = "Closed"
                            counter += 1
                            if counter > 10:
                                counter = 0
                                color = (255, 0, 255)
                                if birds:
                                    currentObject = resetObject()
                                    count += 1
                                    List.append({
                                        'ID': userID,
                                        '시간': nowDatetime,
                                        '점수': str(count)
                                    })
                                else:
                                    gameOver = True
                            cv2.putText(image, eyeStatus, (1050, 680), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
                        cvzone.putTextRect(image, f'Score:  {str(count)}', (50, 680), scale=2, colorR=(255, 0, 255))
                cv2.putText(image, f'Time: {int(totalTime - (time.time() - timeStart))}', (1000, 80), 1, 3,
                            (255, 0, 255), 3)
            else:
                h, w, _ = background.shape
                image[int(360 - h / 2):int(360 + h / 2), int(630 - w / 2):int(630 + w / 2)] = background

                cv2.putText(image, "Game Over", (320, 300), cv2.FONT_HERSHEY_PLAIN, 7, (255, 0, 255), 10)
                cv2.putText(image, f'Your Score: {str(count)}', (450, 400), 1, 3, (255, 0, 255), 4)

                text_def((400, 470), " R = 재 시 작 ", font, (0, 0, 0))
                text_def((710, 470), " Q =  종 료 ", font, (50, 50, 255))

            cv2.imshow("Game", image)
            key = cv2.waitKey(1)
            if key == ord('r'):
                timeStart = time.time()
                resetObject()
                gameOver = False
                count = 0
                currentObject = catch[0]
                birds = True

            elif key == ord('q'):
                break

    final_List = List[-1]
    finalList.append(final_List)
    df = pd.DataFrame(finalList)
    df.to_csv('C:/Users/user/Desktop/pythonProject/pythonProject/Git/Flask/FlaskDB/AI/static/csv/게임.csv')
    print(finalList)
    print('저장 완료됬습니다.')
    cap.release()
    cv2.destroyAllWindows()

@bp.route('/game_camera')
def video_feed():
    global cap
    if Response(gen(cap),mimetype='multipart/x-mixed-replace; boundary=frame'):
        return render_template("test/game.html")    # 윈도우창이 출력시 카메라 페이지로 다시 돌아간다