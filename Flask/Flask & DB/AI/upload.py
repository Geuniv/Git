# 라이브러리
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import datetime
from AI_model import cataract_predict as cp

app = Flask(__name__)
mp_face_detection = mp.solutions.face_detection  # 얼굴 검출
mp_drawing = mp.solutions.drawing_utils  # 얼굴 특징 표시

cap = cv2.VideoCapture(0)


# 메인 서버
@app.route("/")
def hello_world():
    return render_template("main.html")


# 메인카메라 서버
@app.route("/Camera")
def camera():
    return render_template("camera.html")


@app.route("/image_page")
def first():
    return render_template("image_page.html")


def gen(cap):
    # model_selection=0 -> 2m 내, model_selection=1 -> 5m 내
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection:
        while True:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
            success, image = cap.read()

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.detections:
                for detection in results.detections:
                    keypoints = detection.location_data.relative_keypoints
                    right_eye = keypoints[0]  # 왼쪽눈
                    left_eye = keypoints[1]  # 오른쪽눈

                    h, w, _ = image.shape
                    right = image[int(right_eye.y * h - 40):int(right_eye.y * h + 40),
                            int(right_eye.x * w - 60):int(right_eye.x * w + 40)]
                    left = image[int(left_eye.y * h - 40):int(left_eye.y * h + 40),
                           int(left_eye.x * w - 40):int(left_eye.x * w + 60)]

            now = datetime.datetime.now().strftime("%d_%H-%M-%S")
            right_name = './image_/' + 'right_image' + str(now) + ".jpg"
            left_name = './image_/' + 'left_image' + str(now) + ".jpg"
            eye_name = './image_/' + 'eye_image' + str(now) + ".jpg"
            cv2.imshow("Image", image)
            key = cv2.waitKey(1)

            if key == ord('q'):
                break

            elif key == ord('c'):
                print("Right_eye")
                cv2.imwrite(right_name, right)
                cp.image_test(right_name)
                print("Left_eye")
                cv2.imwrite(left_name, left)
                cp.image_test(left_name)

            elif key == ord('r'):
                print("Right_eye")
                cv2.imwrite(right_name, right)
                cp.image_test(right_name)

            elif key == ord('l'):
                print("Left_eye")
                cv2.imwrite(left_name, left)
                cp.image_test(left_name)

    cv2.destroyAllWindows()
    result = cp.image_test('./image_\eye_image18_14-29-55.jpg')
    return result


# 카메라 서버
@app.route('/video_feed')
def video_feed():
    global cap
    if Response(gen(cap), mimetype='multipart/x-mixed-replace; boundary=frame'):
        return render_template("camera.html")  # 윈도우창이 출력시 카메라 페이지로 다시 돌아간다


if __name__ == "__main__":
    app.run(debug=True)
