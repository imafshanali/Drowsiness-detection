from tkinter import Frame
import cv2
import mediapipe as mp
from scipy.spatial import distance
from pygame import mixer
import numpy as np

mixer.init()
mixer.music.load("music.wav")

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	
thresh = 0.25
frame_check = 20
flag = 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

cap = cv2.VideoCapture(0)

end_app = False
mouse_x, mouse_y = -1, -1

button_text = "End Application"
button_width, button_height = 160, 35
margin = 20
button_radius = 6
button_bg_color = (0, 0, 0)
button_alpha = 0.35
button_bg_color_hover = (50, 50, 50)
button_alpha_hover = 0.55
button_text_color = (255, 255, 255)

def draw_rounded_rectangle_with_alpha(img, top_left, bottom_right, radius, color, alpha):
    x1, y1 = top_left
    x2, y2 = bottom_right
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def mouse_callback(event, x, y, flags, param):
    global end_app, mouse_x, mouse_y
    mouse_x, mouse_y = x, y
    button_top_left = param["top_left"]
    button_bottom_right = param["bottom_right"]
    if event == cv2.EVENT_LBUTTONDOWN:
        if button_top_left[0] <= x <= button_bottom_right[0] and button_top_left[1] <= y <= button_bottom_right[1]:
            end_app = True

cv2.namedWindow("Frame")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    button_top_left = (w - button_width - margin, h - button_height - margin)
    button_bottom_right = (w - margin, h - margin)

    cv2.setMouseCallback("Frame", mouse_callback, param={
        "top_left": button_top_left,
        "bottom_right": button_bottom_right,
    })

    if end_app:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            leftEye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
            rightEye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

            leftEye_np = np.array(leftEye, dtype=np.int32)
            rightEye_np = np.array(rightEye, dtype=np.int32)

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            cv2.polylines(frame, [cv2.convexHull(leftEye_np)], True, (0, 255, 0), 1)
            cv2.polylines(frame, [cv2.convexHull(rightEye_np)], True, (0, 255, 0), 1)

            if ear < thresh:
                flag += 1
                print(flag)
                if flag >= frame_check:
                    cv2.putText(frame, "**********************WAKE UP ALERT!**********************", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "**********************WAKE UP ALERT!**********************", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not mixer.music.get_busy():
                        mixer.music.play()
            else:
                flag = 0

    if button_top_left[0] <= mouse_x <= button_bottom_right[0] and button_top_left[1] <= mouse_y <= button_bottom_right[1]:
        bg_color = button_bg_color_hover
        alpha = button_alpha_hover
    else:
        bg_color = button_bg_color
        alpha = button_alpha

    draw_rounded_rectangle_with_alpha(frame, button_top_left, button_bottom_right, button_radius, bg_color, alpha)

    (text_width, text_height), _ = cv2.getTextSize(button_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_x = button_top_left[0] + (button_width - text_width) // 2
    text_y = button_top_left[1] + (button_height + text_height) // 2 - 4
    cv2.putText(frame, button_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, button_text_color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
mixer.music.stop()