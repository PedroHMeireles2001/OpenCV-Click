import cv2
import mediapipe as mp
import pyautogui as gui

mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
ML_modelo = mp_maos.Hands()

resolucao_x = 1280
resolucao_y = 720
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH,resolucao_x)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,resolucao_y)

dedo_levantado = False

def getLandmarkMaos(resultado):
    landmarks = []
    mp_desenho.draw_landmarks(img,resultado.multi_hand_landmarks[0],mp_maos.HAND_CONNECTIONS)
    for landmark in resultado.multi_hand_landmarks[0].landmark:
        landmarks.append(landmark)
    return landmarks

def getDedosLevantados(landmarks):
    dedos = []
    for ponta_dedo in [8,12,16,20]:
        landmark = landmarks[ponta_dedo]
        landmark_anterior = landmarks[ponta_dedo-2]
        dedos.append(landmark.y < landmark_anterior.y)
    return dedos
        

while True:
    # captura webcam
    sucess, img = camera.read()
    if sucess == False:
        continue

    img = cv2.flip(img,1)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    resultado = ML_modelo.process(img_rgb)
    if resultado.multi_hand_landmarks:
        landmarks = getLandmarkMaos(resultado)
        dedos = getDedosLevantados(landmarks)
        if sum(dedos) == 1 and dedo_levantado == False:
            gui.click()
            dedo_levantado = True
        if dedos == [False,False,False,False]:
            dedo_levantado = False
            

    cv2.imshow("Webcam",img)
    tecla = cv2.waitKey(1)
    if tecla == 27:
        break

