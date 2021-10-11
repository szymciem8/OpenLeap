import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

FOCAL_LEN = 200
DIST_AT_F_LEN = 116000

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#Choose camera
cap = cv2.VideoCapture(0)

#Find if the hand is right of left
def get_label(index, hand, results):
    output=None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            
            label = classification.classification[0].label
            score = classification.classification[0].score
            #text = '{} {}'.format(label, round(score, 2))
            text = label
            
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                         [640, 480]).astype(int))
            
            output = text, coords
        
    return output

#Get position of given hand landmark
def get_position(landmark_idx = mp_hands.HandLandmark.INDEX_FINGER_TIP):
    #TODO add right or left hand detection. 
    #Choose proper index instead of fixed one (idx=0)
    x = results.multi_hand_landmarks[0].landmark[landmark_idx].x
    y = results.multi_hand_landmarks[0].landmark[landmark_idx].y
    z = results.multi_hand_landmarks[0].landmark[landmark_idx].y

    return x, y, z

#Calculate distance between to hand landmarks
def get_distance(x1, y1, x2, y2):
    #TODO Add calibration using Z coordinate
    # Z coordinate is roghly proprtional to X coordinate 
    distance = ((x1-x2)**2 + (y1-y2)**2)**0.5
    return distance

def get_distance_bettween_landmarks(landmark_1, landmark_2, screen_size):
    x1, y1, z1 = get_position(landmark_1)
    x2, y2, z2 = get_position(landmark_2)

    #x1 *= screen_size[0] 
    #x2 *= screen_size[0]

    #y1 *= screen_size[1] 
    #y2 *= screen_size[1]

    distance = ((x1-x2)**2 + (y1-y2)**2)**0.5
    z = (z1+z2)/2
    #z1 *= screen_size[0]
    #distance = (2*z1**2)/distance

    #distance = ((4*z1**2)*distance) / (distance**2 + 4*z1**2)
    #distance = (FOCAL_LEN * distance)/(1/z1)
    distance = distance * z1
    print(distance)
    #print("z1:",z1)
    #print('z1', z1, 'dist', distance)

    return distance
    
hand_type = None
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        
        #BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #Flip horizontal
        image = cv2.flip(image, 1)
        
        #Set flag
        image.flags.writeable = False
        
        #Detections
        results = hands.process(image)
        
        #Set flag back to True
        image.flags.writeable = True
        
        #RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        

        #Rendering results
        if results.multi_hand_landmarks:
            n_hands = len(results.multi_hand_landmarks)
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                         mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4), 
                                         mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4))

                dist = get_distance_bettween_landmarks(mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.THUMB_TIP, (640, 480))
                #print(dist)
                # if n_hands > 1:
                #     if get_label(num, hand, results):
                #         hand_type, coord = get_label(num, hand, results)
                #         cv2.putText(image, hand_type, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                #         #print(hand_type)
                                
                #     if hand_type == "Right":
                #         x1, y1, z1 = get_position(mp_hands.HandLandmark.INDEX_FINGER_TIP)
                #         #cv2.putText(image, 'INDEX', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        
                #         x2, y2, z2 = get_position(mp_hands.HandLandmark.THUMB_TIP)
                #         #cv2.putText(image, 'THUMB', (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        
                #         #Draw line that connects index finger tip and thumb tip
                #         cv2.line(image, (x1, y1), (x2, y2), (255, 125, 100), 2)

                #         #Calculate the distance between those two points
                #         dist = str(int(get_distance(x1, y1, x2, y2))) 
                #         cv2.putText(image, dist, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                #         #cv2.putText(image, str(z1), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                # else:
                #         x1, y1, z1 = get_position(mp_hands.HandLandmark.INDEX_FINGER_TIP)
                #         #cv2.putText(image, 'INDEX', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        
                #         x2, y2, z2 = get_position(mp_hands.HandLandmark.THUMB_TIP)
                #         #cv2.putText(image, 'THUMB', (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        
                #         #Draw line that connects index finger tip and thumb tip
                #         cv2.line(image, (x1, y1), (x2, y2), (255, 125, 100), 2)

                #         #Calculate the distance between those two points
                #         dist = str(int(get_distance(x1, y1, x2, y2))) 
                #         cv2.putText(image, dist, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
        #image = cv2.flip(image, 0)
        cv2.imshow("Hand Tracking", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()