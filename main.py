import mediapipe as mp
import cv2
import numpy as np
import math
from dataclasses import dataclass
import math
import uuid
import os

#Initiate camera
cap = cv2.VideoCapture(0)

#CONSTANTS
FOCAL_LEN = 200
DIST_AT_F_LEN = 116000

WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
DIMENSIONS = [WIDTH, HEIGHT]

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

background = np.zeros([HEIGHT,WIDTH,3], dtype=np.uint8)
background.fill(0)


#Dataclass that describes position and other vales of each hand
@dataclass
class Data:
    x : int = 0
    y : int = 0
    z : int = 0
    angle: float = 0.0
    distance: float = 0.0

data = {
    'right' : Data(), 
    'left' : Data()
    }


#Find if the hand is right of left
def left_or_right(index, hand, results, mode='AI'):
    """
    Recognizes if visible hands (or hand) are left or right. 

    parameters: int, mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList, type, string

    returns: tuple (label, coordinates)

    """

    output=None
    label = 'right'

    coords = np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y))

    if mode == 'AI':
        for idx, classification in enumerate(results.multi_handedness):
            if classification.classification[0].index == index:

                label = classification.classification[0].label.lower()
                score = classification.classification[0].score
                #text = '{} {}'.format(label, round(score, 2))

                coords = tuple(np.multiply(coords, [640, 480]).astype(int))

    elif mode == 'position':
        #Get x values from both hands and compare
        if len(results.multi_handedness) >= 2:
            for i in [0, 1]:
                if index == i:
                    another_hand_x = results.multi_hand_landmarks[1-index].landmark[mp_hands.HandLandmark.WRIST].x
                    if coords[index] > another_hand_x:
                        label='right'
                    else:
                        label='left' 

        coords = tuple(np.multiply(coords, [640, 480]).astype(int))

    elif mode == 'auto':
        #If there is only one hand, use AI mode. For more than one use position mode.
        pass

    if output is None:
        label='right'

        coords = tuple(np.multiply(coords, [640, 480]).astype(int))

    output = label, coords

    return output


#Get position of given hand landmark, such as wrist or tip on an index finger
def get_position(results, index=0, landmark_idx = mp_hands.HandLandmark.INDEX_FINGER_TIP, dim=None):

    """
    Finds normalized position or of given hand landmark. Additionally, it can calculate the position on the screen
    if dimensions are give. 

    parameters: int, mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList, list

    returns: int, int, int
    """

    #TODO add right or left hand detection.
    x = results.multi_hand_landmarks[index].landmark[landmark_idx].x
    y = results.multi_hand_landmarks[index].landmark[landmark_idx].y
    z = results.multi_hand_landmarks[index].landmark[landmark_idx].y

    if dim is None:
        #Choose proper index instead of fixed one (idx=0)
        return x, y, z
    else:
        x = int(x*dim[0])
        y = int(y*dim[1])
        z = int(z*dim[0])

    return x, y, z

#Calculate distance between two different hand landmarks
def get_distance_bettween_landmarks(results, index, landmark_1, landmark_2, dim):

    """
    Calculates distance between two given hand landmarks. 

    parameters: int, 
                mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList, 
                mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList, 
                list

    returns: float
    """

    x1, y1, z1 = get_position(results, 0, landmark_1, dim)
    x2, y2, z2 = get_position(results, 0, landmark_2, dim)

    x1 *= dim[0]
    x2 *= dim[0]

    y1 *= dim[1]
    y2 *= dim[1]

    distance = math.sqrt(((x1-x2)**2 + (y1-y2)**2))

    return distance

def get_angle(results, index, landmark_idx, mode='half', unit='radians'):
    """
    Calculates angle using atan2 with wrist as a base. 

    parameters:

    returns:
    """

    angle=0

    wrist_x = results.multi_hand_landmarks[index].landmark[mp_hands.HandLandmark.WRIST].x
    wrist_y = results.multi_hand_landmarks[index].landmark[mp_hands.HandLandmark.WRIST].y

    landmark_x = results.multi_hand_landmarks[index].landmark[landmark_idx].x
    landmark_y = results.multi_hand_landmarks[index].landmark[landmark_idx].y

    realitive_x = landmark_x - wrist_x
    realitive_y = landmark_y - wrist_y

    angle = math.atan2(realitive_y, realitive_x)

    if mode=='half' and angle>0: angle=0

    if unit=='degree':
        angle = 180*abs(angle)/math.pi

    return angle

def show_data():
    """
    Shows collected and calculated data in opencv window. 
    """
    pass

def main():

    """
    Main function that runs the core of the program. 
    """

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
                for index, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(background, hand, mp_hands.HAND_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4))

                    
                    
                    if n_hands > 0:
                        #If there are two hands
                        if left_or_right(index, hand, results, 'position'):
                            hand_type, coord = left_or_right(index, hand, results, 'position')
                            cv2.putText(background, hand_type, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                            x, y, z = get_position(results, index, mp_hands.HandLandmark.WRIST, DIMENSIONS)

                            data[hand_type].x = x
                            data[hand_type].y = y
                            data[hand_type].z = z
                            data[hand_type].distance = get_distance_bettween_landmarks(
                                                                                        results,
                                                                                        index, 
                                                                                        mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                                                                                        mp_hands.HandLandmark.THUMB_TIP, 
                                                                                        DIMENSIONS
                                                                                        )
                                                                                        
                            data[hand_type].angle = get_angle(results, index, mp_hands.HandLandmark.MIDDLE_FINGER_MCP, unit='degree')

                            cv2.putText(image, hand_type, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                            x1, y1, z1 = get_position(results, index, mp_hands.HandLandmark.INDEX_FINGER_TIP, DIMENSIONS)
                            x2, y2, z2 = get_position(results, index, mp_hands.HandLandmark.THUMB_TIP, DIMENSIONS)
                            
                            #Draw line that connects index finger tip and thumb tip
                            cv2.line(background, (x1, y1), (x2, y2), (255, 125, 100), 2)

                    print(data)

            cv2.imshow("Hand Tracking", background)
            background.fill(0)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__=='__main__':
    main()