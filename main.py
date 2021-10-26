import mediapipe as mp
import cv2
import numpy as np
import math
from dataclasses import dataclass
import math
import pickle
import pandas as pd
import warnings

#Initiate camera
cap = cv2.VideoCapture(0)

#OPTIONS
SCREEN_SHOW=True
SCREEN_TYPE='CAM' # black or cam

#CONSTANTS
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
DIMENSIONS = [WIDTH, HEIGHT]

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#Dataclass that describes position and other values of each hand
@dataclass
class Data:
    x : int = 0
    y : int = 0
    z : int = 0
    distance: float = 0.0
    angle: float = 0.0
    gesture: str = None

data = {
    'right' : Data(), 
    'left' : Data()
    }


def left_or_right(index, hand, results, mode='AI'):
    """
    Recognizes if visible hands (or hand) are left or right. 

    parameters: int, mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList, type, str

    returns: str
    """

    label = 'right'

    coords = np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y))

    if mode == 'AI':
        for idx, classification in enumerate(results.multi_handedness):
            if classification.classification[0].index == index:

                label = classification.classification[0].label.lower()
                #core = classification.classification[0].score
                #text = '{} {}'.format(label, round(score, 2))

                return label


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

                    return label

        else:
            return left_or_right(index, hand, results, mode='AI')

    # if output is None:
    #     label='right'
    #     coords = tuple(np.multiply(coords, DIMENSIONS).astype(int))

    # output = label, coords

    return label


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

def get_distance_bettween_landmarks(results, index, landmark_1, landmark_2, normalized=True):
    """
    Calculates distance between two given hand landmarks. 

    parameters: int, 
                mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList, 
                mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList, 
                list, 
                boolean

    returns: float
    """

    x1, y1, z1 = get_position(results, index, landmark_1, DIMENSIONS)
    x2, y2, z2 = get_position(results, index, landmark_2, DIMENSIONS)

    x1 *= WIDTH
    x2 *= WIDTH

    y1 *= HEIGHT
    y2 *= HEIGHT

    distance = math.sqrt(((x1-x2)**2 + (y1-y2)**2))

    return distance

def get_angle(results, index, landmark_idx, mode='half', unit='radians'):
    """
    Calculates angle using atan2 with wrist as a base. 

    parameters: list, int, int, str, str

    returns: float
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


def show_data(data, img=None, console=True, on_image=True):
    """
    Shows collected and calculated data in opencv window. 

    parameters: dictionary, 2D list, Boolean, Boolean

    returns: None
    """

    if console:
        for index, (key, d) in enumerate(data.items()):
            print('%s hand:\tX=%d,\tY=%d,\tZ=%d,\tdistance=%.2f,\tangle=%f,\tgesture="%s"' %(key.capitalize(), d.x, d.y, d.z, d.distance, d.angle, d.gesture))

    
    if on_image:
        for index, (key, d) in enumerate(data.items()):
            i=1
            for field in d.__dataclass_fields__:
                value = getattr(d, field)
                #print(field, value)

                if field == 'distance' or field == 'angle':
                    #vars(d)['distnace'] = "{:.2f}".format(value)
                    text = "%s = %s" %(field, str(int(value)))
                    cv2.putText(img, text, (10+(1-index)*400, 25*i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    text = "%s = %s" %(field, str(value))
                    cv2.putText(img, text, (10+(1-index)*400, 25*i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
                i+=1
    

def get_gesture(results, model, index):
    """
    Returns rocognized gesture based on pretrained model.

    parameters: list, sklearn.pipleine.Pipeline, int

    returns: str
    """

    hand_landmarks = results.multi_hand_landmarks[index].landmark
    hand_coords = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks]).flatten())

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        x = pd.DataFrame([hand_coords])
        gesture_class = model.predict(x)[0]

    return gesture_class


def main(model):
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

            if SCREEN_SHOW:
                if SCREEN_TYPE=='BLACK':
                    background = np.zeros([HEIGHT,WIDTH,3], dtype=np.uint8)
                    background.fill(0)
                elif SCREEN_TYPE == 'CAM':
                    background = image

            #Rendering results
            if results.multi_hand_landmarks:
                n_hands = len(results.multi_hand_landmarks)
                for index, hand in enumerate(results.multi_hand_landmarks):
        
                    if n_hands >= 1:
                        #If there are two hands
                        if left_or_right(index, hand, results, 'position'):
                            hand_type = left_or_right(index, hand, results, 'position')
                            x, y, z = get_position(results, index, mp_hands.HandLandmark.WRIST, DIMENSIONS)

                            data[hand_type].x = x
                            data[hand_type].y = y
                            data[hand_type].z = z
                            data[hand_type].distance = get_distance_bettween_landmarks(
                                                                                        results,
                                                                                        index, 
                                                                                        mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                                                                                        mp_hands.HandLandmark.THUMB_TIP
                                                                                        )
                                                                                    
                            data[hand_type].angle = get_angle(results, index, mp_hands.HandLandmark.MIDDLE_FINGER_MCP, unit='degree')
                            data[hand_type].gesture = get_gesture(results, model, index)
                            
                            x1, y1, z1 = get_position(results, index, mp_hands.HandLandmark.INDEX_FINGER_TIP, DIMENSIONS)
                            x2, y2, z2 = get_position(results, index, mp_hands.HandLandmark.THUMB_TIP, DIMENSIONS)

                            if SCREEN_SHOW:

                                #Show on screen
                                cv2.putText(background, hand_type, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                                #Draw line that connects index finger tip and thumb tip
                                cv2.line(background, (x1, y1), (x2, y2), (255, 125, 100), 2)

                                mp_drawing.draw_landmarks(background, hand, mp_hands.HAND_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4))

                                show_data(data, background)

                            else:
                                show_data(data, on_image=False)

            if SCREEN_SHOW:
                cv2.imshow("Hand Tracking", background)
                background.fill(0)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__=='__main__':
    with open('gesture_recognition.pkl', 'rb') as f:
        model = pickle.load(f)

    main(model)