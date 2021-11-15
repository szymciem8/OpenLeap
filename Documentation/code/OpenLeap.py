import mediapipe as mp
import cv2
import numpy as np
import math
from dataclasses import dataclass
import math
import pickle
import pandas as pd
import warnings
import os


class OpenLeap():

    #Dataclass that describes position and other values of each hand
    @dataclass
    class Data:
        x : float = 0
        y : float = 0
        z : float = 0
        distance: float = 0.0
        angle: float = 0.0
        gesture: str = None

    def __init__(self, 
                SCREEN_SHOW=False, 
                SCREEN_TYPE='BLACK', 
                SHOW_DATA_IN_CONSOLE=False, 
                SHOW_DATA_ON_IMAGE=False, 
                NORMALIZED_POSITION=True,
                GESTURE_MODEL='sign_language'
        ):

        super().__init__()

        self.data = {
                'right' : self.Data(), 
                'left' : self.Data()
                }

        if GESTURE_MODEL=='basic':
            file_name='gesture_recognition.pkl'
        elif GESTURE_MODEL=='sign_language':
            file_name='sign_language_alphabet.pkl'

        this_dir, this_filename = os.path.split(__file__)  # Get path of data.pkl
        data_path = os.path.join(this_dir, file_name)
        with open(data_path, 'rb') as f:
            self.model = pickle.load(f)

        #Initiate camera
        self.cap = cv2.VideoCapture(0)

        #OPTIONS
        self.SCREEN_SHOW=SCREEN_SHOW
        self.SCREEN_TYPE=SCREEN_TYPE # black or cam

        self.SHOW_DATA_IN_CONSOLE=SHOW_DATA_IN_CONSOLE
        self.SHOW_DATA_ON_IMAGE=SHOW_DATA_ON_IMAGE

        self.NORMALIZED_POSITION=NORMALIZED_POSITION

        #CONSTANTS
        self.WIDTH = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.HEIGHT = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.DIMENSIONS = [self.WIDTH, self.HEIGHT]

        #MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
        self.results = None

        self.relative_position = {}
        self.relative_position['right'] = np.zeros((21,3))
        self.relative_position['left'] = np.zeros((21,3))


    def close_window(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def detect_key(self, key):
        if cv2.waitKey(10) & 0xFF == ord(key):
            return True

        return False

    def left_or_right(self, index, hand, mode='AI'):
        """
        Recognizes if visible hands (or hand) are left or right. 

        parameters: int, mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList, type, str

        returns: str
        """

        label = 'right'

        coords = np.array((hand.landmark[self.mp_hands.HandLandmark.WRIST].x, hand.landmark[self.mp_hands.HandLandmark.WRIST].y))

        if mode == 'AI':
            for idx, classification in enumerate(self.results.multi_handedness):
                if classification.classification[0].index == index:

                    label = classification.classification[0].label.lower()
                    #core = classification.classification[0].score
                    #text = '{} {}'.format(label, round(score, 2))

                    return label


        elif mode == 'position':
            #Get x values from both hands and compare
            if len(self.results.multi_handedness) >= 2:
                for i in [0, 1]:
                    if index == i:
                        another_hand_x = self.results.multi_hand_landmarks[1-index].landmark[self.mp_hands.HandLandmark.WRIST].x
                        if coords[index] > another_hand_x:
                            label='right'
                        else:
                            label='left' 

                        return label

            else:
                return self.left_or_right(index, hand, mode='AI')


        return label


    def get_position(self, index=0, landmark_idx=1, normalized=False):
        """
        Finds normalized position or of given hand landmark. Additionally, it can calculate the position on the screen
        if dimensions are give. 

        parameters: int, mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList, list

        returns: int, int, int
        """

        #TODO add right or left hand detection.
        x = self.results.multi_hand_landmarks[index].landmark[landmark_idx].x
        y = self.results.multi_hand_landmarks[index].landmark[landmark_idx].y
        z = self.results.multi_hand_landmarks[index].landmark[landmark_idx].y

        if normalized:
            #Choose proper index instead of fixed one (idx=0)
            return x, y, z
        else:
            x = int(x*self.WIDTH)
            y = int(y*self.HEIGHT)
            z = int(z*self.WIDTH)

        return x, y, z

    def get_distance_bettween_landmarks(self, index, landmark_1, landmark_2, normalized=True):
        """
        Calculates distance between two given hand landmarks. 

        parameters: int, 
                    mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList, 
                    mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList, 
                    list, 
                    boolean

        returns: float
        """
        if normalized:
            x1, y1, z1 = self.get_position(index, landmark_1, normalized=True)
            x2, y2, z2 = self.get_position(index, landmark_2, normalized=True)
        else:
            x1, y1, z1 = self.get_position(index, landmark_1)
            x2, y2, z2 = self.get_position(index, landmark_2)

        distance = math.sqrt(((x1-x2)**2 + (y1-y2)**2))

        return distance

    def get_angle(self, index, landmark_idx, mode='half', unit='radians'):
        """
        Calculates angle using atan2 with wrist as a base. 

        parameters: list, int, int, str, str

        returns: float
        """

        angle=0

        wrist_x = self.results.multi_hand_landmarks[index].landmark[self.mp_hands.HandLandmark.WRIST].x
        wrist_y = self.results.multi_hand_landmarks[index].landmark[self.mp_hands.HandLandmark.WRIST].y

        landmark_x = self.results.multi_hand_landmarks[index].landmark[landmark_idx].x
        landmark_y = self.results.multi_hand_landmarks[index].landmark[landmark_idx].y

        realitive_x = landmark_x - wrist_x
        realitive_y = landmark_y - wrist_y

        angle = math.atan2(realitive_y, realitive_x)

        if mode=='half' and angle>0: angle=0

        if unit=='degree':
            angle = 180*abs(angle)/math.pi

        return angle


    def show_data(self, image=None, console=False, on_image=False):
        """
        Shows collected and calculated data in opencv window.

        parameters: dictionary, 2D list, Boolean, Boolean

        returns: None
        """

        if console:
            for index, (key, d) in enumerate(self.data.items()):
                print('%s hand:\tX=%f,\tY=%f,\tZ=%f,\tdistance=%.2f,\tangle=%f,\tgesture="%s"' %(key.capitalize(), d.x, d.y, d.z, d.distance, d.angle, d.gesture))

        
        if on_image:
            for index, (key, d) in enumerate(self.data.items()):
                i=1
                for field in d.__dataclass_fields__:
                    value = getattr(d, field)
                    #print(field, value)

                    if field == 'distance' or field == 'angle':
                        #vars(d)['distnace'] = "{:.2f}".format(value)
                        text = "%s = %s" %(field, str(int(value)))
                        cv2.putText(image, text, (10+(1-index)*400, 25*i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        text = "%s = %s" %(field, str(value))
                        cv2.putText(image, text, (10+(1-index)*400, 25*i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
                    i+=1
    
    def get_gesture(self, index):
        """
        Returns rocognized gesture based on pretrained model.

        parameters: list, sklearn.pipleine.Pipeline, int

        returns: str
        """

        hand_landmarks = self.results.multi_hand_landmarks[index].landmark
        wrist = hand_landmarks[0]
            
        hand_landmarks_row = np.zeros((20,3))
        for i in range(1, len(hand_landmarks)):
            hand_landmarks_row[i-1]=[hand_landmarks[i].x-wrist.x, hand_landmarks[i].y-wrist.y, hand_landmarks[i].z-wrist.z]
            
        hand_landmarks_row = hand_landmarks_row.flatten()
        hand_landmarks_row = list(hand_landmarks_row/np.max(np.absolute(hand_landmarks_row)))
        
        #Make Detections
        x = pd.DataFrame([hand_landmarks_row])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            x = pd.DataFrame([hand_landmarks_row])
            gesture_class = self.model.predict(x)[0]

        return gesture_class


    def main(self):
        """
        Main function that runs the core of the program. 
        """

        hand_type = None
        if self.cap.isOpened():
            ret, frame = self.cap.read()

            #BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #Flip horizontal
            image = cv2.flip(image, 1)

            #Set flag
            image.flags.writeable = False

            #Detections
            self.results = self.hands.process(image)

            #Set flag back to True
            image.flags.writeable = True

            #RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if self.SCREEN_SHOW:
                if self.SCREEN_TYPE=='BLACK':
                    background = np.zeros([self.HEIGHT,self.WIDTH,3], dtype=np.uint8)
                    background.fill(0)
                elif self.SCREEN_TYPE == 'CAM':
                    background = image

            #Rendering results
            if self.results.multi_hand_landmarks:
                n_hands = len(self.results.multi_hand_landmarks)
                for index, hand in enumerate(self.results.multi_hand_landmarks):
        
                    if n_hands >= 1:
                        #If there are two hands
                        if self.left_or_right(index, hand, 'position'):
                            hand_type = self.left_or_right(index, hand, 'position')
                            x, y, z = self.get_position(index, self.mp_hands.HandLandmark.WRIST, normalized=self.NORMALIZED_POSITION)

                            self.data[hand_type].x = x
                            self.data[hand_type].y = y
                            self.data[hand_type].z = z
                            self.data[hand_type].distance = self.get_distance_bettween_landmarks(
                                                                                        index, 
                                                                                        self.mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                                                                                        self.mp_hands.HandLandmark.THUMB_TIP, 
                                                                                        normalized=False
                                                                                        )
                                                                                    
                            self.data[hand_type].angle = self.get_angle(index, self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP, unit='degree')
                            self.data[hand_type].gesture = self.get_gesture(index)

                            if self.SCREEN_SHOW:

                                x1, y1, z1 = self.get_position(index, self.mp_hands.HandLandmark.INDEX_FINGER_TIP, normalized= not self.NORMALIZED_POSITION)
                                x2, y2, z2 = self.get_position(index, self.mp_hands.HandLandmark.THUMB_TIP, normalized= not self.NORMALIZED_POSITION)

                                #Draw line that connects index finger tip and thumb tip
                                cv2.line(background, (x1, y1), (x2, y2), (255, 125, 100), 2)

                                self.mp_drawing.draw_landmarks(background, hand, self.mp_hands.HAND_CONNECTIONS,
                                            self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
                                            self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4))
                                
                                if self.SHOW_DATA_ON_IMAGE:
                                    #Show on screen
                                    x, y, z = self.get_position(index, self.mp_hands.HandLandmark.WRIST, normalized= not self.NORMALIZED_POSITION)
                                    cv2.putText(background, hand_type, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                                self.show_data(background, console=self.SHOW_DATA_IN_CONSOLE, on_image=self.SHOW_DATA_ON_IMAGE)

                            self.show_data(console=self.SHOW_DATA_IN_CONSOLE, on_image=self.SHOW_DATA_ON_IMAGE)

            if self.SCREEN_SHOW:
                cv2.imshow("Hand Tracking", background)
                background.fill(0)

    def loop(self):
        '''
        Runs main function perpetually. 
        '''

        while True:
            self.main()

            if self.detect_key('q'):
                self.close_window()
                break

if __name__=='__main__':
    '''
    Use example of OpenLeap object. 
    '''

    controller = OpenLeap(SHOW_DATA_IN_CONSOLE=False, SCREEN_SHOW=True, SCREEN_TYPE='CAM', SHOW_DATA_ON_IMAGE=True)

    while True:
        controller.main()
        print(controller.relative_position['right'])
        if controller.detect_key('q'):
            controller.close_window()
            break