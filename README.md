# OpenLeap

## Table of contents
* [General Info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General Info
OpenLeap is an open source project that allows you to add hand gesture control to your Python projects. 

## Technologies

Project was created with technologies:

- Python
- OpenCV
- MediaPipe
- SciKit Learn

## Setup
OpenLeap can be installed using pip, as shown below.

```
$ pip install openleap
```

## Simple Example

Test openleap controller with an example program. Code below will create an instance of opencv window with feed from the camera. 



```
import openleap

controller = openleap.OpenLeap(screen_show=True, screeen_type='BLACK', show_data_on_image=True, gesture_model='sign_language')

controller.loop()

```

OpenLeap object can be created with couple of options. 
- **screen_show** - if set to True, window with camera feed will be created. 
- **screen_type** - "CAM" or "BLACK" background. 
- **show_data_on_image** - descriptive
- **show_data_in_console** - descriptive
- **gesture_model** - chose gesture recognition model, 'basic' or 'sign_language'

## Access hand information

Recognized gestures, hand position, tilt and so on are stored in a dictionary called 'data' that consists of two dataclass objects for right and left hand. Dataclass object is of given structure:

```
@dataclass
class Data:
    x : float = 0
    y : float = 0
    z : float = 0
    distance: float = 0.0
    angle: float = 0.0
    gesture: str = None
```

Dataclass containing all of the data above is continuously being updated in **main()** or **loop()** function depending on which one is being used. 