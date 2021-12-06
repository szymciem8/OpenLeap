# OpenLeap

## Table of contents
- [OpenLeap](#openleap)
  - [Table of contents](#table-of-contents)
  - [General Info](#general-info)
  - [Technologies](#technologies)
  - [Setup](#setup)
  - [Simple Example](#simple-example)
  - [Access Hand Information](#access-hand-information)
    - [Example](#example)
    - [Another Example](#another-example)

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

controller = openleap.OpenLeap(screen_show=True, 
                               screeen_type='BLACK', 
                               show_data_on_image=True,
                               show_data_in_console=True,
                               gesture_model='sign_language')

controller.loop()

```

<p align="center">
  <img src="https://raw.githubusercontent.com/szymciem8/OpenLeap/main/Documentation/images/example_program.gif?token=AMBI64BGASHC4OPJW6OD3YDBV2BJK" width="850" />
</p>

OpenLeap returns relative position of each hand, distance between thumb tip and index finger tip, rotation angle by wrist point and recognized gesture. There are two models for gesture recognition. 

The first one can recognized wheter hand is opened or closed into fist, second model can recognized sign language alphabet as shown below. 

<p align="center">
  <img src="https://pastevents.impactcee.com/wp-content/uploads/2016/10/DayTranslationsBlog-Learn-American-Sign-Language.jpg" width="850" />
</p>


OpenLeap object can be created with couple of options. 
- **screen_show** - if set to True, window with camera feed will be created. 
- **screen_type** - "CAM" or "BLACK" background. 
- **show_data_on_image** - descriptive
- **show_data_in_console** - descriptive
- **gesture_model** - chose gesture recognition model, "basic" or "sign_language"

## Access Hand Information

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

### Example

```
if controller.data['right'].gesture == 'open':
    print('Right hand is opened!')
elif controller.data['right'].gesture == 'fist':
    print('Right hand is closed!')
```

### Another Example

```
if controller.data['right'].distance < 20:
    print('Click has been detected!')
```