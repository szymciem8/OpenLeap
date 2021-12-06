from openleap import OpenLeap
import pygame
from pygame.locals import *
import colorsys


controller = OpenLeap(screen_show=True, show_data_on_image=True, screen_type='BLACK', show_data_in_console=True, gesture_model='basic')

HEIGHT, WIDTH = 900, 900
pygame.init()

# # flags = FULLSCREEN | DOUBLEBUF

flags = DOUBLEBUF

screen = pygame.display.set_mode((HEIGHT, WIDTH), flags, 16)
screen.fill(pygame.Color(0, 150, 150))
r, g, b = 0, 0, 0
color, saturation = 0, 0

while True:
    controller.main()

    if controller.data['right'].gesture == 'open':
        color = controller.data['right'].angle/180

    if controller.data['left'].gesture == 'open' and controller.data['right'].gesture == 'open':
        saturation = controller.data['left'].distance/180
        if saturation<0: saturation=0
        if saturation>1: saturation=1

    r, g, b = colorsys.hsv_to_rgb(color, saturation, 1)

    r = int(255*r)
    g = int(255*g)
    b = int(255*b)

    screen.fill(pygame.Color(r, g, b))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run=False

    if controller.detect_key('q'):
        controller.close_window()
        break

    pygame.display.update()