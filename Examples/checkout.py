from openleap import OpenLeap
import pygame
from pygame.locals import *
import os


def sort_dict(dictionary):
    ret_dict = {}
    for key, element in dictionary.items():
        if element > 0:
            ret_dict[key] = element

    return ret_dict


def draw_cursor():
    global cursor

    x = (controller.data['right'].x-0.25)/0.75
    if x>1: x=1
    if x<0: x=0
    x *= WIDTH

    y = (controller.data['right'].y-0.4)/0.6
    if y>1: y=1
    if y<0: y=0
    y *= HEIGHT

    cursor = pygame.Rect(x, y, 10, 10)

    if controller.data['right'].gesture == 'open':
        pygame.draw.rect(screen, pygame.Color(0,0,0), cursor)
    elif controller.data['right'].gesture == 'fist':
        pygame.draw.rect(screen, pygame.Color(255,0,0), cursor)
    elif controller.data['right'].distance < 25:
        pygame.draw.rect(screen, pygame.Color(255,0,0), cursor)
    else:
        pygame.draw.rect(screen, pygame.Color(0,0,0), cursor)
    pygame.display.update(cursor)



# controller = OpenLeap(show_data_in_console==False, screen_show==True, screen_type=='BLACK', show_data_on_image==True)
controller = OpenLeap(screen_show=True, show_data_on_image=True, screen_type='BLACK', show_data_in_console=True)

HEIGHT, WIDTH = 900, 880
pygame.init()

# # flags = FULLSCREEN | DOUBLEBUF

flags = DOUBLEBUF

screen = pygame.display.set_mode((HEIGHT, WIDTH), flags, 16)
screen.fill(pygame.Color(0, 150, 150))

'''
CURSOR
'''
cursor = pygame.Rect(0, 0, 10, 10)

'''
BUTTONS
'''
LIGHT_BUTTON_COLOR=(207,200,68)
DARK_BUTTON_COLOR=(157,150,18)

buttons = {}
buttons_x=50

buttons['burger'] = pygame.Rect(buttons_x, 10, 250, 200)
buttons['pizza'] = pygame.Rect(buttons_x, 230, 250, 200)
buttons['taco'] = pygame.Rect(buttons_x, 450, 250, 200)
buttons['fries'] = pygame.Rect(buttons_x, 670, 250, 200)

images = {}

this_dir, this_filename = os.path.split(__file__)  # Get path of data.pkl
# images['burger'] = pygame.image.load('graphics/burger.png')
for key, element in buttons.items():
    data_path = os.path.join(this_dir, 'graphics/'+key+'.png')
    images[key] = pygame.image.load(data_path)

'''
ORDER LIST
'''
pygame.font.init()
order_list_font_big = pygame.font.SysFont('Times New Roman', 72)
order_list_font_small = pygame.font.SysFont('Times New Roman', 30)

order_list={'burger':0, 'pizza':0, 'taco':0, 'fries':0}
order_list_to_show = {}

gesture_flag=True
last_gesture=None

while True:
    controller.main()

    #Background
    screen.fill(pygame.Color(0, 150, 150))

    #Buttons
    for key, element in buttons.items():

        if element.colliderect(cursor):
            pygame.draw.rect(screen, LIGHT_BUTTON_COLOR, element)

            if controller.data['right'].distance < 25 and gesture_flag==True:
                gesture_flag=False
                order_list[key] += 1
                order_list_to_show = sort_dict(order_list)

                print(order_list)
        else:
            pygame.draw.rect(screen, DARK_BUTTON_COLOR, element)

        screen.blit(images[key], (element.x+20,element.y))

    #Shopping list
    screen.blit(order_list_font_big.render('Order list', False, (0,0,0)), (450, 50))

    i=0
    for key, element in order_list_to_show.items():
        if element > 0:
            screen.blit(order_list_font_small.render(f'{i+1}. {key}, x {element}', False, (0,0,0)), (480, 150+50*i))
        i += 1


    #Cursor
    draw_cursor()

    if controller.detect_key('q'):
        controller.close_window()
        break

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run=False

    if controller.data['right'].distance > 25:
        gesture_flag=True

    pygame.display.update()