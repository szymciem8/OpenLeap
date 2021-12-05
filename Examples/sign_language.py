from openleap import *

controller = OpenLeap(screen_show=True, screen_type='BLACK', show_data_on_image=True)

# controller = OpenLeap(show_data_in_console=True, screen_show=True, screen_type='BLACK', show_data_on_image=True)

while True:
    controller.main()
    # print(controller.relative_position['right'])
    if controller.detect_key('q'):
        controller.close_window()
        break