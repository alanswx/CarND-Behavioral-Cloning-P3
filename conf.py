'''
steering our bot takes a -1, to 1 range
but our NN likes larger numbers to train against
so scale our steering output by this constant
'''
#STEERING_NN_SCALE = 30.0
STEERING_NN_SCALE = 1

'''
When mounting the camera upside down, it's helpful to flip both of these.
'''
CAMERA_FLIP_VERT = False
CAMERA_FLIP_HORZ = False

'''
Should images be left RGB or should we try just one channel?
'''
GREY_SCALE = False

'''
image dimensions
'''
row, col, ch = 160, 120, 3
#row, col, ch = 256, 256, 3

'''
These are the 0-15 interger plug in that we use for the Adafruit Servo Hat
'''
ada_steering_servo_channel = 1
ada_esc_motor_channel = 4

'''
These are the integer ids of the axis on the joystick for input
'''
steering_js_axis_id = 0
throttle_js_axis_id = 3

'''
we ge inputput from +- js_axis_scale and then normalize them for input to robot
'''
js_axis_scale = 32767.0


'''
These low, hi, med values were observed by using the 
arrows.py script to see what worked. User can adjust to suit
their bot.
'''
ada_steering_low = 310
ada_steering_hi = 530
ada_steering_mid = 400


'''
These low, hi, med values were observed by using the 
arrows.py script to see what worked. User can adjust to suit
their bot.
'''
ada_esc_init_lo = 200 #reverse
ada_esc_init_hi = 600 #full forward
ada_esc_init_mid = 400 #idle

ada_esc_throttle_lo = 350 #reverse
ada_esc_throttle_hi = 450 #full forward
ada_esc_throttle_mid = 400 #idle

'''
gpio pin we use for status led 
'''
status_pin = 23

