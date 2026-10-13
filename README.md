# Picamera-Robot-Car
Raspberry Pi 4 installed, VNC IP address connected for remote control, and wired Raspberry Pi 4 to motor driver successfully.

Tested the motor control and could manually command it to move accordingly, successfully recorded the time to turn the car at exact angles in both directions, and measured the distance for varying PWM values and calculated the speed of the robot car using 80% PWM.   

Successfully installed the picamera2, used Template Matching and Teachable to detect shapes, symbols and arrows accurately, and was able to follow the centre black line closely throughout the whole track with PID control.   

Combined the detection system with the line-following algorithm using multithreading to ensure smooth autonomous navigation, developed and calibrated custom movements for all required symbols and arrows, including a timed 360-degree rotation, implemented a robust color-masking priority system to navigate crossroads accurately

Demo video of the multicoloured line following with shape and symbol detection:
https://github.com/user-attachments/assets/96dce159-bce9-4151-9e3d-0fef62bf848d

