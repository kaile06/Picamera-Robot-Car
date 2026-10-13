# Picamera-Robot-Car
This project is an autonomous robot car powered by a Raspberry Pi 4. Programmed in Python to navigate complex physical tracks. The core movement of the robot car relies on an L298N motor driver that is calibrated for precise rotational angles and PWM speed control.

Tested the motor control and could manually command it to move accordingly, successfully recorded the time to turn the car at exact angles in both directions, and measured the distance for varying PWM values and calculated the speed of the robot car using 80% PWM. 

Installed a picamera2, used Template Matching and Teachable to detect shapes, symbols and arrows accurately, and was able to follow the centre of black line closely throughout the whole track with PID control.Combined the detection system with the line-following algorithm using multithreading to ensure smooth autonomous navigation. Developed and calibrated custom movements for all required symbols and arrows, including a timed 360-degree rotation. Implemented a robust color-masking priority system to navigate coloured line shortcuts accurately

Demo video of the multicoloured line following with shape and symbol detection:
https://github.com/user-attachments/assets/96dce159-bce9-4151-9e3d-0fef62bf848d

