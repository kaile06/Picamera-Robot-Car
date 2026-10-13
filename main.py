# -- coding: utf-8 --

import sys
import importlib.util

# --- PYTHON 3.13 FIX ---
class FakeImp:
    def find_module(self, name):
        if importlib.util.find_spec(name) is None:
            raise ImportError(f"No module named {name}")
        return (None, None, None) 
sys.modules['imp'] = FakeImp()

from picamera2 import Picamera2
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
from tensorflow.lite.python import interpreter as tflite 

# ==========================================
# 1. HARDWARE & PID SETUP
# ==========================================
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# Pins
IN1_RIGHT, IN2_RIGHT, ENA_RIGHT = 15, 16, 18
IN3_LEFT, IN4_LEFT, ENB_LEFT = 11, 12, 13

GPIO.setup([IN1_RIGHT, IN2_RIGHT, ENA_RIGHT, IN3_LEFT, IN4_LEFT, ENB_LEFT], GPIO.OUT)

pwm_right = GPIO.PWM(ENA_RIGHT, 1000)
pwm_left = GPIO.PWM(ENB_LEFT, 1000)
pwm_right.start(0); pwm_left.start(0)

# Speed Variables
SPEED_FORWARD = 36 
SPEED_TURN = 65
SPEED_SPIN = 100 

# PID Variables
Kp = 0.45  
Ki = 0.00
Kd = 0.15
last_error = 0
integral = 0
last_emergency_time = 0 

# Motor Functions
def stop_car():
    GPIO.output([IN1_RIGHT, IN2_RIGHT, IN3_LEFT, IN4_LEFT], False)
    pwm_right.ChangeDutyCycle(0); pwm_left.ChangeDutyCycle(0)

def move_forward():
    GPIO.output(IN1_RIGHT, True); GPIO.output(IN2_RIGHT, False)
    GPIO.output(IN3_LEFT, True); GPIO.output(IN4_LEFT, False)
    pwm_right.ChangeDutyCycle(SPEED_FORWARD); pwm_left.ChangeDutyCycle(SPEED_FORWARD)

def move_backward():
    GPIO.output(IN1_RIGHT, False); GPIO.output(IN2_RIGHT, True)
    GPIO.output(IN3_LEFT, False); GPIO.output(IN4_LEFT, True)
    pwm_right.ChangeDutyCycle(SPEED_FORWARD); pwm_left.ChangeDutyCycle(SPEED_FORWARD)

def turn_left():
    GPIO.output(IN1_RIGHT, True); GPIO.output(IN2_RIGHT, False)
    GPIO.output(IN3_LEFT, False); GPIO.output(IN4_LEFT, True)
    pwm_right.ChangeDutyCycle(SPEED_SPIN); pwm_left.ChangeDutyCycle(SPEED_SPIN)

def turn_right():
    GPIO.output(IN1_RIGHT, False); GPIO.output(IN2_RIGHT, True)
    GPIO.output(IN3_LEFT, True); GPIO.output(IN4_LEFT, False)
    pwm_right.ChangeDutyCycle(SPEED_SPIN); pwm_left.ChangeDutyCycle(SPEED_SPIN)

# ==========================================
# 2. VISION INIT
# ==========================================
print("Booting Master Vision System...")

try:
    interpreter = tflite.Interpreter(model_path='model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    with open('labels.txt', 'r') as f: labels = [line.strip() for line in f.readlines()]
except Exception as e: print(f"AI Model Error: {e}")

def load_template(path):
    img = cv2.imread(path, 0)
    if img is None: return None
    _, th = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
    x, y, w, h = cv2.boundingRect(th)
    return cv2.resize(img[y:y+h, x:x+w] if w > 0 else img, (40, 40))

templates = {
    "Up": load_template("up_arrow.png"), 
    "Down": load_template("down_arrow.png"),
    "Left": load_template("left_arrow.png"), 
    "Right": load_template("right_arrow.png"),
    "PressButton": load_template("pressbutton.png") 
}

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(config); picam2.start()

cooldown_until = 0
frame_counter = 0

# ==========================================
# 3. MAIN INTEGRATED LOOP
# ==========================================
try:
    while True:
        frame_high = picam2.capture_array()
        if frame_high is None: continue
        
        frame_bgr = cv2.cvtColor(frame_high, cv2.COLOR_RGB2BGR)
        frame_low = cv2.resize(frame_bgr, (160, 120))
        gray_low = cv2.cvtColor(frame_low, cv2.COLOR_BGR2GRAY)
        
        pause_this_frame = False
        current_time = time.time()
        frame_counter += 1

        # --- BRAIN 1: SYMBOLS & TEMPLATES ---
        if current_time > cooldown_until:
            matched_template = False
            
            for name, temp in templates.items():
                if temp is None: continue
                res = cv2.matchTemplate(gray_low, temp, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                
                thresh = 0.6 if name == "PressButton" else 0.7
                
                if max_val >= thresh:
                    print(f"\n** MATCH: {name} ({max_val:.2f}) **")
                    stop_car()
                    
                    if name in ["Left", "Right"]:
                        move_backward(); time.sleep(0.3); stop_car()
                        if name == "Left": turn_right() 
                        else: turn_left()
                        time.sleep(1.5) 
                    elif name == "PressButton":
                        cv2.waitKey(2000)
                        SPEED_FORWARD = 70 
                        print("SPRINT ACTIVATED")
                    else: 
                        cv2.waitKey(1000)
                    
                    stop_car()
                    cooldown_until = time.time() + 2.0 
                    pause_this_frame = True
                    matched_template = True; integral = 0
                    break

            if not matched_template and not pause_this_frame and (frame_counter % 3 == 0):
                try: # <--- THIS IS THE TRY BLOCK THAT WAS MISSING ITS EXCEPT
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    resized_ai = cv2.resize(frame_rgb, (224, 224))
                    input_data = np.expand_dims(resized_ai, axis=0)

                    if input_details[0]['dtype'] == np.float32:
                        input_data = (np.float32(input_data) / 127.5) - 1.0
                    else:
                        input_data = np.uint8(input_data)

                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    out = interpreter.get_tensor(output_details[0]['index'])[0]
                    
                    if output_details[0]['dtype'] == np.uint8: out = out / 255.0  
                    
                    best_symbol = None
                    best_conf = 0.0
                    
                    for i, conf in enumerate(out):
                        label = labels[i].lower()
                        if any(x in label for x in ["background", "none", "floor"]): continue
                            
                        if conf > 0.50 and conf > best_conf:
                            best_conf = conf
                            best_symbol = label
                    
                    if best_symbol is not None:
                        if any(x in best_symbol for x in ["danger", "stop"]):
                            print(f"\n** AI PRIORITY TRIGGER: {best_symbol} ({best_conf*100:.1f}%) **")
                            stop_car(); cv2.waitKey(2000)
                            cooldown_until = time.time() + 2.5
                            pause_this_frame = True; integral = 0
                            
                        elif "recycle" in best_symbol:
                            print(f"\n** AI PRIORITY TRIGGER: {best_symbol} ({best_conf*100:.1f}%) **")
                            GPIO.output(IN3_LEFT, True); GPIO.output(IN4_LEFT, False)
                            GPIO.output(IN1_RIGHT, False); GPIO.output(IN2_RIGHT, True)
                            pwm_left.ChangeDutyCycle(100); pwm_right.ChangeDutyCycle(100)
                            time.sleep(1.8); stop_car()
                            cooldown_until = time.time() + 2.5
                            pause_this_frame = True; integral = 0
                            
                        # --- FINGERPRINT & QR CODE TRIGGER ---
                        elif any(x in best_symbol for x in ["fingerprint", "qr"]):
                            print(f"\n** AI PRIORITY TRIGGER: {best_symbol} ({best_conf*100:.1f}%) **")
                            stop_car()
                            cv2.waitKey(2000) # Stops for 2 seconds
                            cooldown_until = time.time() + 2.5
                            pause_this_frame = True; integral = 0
                            
                        else:
                            print(f"-> AI SAW: {best_symbol} ({best_conf*100:.1f}%) [NO ACTION PROGRAMMED]")

                # ---> THIS IS THE EXCEPT BLOCK THAT ACCIDENTALLY GOT DELETED! <---
                except Exception as e: 
                    print(f"\n[!] AI MODEL CRASHED: {e}")

        # --- BRAIN 2: PID LINE FOLLOWING & FAILSAFES ---
        if not pause_this_frame:
            crop = frame_bgr[160:480, 0:640]
            
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mask_yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
            mask_red = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])) | cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
            mask_color = mask_yellow | mask_red 
            
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            _, mask_black = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours_c, _ = cv2.findContours(mask_color, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # SMART PRIORITY HANDOFF
            if len(contours_c) > 0 and cv2.contourArea(max(contours_c, key=cv2.contourArea)) > 2000:
                mask_view = mask_color
            else:
                mask_view = mask_black

            line_found = False
            contours, _ = cv2.findContours(mask_view, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 500:
                    line_found = True
                    
                    x, y, w, h = cv2.boundingRect(c)
                    
                    # --- DYNAMIC JUNCTION DETECTOR ---
                    if w > 350 and (current_time - last_emergency_time > 1.5):
                        M = cv2.moments(c)
                        cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 320
                        junction_error = 320 - cx 
                        
                        direction = "RIGHT" if junction_error > 0 else "LEFT"
                        print(f"* JUNCTION DETECTED (Width: {w}): Deciding to turn {direction} (Error: {junction_error}) *")
                        
                        stop_car(); move_backward(); time.sleep(0.25); stop_car()
                        
                        if junction_error > 0: turn_right()
                        else: turn_left()
                        
                        time.sleep(0.5) 
                        stop_car()
                        
                        last_emergency_time = time.time()
                        last_error = 0; integral = 0
                        
                    # --- PURE PID STEERING ---
                    else:
                        M = cv2.moments(c)
                        if M['m00'] != 0:
                            topmost = tuple(c[c[:, :, 1].argmin()][0])
                            center_x = int(M['m10'] / M['m00'])
                            
                            cx = int((topmost[0] * 0.6) + (center_x * 0.4))
                            error = 320 - cx 
                            
                            speed_penalty = abs(error) * 0.2
                            dynamic_speed = max(25, SPEED_FORWARD - speed_penalty)

                            integral += error
                            derivative = error - last_error
                            corr = (Kp * error) + (Ki * integral) + (Kd * derivative)
                            last_error = error
                            
                            l_spd = dynamic_speed + corr
                            r_spd = dynamic_speed - corr
                            
                            if l_spd < 0:
                                GPIO.output(IN3_LEFT, False); GPIO.output(IN4_LEFT, True) 
                            else:
                                GPIO.output(IN3_LEFT, True); GPIO.output(IN4_LEFT, False) 
                                
                            if r_spd < 0:
                                GPIO.output(IN1_RIGHT, False); GPIO.output(IN2_RIGHT, True) 
                            else:
                                GPIO.output(IN1_RIGHT, True); GPIO.output(IN2_RIGHT, False) 
                            
                            pwm_left.ChangeDutyCycle(min(100, abs(l_spd)))
                            pwm_right.ChangeDutyCycle(min(100, abs(r_spd)))
            
            # --- FAILSAFES ---
            if not line_found:
                if last_error != 0:
                    if last_error > 0: turn_right()
                    else: turn_left()
                else:
                    print("* BLIND SEARCH: Spinning Left *")
                    turn_left()
                
                integral = 0 

        cv2.imshow("Main Brain", frame_low)
        cv2.imshow("Line Mask", cv2.resize(mask_view, (320, 160)))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    stop_car(); picam2.stop(); cv2.destroyAllWindows(); GPIO.cleanup()