#!/usr/bin/env python3
from ev3dev2.motor import *
from ev3dev2.sensor.lego import *
from ev3dev2.sound import Sound
from threading import Thread 
from random import randint
from time import sleep

steer = MoveSteering(OUTPUT_B, OUTPUT_C)
mmotor = MediumMotor()
ts = TouchSensor()
cs = ColorSensor()
ir = InfraredSensor()
sound = Sound()

def touch():
	while True:
		ts.wait_for_pressed()
		steer.off()
		mmotor.on_for_rotations(75,5)

def light():
        while True:
               sound.speak(str(cs.ambient_light_intensity))
               print(str(cs.ambient_light_intensity), file=light.txt)
               sleep(10)

t1 = Thread(target=touch)
t2 = Thread(target=light)

t1.start()
t2.start()

while True:
	if ir.proximity < 40:
		steer.off()
		steer.on_for_rotations(0,-50,3)
		steer.on_for_rotations([-100, 100][randint(0,1)],50,1)
	else:
		steer.on(0,50)
	











