#!/usr/bin/env python3
from ev3dev2.led import Leds
from ev3dev2.sensor.lego import TouchSensor

leds = Leds()
ts = TouchSensor()

leds.all_off()
while True:
	if ts.is_pressed:
		leds.set_color('LEFT', 'GREEN')
		leds.set_color('RIGHT', 'GREEN')
	else:
		leds.set_color('LEFT', 'RED')
		leds.set_color('RIGHT', 'RED')