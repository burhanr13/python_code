#!/usr/bin/env python3
from ev3dev2.motor import MoveSteering, OUTPUT_B, OUTPUT_C

steer = MoveSteering(OUTPUT_B, OUTPUT_C)

steer.on_for_rotations(0, 75, 5)
steer.on_for_rotations(50, 75, 5)
steer.on_for_rotations(0, 75, 5)
steer.on_for_rotations(-50, 75, 5)
