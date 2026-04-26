

from controller import Robot
import time

# Create the Robot instance
robot = Robot()

# motor devices
left_motor = robot.getDevice('left_motor')
right_motor = robot.getDevice('right_motor')


left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

MAX_VELOCITY = 3

left_motor.setVelocity(MAX_VELOCITY)
right_motor.setVelocity(MAX_VELOCITY)

#  timestep
timestep = int(robot.getBasicTimeStep())

# Run for 3 seconds
start_time = robot.getTime()
while robot.step(timestep) != -1:
    if robot.getTime() - start_time >= 3.0:
        # Stop motors after 3 seconds
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        break

print("Movement complete!")