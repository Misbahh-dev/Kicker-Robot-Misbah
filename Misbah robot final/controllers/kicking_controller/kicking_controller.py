
from controller import Robot, Motion

class NaoGoalTracker(Robot):
    def __init__(self):
        super().__init__()
        self.timeStep = int(self.getBasicTimeStep())

        self.cameraTop = self.getDevice('CameraTop')
        self.cameraBottom = self.getDevice('CameraBottom')
        self.cameraTop.enable(4 * self.timeStep)
        self.cameraBottom.enable(4 * self.timeStep)

        self.mot_forwards   = Motion('../../motions/Forwards.motion')
        self.mot_side_left  = Motion('../../motions/SideStepLeft.motion')
        self.mot_side_right = Motion('../../motions/SideStepRight.motion')
        self.mot_kick       = Motion('../../motions/shoot.motion')

        self.currently_playing = None
        self.cooldown = 0

        print('[NaoGoalTracker] Motions loaded')

    def _play_motion(self, motion, cooldown=67):
        if self.currently_playing and not self.currently_playing.isOver():
            self.currently_playing.stop()
        motion.play()
        self.currently_playing = motion
        self.cooldown = cooldown

    def _is_busy(self):
        return self.cooldown > 0 or (self.currently_playing and not self.currently_playing.isOver())

    def run(self):
        while self.step(self.timeStep) != -1:
            if self.cooldown > 0:
                self.cooldown -= 1
            

robot = NaoGoalTracker()
robot.run()
