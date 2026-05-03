"""
Robot Kicker – switch to ALIGN_BALL state on detection
"""
import cv2
import numpy as np
from controller import Robot, Motion

# Colour thresholds (same as commit 4)
BALL_WHITE_LOWER = np.array([0,   0, 180], dtype=np.uint8)
BALL_WHITE_UPPER = np.array([180, 50, 255], dtype=np.uint8)
BALL_BLACK_LOWER = np.array([0,   0,   0], dtype=np.uint8)
BALL_BLACK_UPPER = np.array([180, 80,  55], dtype=np.uint8)
GRASS_LOWER      = np.array([35,  60,  60], dtype=np.uint8)
GRASS_UPPER      = np.array([85, 255, 255], dtype=np.uint8)
MIN_BALL_AREA = 400

def detect_ball(image_data, width, height):
    frame = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))
    bgr   = frame[:, :, :3].copy()
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    not_grass  = cv2.bitwise_not(cv2.inRange(hsv, GRASS_LOWER, GRASS_UPPER))
    white_mask = cv2.bitwise_and(cv2.inRange(hsv, BALL_WHITE_LOWER, BALL_WHITE_UPPER), not_grass)
    black_mask = cv2.bitwise_and(cv2.inRange(hsv, BALL_BLACK_LOWER, BALL_BLACK_UPPER), not_grass)
    k_expand  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    dilated_w = cv2.dilate(white_mask, k_expand, iterations=2)
    ball_mask = cv2.bitwise_or(white_mask, cv2.bitwise_and(dilated_w, black_mask))
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) >= MIN_BALL_AREA:
            return True
    return False

class NaoGoalTracker(Robot):
    S_SEARCH_BALL   = 'SEARCH_BALL'
    S_ALIGN_BALL    = 'ALIGN_BALL'

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
        self.state = self.S_SEARCH_BALL
        self.cam_w_bot = self.cameraBottom.getWidth()
        self.cam_h_bot = self.cameraBottom.getHeight()

    def _play_motion(self, motion, cooldown=67):
        if self.currently_playing and not self.currently_playing.isOver():
            self.currently_playing.stop()
        motion.play()
        self.currently_playing = motion
        self.cooldown = cooldown

    def _is_busy(self):
        return self.cooldown > 0 or (self.currently_playing and not self.currently_playing.isOver())

    def _state_search_ball(self, ball_detected):
        if ball_detected:
            print('[NaoGoalTracker] Ball found! -> ALIGN_BALL')
            self.state = self.S_ALIGN_BALL
        else:
            print('[NaoGoalTracker] Searching ...')

    def _state_align_ball(self):
        print('[NaoGoalTracker] Aligning (placeholder)')

    def run(self):
        while self.step(self.timeStep) != -1:
            if self.cooldown > 0:
                self.cooldown -= 1

            bot_raw = self.cameraBottom.getImage()
            ball_detected = detect_ball(bot_raw, self.cam_w_bot, self.cam_h_bot)

            if self.state == self.S_SEARCH_BALL:
                self._state_search_ball(ball_detected)
            elif self.state == self.S_ALIGN_BALL:
                self._state_align_ball()

robot = NaoGoalTracker()
robot.run()
