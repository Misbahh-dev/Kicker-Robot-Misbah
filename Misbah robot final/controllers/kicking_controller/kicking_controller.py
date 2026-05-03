"""
Robot Kicker – APPROACH_BALL walks toward ball using area threshold
"""
import cv2
import numpy as np
from collections import deque
from controller import Robot, Motion

# Colour thresholds (unchanged)
BALL_WHITE_LOWER = np.array([0,   0, 180], dtype=np.uint8)
BALL_WHITE_UPPER = np.array([180, 50, 255], dtype=np.uint8)
BALL_BLACK_LOWER = np.array([0,   0,   0], dtype=np.uint8)
BALL_BLACK_UPPER = np.array([180, 80,  55], dtype=np.uint8)
GRASS_LOWER      = np.array([35,  60,  60], dtype=np.uint8)
GRASS_UPPER      = np.array([85, 255, 255], dtype=np.uint8)
MIN_BALL_AREA = 400
BALL_CENTRE_MIN = 0.48
BALL_CENTRE_MAX = 0.502
BALL_CENTRE_MID = (BALL_CENTRE_MIN + BALL_CENTRE_MAX) / 2.0
BALL_CLOSE_AREA = 1950
AREA_SMOOTH_WINDOW = 10

def detect_ball(image_data, width, height):
    frame = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))
    bgr   = frame[:, :, :3].copy()
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    not_grass = cv2.bitwise_not(cv2.inRange(hsv, GRASS_LOWER, GRASS_UPPER))
    white_mask = cv2.bitwise_and(cv2.inRange(hsv, BALL_WHITE_LOWER, BALL_WHITE_UPPER), not_grass)
    black_mask = cv2.bitwise_and(cv2.inRange(hsv, BALL_BLACK_LOWER, BALL_BLACK_UPPER), not_grass)
    k_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    dilated_w = cv2.dilate(white_mask, k_expand, iterations=2)
    ball_mask = cv2.bitwise_or(white_mask, cv2.bitwise_and(dilated_w, black_mask))
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_area = 0
    best_cx = 0.5
    for c in contours:
        area = cv2.contourArea(c)
        if area >= MIN_BALL_AREA and area > best_area:
            best_area = area
            x, y, w, h = cv2.boundingRect(c)
            best_cx = (x + w/2.0) / width
    return best_area > 0, best_cx, best_area

class NaoGoalTracker(Robot):
    S_SEARCH_BALL   = 'SEARCH_BALL'
    S_ALIGN_BALL    = 'ALIGN_BALL'
    S_APPROACH_BALL = 'APPROACH_BALL'

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
        self.last_ball_cx = 0.5
        self.ball_area_history = deque(maxlen=AREA_SMOOTH_WINDOW)

    def _play_motion(self, motion, cooldown=67):
        if self.currently_playing and not self.currently_playing.isOver():
            self.currently_playing.stop()
        motion.play()
        self.currently_playing = motion
        self.cooldown = cooldown

    def _is_busy(self):
        return self.cooldown > 0 or (self.currently_playing and not self.currently_playing.isOver())

    def _clear_ball_history(self):
        self.ball_area_history.clear()

    # SEARCH_BALL same as before
    def _state_search_ball(self, ball_detected):
        if ball_detected:
            print('[NaoGoalTracker] Ball found -> ALIGN_BALL')
            self.state = self.S_ALIGN_BALL
        else:
            print('[NaoGoalTracker] Searching ...')

    def _state_align_ball(self, ball_detected, ball_cx):
        if not ball_detected:
            print('[NaoGoalTracker] Ball lost -> go back to SEARCH')
            self.state = self.S_SEARCH_BALL
            return
        if BALL_CENTRE_MIN <= ball_cx <= BALL_CENTRE_MAX:
            print(f'[NaoGoalTracker] Centred -> APPROACH_BALL')
            self.state = self.S_APPROACH_BALL
            return
        if self._is_busy():
            return
        error = ball_cx - BALL_CENTRE_MID
        if error > 0:
            self._play_motion(self.mot_side_right)
        else:
            self._play_motion(self.mot_side_left)

    def _state_approach_ball(self, ball_detected, ball_area):
        if not ball_detected:
            print('[NaoGoalTracker] Ball lost during approach -> SEARCH')
            self.state = self.S_SEARCH_BALL
            return
        if ball_area >= BALL_CLOSE_AREA:
            print(f'[NaoGoalTracker] Ball close (area={int(ball_area)}) -> KICK')
            self._stop_motion()
            self._clear_ball_history()
            self.state = self.S_KICK  # will be defined later
            return
        if self._is_busy():
            return
        print(f'[NaoGoalTracker] Walking to ball (area={int(ball_area)})')
        self._play_motion(self.mot_forwards)

    # Placeholder kick state (no actual kick yet)
    def _state_kick(self):
        print('[NaoGoalTracker] KICK! (placeholder)')
        # In next commit we will add real kick

    def run(self):
        while self.step(self.timeStep) != -1:
            if self.cooldown > 0:
                self.cooldown -= 1

            bot_raw = self.cameraBottom.getImage()
            ball_detected, ball_cx, ball_area = detect_ball(bot_raw, self.cam_w_bot, self.cam_h_bot)
            if ball_detected:
                self.last_ball_cx = ball_cx
                self.ball_area_history.append(ball_area)
            ball_cx = self.last_ball_cx
            smoothed_area = (sum(self.ball_area_history) / len(self.ball_area_history)
                             if self.ball_area_history else 0.0)

            if self.state == self.S_SEARCH_BALL:
                self._state_search_ball(ball_detected)
            elif self.state == self.S_ALIGN_BALL:
                self._state_align_ball(ball_detected, ball_cx)
            elif self.state == self.S_APPROACH_BALL:
                self._state_approach_ball(ball_detected, smoothed_area)
            elif self.state == self.S_KICK:
                self._state_kick()

robot = NaoGoalTracker()
robot.run()
