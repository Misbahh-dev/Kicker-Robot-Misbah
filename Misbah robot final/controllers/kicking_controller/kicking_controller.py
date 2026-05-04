# Robot Kicker - Misbah Al Rehman - 24173647

import cv2
import numpy as np
from collections import deque
from controller import Robot, Motion

# Cooldown steps between motion commands 
MOTION_COOLDOWN_STEPS = 67

# Number of frames to average ball area over to reduce sway-caused jitter
AREA_SMOOTH_WINDOW = 10

# Continuous frames without ball before treating it as truly lost
BALL_LOST_FRAMES = 8

# Normalised range that counts as "centred" in the frame
BALL_CENTRE_MIN = 0.48
BALL_CENTRE_MAX = 0.502
BALL_CENTRE_MID = (BALL_CENTRE_MIN + BALL_CENTRE_MAX) / 2.0

# Allowed drift during approach before re-aligning
APPROACH_DRIFT_TOL = 0.15

# Ball area threshold that triggers the kick
BALL_CLOSE_AREA = 1950

# Ball area at which the foot-nudge sidesteps begin
BALL_NUDGE_AREA = 1000

# Number of sidesteps in the nudge sequence(1 for left/centre ball, 2 for right ball)
NUDGE_STEPS_TOTAL = 2

# Steps between goal-post log messages 
GOAL_LOG_INTERVAL = 62

# Minimum areas to filter out noise
MIN_BALL_AREA = 400
MIN_GOAL_AREA = 800

# Color thresholds for goal, ball, grass (HSV)
GOAL_WHITE_LOWER = np.array([0,   0, 200], dtype=np.uint8)
GOAL_WHITE_UPPER = np.array([180, 40, 255], dtype=np.uint8)

GOAL_GREY_LOWER = np.array([0,   0,  70], dtype=np.uint8)
GOAL_GREY_UPPER = np.array([180, 50, 185], dtype=np.uint8)

BALL_WHITE_LOWER = np.array([0,   0, 180], dtype=np.uint8)
BALL_WHITE_UPPER = np.array([180, 50, 255], dtype=np.uint8)

BALL_BLACK_LOWER = np.array([0,   0,   0], dtype=np.uint8)
BALL_BLACK_UPPER = np.array([180, 80,  55], dtype=np.uint8)

GRASS_LOWER = np.array([35,  60,  60], dtype=np.uint8)
GRASS_UPPER = np.array([85, 255, 255], dtype=np.uint8)

#upper camera code
def detect_goal_post(image_data, width, height):
     # Convert camera image to HSV
    frame = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))
    bgr   = frame[:, :, :3].copy()
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

  # Combine white poles and grey net into goal mask
    white_mask = cv2.inRange(hsv, GOAL_WHITE_LOWER, GOAL_WHITE_UPPER)
    grey_mask  = cv2.inRange(hsv, GOAL_GREY_LOWER,  GOAL_GREY_UPPER)
    grey_mask = cv2.bitwise_and(grey_mask, cv2.bitwise_not(white_mask))

    # Dilate white broadly so adjacent grey net pixels get pulled into the mask
    k_large       = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    dilated_white = cv2.dilate(white_mask, k_large, iterations=2)

    # Clean up mask and find contours
    goal_mask = cv2.bitwise_or(white_mask, cv2.bitwise_and(dilated_white, grey_mask))

    goal_mask = cv2.morphologyEx(goal_mask, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
    goal_mask = cv2.morphologyEx(goal_mask, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    contours, _ = cv2.findContours(goal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid      = [c for c in contours if cv2.contourArea(c) >= MIN_GOAL_AREA]
    debug      = bgr.copy()
    detected   = False
    cx_norm    = 0.5
    total_area = 0.0

    if valid:
        # Combine all valid contours into one bounding box
        pts        = np.vstack(valid)
        x, y, w, h = cv2.boundingRect(pts)
        total_area  = sum(cv2.contourArea(c) for c in valid)
        cx_norm     = (x + w / 2.0) / width
        detected    = True
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug, f'GOAL cx={cx_norm:.2f}',
                    (x, max(y - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return detected, cx_norm, total_area, debug

#bottom camera code
def detect_ball(image_data, width, height):
        # Convert camera image to HSV
    frame = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))
    bgr   = frame[:, :, :3].copy()
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

     # Remove grass, then detect white and black ball panels
    not_grass  = cv2.bitwise_not(cv2.inRange(hsv, GRASS_LOWER, GRASS_UPPER))
    white_mask = cv2.bitwise_and(cv2.inRange(hsv, BALL_WHITE_LOWER, BALL_WHITE_UPPER), not_grass)
    black_mask = cv2.bitwise_and(cv2.inRange(hsv, BALL_BLACK_LOWER, BALL_BLACK_UPPER), not_grass)

     # Merge white and black regions that touchs
    k_expand  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    dilated_w = cv2.dilate(white_mask, k_expand, iterations=2)
    ball_mask = cv2.bitwise_or(white_mask, cv2.bitwise_and(dilated_w, black_mask))

     # Clean up mask and find contours
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid      = [c for c in contours if cv2.contourArea(c) >= MIN_BALL_AREA]
    debug      = bgr.copy()
    detected   = False
    cx_norm    = 0.5
    total_area = 0.0

    if valid:
        # Use the largest contour 
        best        = max(valid, key=cv2.contourArea)
        total_area  = cv2.contourArea(best)
        x, y, w, h  = cv2.boundingRect(best)
        cx_norm     = (x + w / 2.0) / width
        detected    = True
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 165, 255), 2)
        cv2.putText(debug, f'BALL cx={cx_norm:.2f} area={int(total_area)}',
                    (x, max(y - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    return detected, cx_norm, total_area, debug


class NaoGoalTracker(Robot):

    S_SEARCH_BALL   = 'SEARCH_BALL'
    S_ALIGN_BALL    = 'ALIGN_BALL'
    S_APPROACH_BALL = 'APPROACH_BALL'
    S_KICK          = 'KICK'
    DONE          = 'DONE'

    def __init__(self):
        super().__init__()
        self.timeStep = int(self.getBasicTimeStep())

        self._enable_devices()
        self._load_motions()

        self.currently_playing   = None
        self.cooldown            = 0
        self.state               = self.S_SEARCH_BALL
        self.ball_area_history   = deque(maxlen=AREA_SMOOTH_WINDOW)
        self.ball_missing_frames = 0
        self.last_ball_cx        = 0.5
        self.goal_log_countdown  = 0
        self.nudge_count         = 0
        self.nudge_done          = False

        print('Nao Kicker Robot, By Misbah Al Rehman SRN: 24173647')
        print('[NaoKickerBot] Ready — searching for ball ...')

    def _enable_devices(self):
        # Get camera devices and enable them at 4x the base time step
        self.cameraTop    = self.getDevice('CameraTop')
        self.cameraBottom = self.getDevice('CameraBottom')
        self.cameraTop.enable(4 * self.timeStep)
        self.cameraBottom.enable(4 * self.timeStep)
        self.cam_w_top  = self.cameraTop.getWidth()
        self.cam_h_top  = self.cameraTop.getHeight()
        self.cam_w_bot  = self.cameraBottom.getWidth()
        self.cam_h_bot  = self.cameraBottom.getHeight()

    def _load_motions(self):
        # Load all required motion files from the motions directory
        self.mot_forwards   = Motion('../../motions/Forwards.motion')
        self.mot_side_left  = Motion('../../motions/SideStepLeft.motion')
        self.mot_side_right = Motion('../../motions/SideStepRight.motion')
        self.mot_kick       = Motion('../../motions/shoot.motion')

    def _play_motion(self, motion, cooldown=MOTION_COOLDOWN_STEPS):
        # Stop any running motion cleanly before starting the new one
        if self.currently_playing and not self.currently_playing.isOver():
            self.currently_playing.stop()
        motion.play()
        self.currently_playing = motion
        self.cooldown = cooldown

    def _stop_motion(self):
        # Halt the current motion immediately and clear the cooldown
        if self.currently_playing and not self.currently_playing.isOver():
            self.currently_playing.stop()
        self.cooldown = 0

    def _is_busy(self):
        # True if a motion is still playing or the cooldown hasn't expired
        if self.cooldown > 0:
            return True
        if self.currently_playing and not self.currently_playing.isOver():
            return True
        return False

    def _clear_ball_history(self):
        # Wipe the smoothing buffer so stale readings don't affect next approach
        self.ball_area_history.clear()

    def _kick_immediately(self, reason='Ball at feet'):
        # Ball disappeared — assume it's in the blind spot and kick now
        print(f'[NaoKickerBot] {reason} — KICK')
        self._stop_motion()
        self._clear_ball_history()
        self.state = self.S_KICK

    # State 1: Search Ball : rotates on the spot until the ball appears in the bottom camera.
    def _state_search_ball(self, ball_detected):
        # Transition to alignment as soon as the ball appears in frame
        if ball_detected:
            print('[NaoKickerBot] Ball found — ALIGN_BALL')
            self.state = self.S_ALIGN_BALL
            return
        print('[NaoKickerBot] Waiting for ball to appear ...')

    # State 2: Align Ball : sidesteps left or right until the ball is centred in the frame.
    def _state_align_ball(self, ball_detected, ball_cx):
        # If ball is gone, assume it's at feet and kick immediately
        if not ball_detected:
            self._kick_immediately('Ball lost during align — assumed at feet')
            return

        # Move on once the ball sits inside the centre band
        if BALL_CENTRE_MIN <= ball_cx <= BALL_CENTRE_MAX:
            print(f'[NaoKickerBot] Ball centred (cx={ball_cx:.3f}) — APPROACH_BALL')
            self.nudge_count = 0
            self.nudge_done  = False
            self.state = self.S_APPROACH_BALL
            return

        if self._is_busy():
            return

        # Step in whichever direction reduces the error to the target centre
        error = ball_cx - BALL_CENTRE_MID
        if error > 0:
            print(f'[NaoKickerBot] Ball RIGHT (cx={ball_cx:.3f}) — sidestep right')
            self._play_motion(self.mot_side_right)
        else:
            print(f'[NaoKickerBot] Ball LEFT  (cx={ball_cx:.3f}) — sidestep left')
            self._play_motion(self.mot_side_left)

    # State 3: Approach Ball : walks forward toward the ball, nudging it onto the kicking foot when close.
    def _state_approach_ball(self, ball_detected, ball_cx, ball_area):
        # Ball gone during approach, assume ball in blind-spot and kick
        if not ball_detected:
            self._kick_immediately('Ball lost during approach — assumed at feet')
            return

        if ball_area >= BALL_CLOSE_AREA:
            # Re-align if close but off-centre and nudge hasn't been done yet
            if not self.nudge_done and not (BALL_CENTRE_MIN <= ball_cx <= BALL_CENTRE_MAX):
                print(f'[NaoKickerBot] Ball close but off-centre (cx={ball_cx:.3f}) — ALIGN_BALL first')
                self._stop_motion()
                self.nudge_count = 0
                self.nudge_done  = False
                self.state = self.S_ALIGN_BALL
                return
            # Ball is close enough and lined up, transition to kick
            print(f'[NaoKickerBot] Ball close (area={int(ball_area)}) — KICK')
            self._stop_motion()
            self._clear_ball_history()
            self.state = self.S_KICK
            return

        # Issue nudge sidesteps one at a time when ball reaches nudge threshold
        if not self.nudge_done and ball_area >= BALL_NUDGE_AREA:
            if self._is_busy():
                return
            if self.nudge_count < NUDGE_STEPS_TOTAL:
                self.nudge_count += 1
                print(f'[NaoKickerBot] Foot nudge {self.nudge_count}/{NUDGE_STEPS_TOTAL}: sidestep right (area={int(ball_area)})')
                self._play_motion(self.mot_side_right)
                return
            else:
                self.nudge_done = True
                print(f'[NaoKickerBot] Foot nudge complete ({NUDGE_STEPS_TOTAL} sidesteps right done)')

        # Re-align on excessive ball drift
        error = ball_cx - BALL_CENTRE_MID
        if abs(error) > APPROACH_DRIFT_TOL:
            if self.nudge_done and error < 0:
                pass
            else:
                print(f'[NaoKickerBot] Ball drifted (cx={ball_cx:.3f}) — ALIGN_BALL')
                self.nudge_count = 0
                self.nudge_done  = False
                self.state = self.S_ALIGN_BALL
                return

        if self._is_busy():
            return

        # Walk one step forward toward the ball
        print(f'[NaoKickerBot] Walking to ball (area={int(ball_area)} / {BALL_CLOSE_AREA})')
        self._play_motion(self.mot_forwards)

    # State 4: Kick : executes the shoot motion once the robot is free, then exits.
    def _state_kick(self):
        # Wait until free, fire the shoot motion, then move to DONE
        if self._is_busy():
            return
        print('[NaoKickerBot] >>> KICK! <<<')
        self._play_motion(self.mot_kick, cooldown=120)
        self.state = self.DONE

    def run(self):
        while self.step(self.timeStep) != -1:

            if self.cooldown > 0:
                self.cooldown -= 1

            # Top camera, detect goal post for logging only 
            top_raw = self.cameraTop.getImage()
            goal_detected, goal_cx, goal_area, _ = detect_goal_post(
                top_raw, self.cam_w_top, self.cam_h_top)
            if goal_detected:
                if self.goal_log_countdown <= 0:
                    print(f'[NaoKickerBot] Goal post visible: cx={goal_cx:.2f}, area={int(goal_area)}')
                    self.goal_log_countdown = GOAL_LOG_INTERVAL
                else:
                    self.goal_log_countdown -= 1
            else:
                self.goal_log_countdown = 0

            # Bottom camera, detect ball for active control
            bot_raw = self.cameraBottom.getImage()
            ball_detected, ball_cx, ball_area, _ = detect_ball(
                bot_raw, self.cam_w_bot, self.cam_h_bot)

            # Retain last confirmed position during brief ball missing
            if ball_detected:
                self.last_ball_cx = ball_cx
            ball_cx = self.last_ball_cx

            # Increment missing counter, reset on detection
            if ball_detected:
                self.ball_missing_frames = 0
            else:
                self.ball_missing_frames += 1
            ball_visible = ball_detected or (self.ball_missing_frames < BALL_LOST_FRAMES)

            # Append to smoothing buffer and compute running average
            if ball_detected:
                self.ball_area_history.append(ball_area)
            smoothed_area = (sum(self.ball_area_history) / len(self.ball_area_history)
                             if self.ball_area_history else 0.0)

            if self.state == self.S_SEARCH_BALL:
                self._state_search_ball(ball_detected)

            elif self.state == self.S_ALIGN_BALL:
                self._state_align_ball(ball_visible, ball_cx)

            elif self.state == self.S_APPROACH_BALL:
                self._state_approach_ball(ball_visible, ball_cx, smoothed_area)

            elif self.state == self.S_KICK:
                self._state_kick()

            elif self.state == self.DONE:
                # Kick animation finished, shut simulation down
                if not self._is_busy():
                    print('[NaoKickerBot] Kick complete. Shutting down.')
                    break


robot = NaoGoalTracker()
robot.run()
