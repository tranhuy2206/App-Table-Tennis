import cv2
import numpy as np
from processor_base import ProcessorBase

class BallProcessor(ProcessorBase):
    def __init__(self, frame_w=640, frame_h=480, net_buffer=3):
        self.FRAME_WIDTH  = frame_w
        self.FRAME_HEIGHT = frame_h
        self.NET_BUFFER   = net_buffer
        self.size = (self.FRAME_WIDTH, self.FRAME_HEIGHT)

        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

        self.ball_count = 0
        self.prev_ball_pos = None
        self.ball_direction = 0
        self.net_x_position = self.FRAME_WIDTH // 2

        self.kernel = np.ones((5,5), np.uint8)

    def process(self, frame_bgr):

        frame = cv2.resize(frame_bgr, self.size)

        fgmask = self.fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 100:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
                    cv2.circle(frame, center, 8, (0, 255, 0), -1)

        if center is not None:
            if self.prev_ball_pos is not None:
                # trái -> phải
                if (self.prev_ball_pos[0] < (self.net_x_position - self.NET_BUFFER)
                    and center[0] > (self.net_x_position + self.NET_BUFFER)):
                    if self.ball_direction != 1:
                        self.ball_count += 1
                    self.ball_direction = 1
                # phải -> trái
                elif (self.prev_ball_pos[0] > (self.net_x_position + self.NET_BUFFER)
                      and center[0] < (self.net_x_position - self.NET_BUFFER)):
                    if self.ball_direction != -1:
                        self.ball_count += 1
                    self.ball_direction = -1
            self.prev_ball_pos = center

        # vẽ lưới & đếm
        x = self.net_x_position
        cv2.line(frame, (x, 0), (x, self.FRAME_HEIGHT), (255, 0, 0), 2)
        cv2.line(frame, (x - self.NET_BUFFER, 0), (x - self.NET_BUFFER, self.FRAME_HEIGHT), (255, 255, 0), 1)
        cv2.line(frame, (x + self.NET_BUFFER, 0), (x + self.NET_BUFFER, self.FRAME_HEIGHT), (255, 255, 0), 1)
        cv2.putText(frame, f"Count: {self.ball_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        side = {
            "mask": fgmask,
            "count": self.ball_count
        }

        return frame, side