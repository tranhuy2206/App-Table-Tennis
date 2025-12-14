import PoseModule as pm
from processor_base import ProcessorBase

class PoseProcessor(ProcessorBase):
    def __init__(self):
        self.detector = pm.poseDetector()

    def process(self, frame_bgr):
        frame_out = self.detector.findPose(frame_bgr)                   # vẽ skeleton nếu draw=True mặc định
        lm_list   = self.detector.findPosition(frame_out, draw=False)   # nếu cần dùng landmarks
        side = {"lmList": lm_list}                                      # dữ liệu phụ (tuỳ dùng/không)
        return frame_out, side