# ---- Minimal player chỉ cho tab Compare (không đụng tab 1/2) ----
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
import cv2

class AutoplayPreview:
    def __init__(self, mw, video_label, processor, interval_ms=33, loop=True):
        self.mw = mw
        self.label = video_label
        self.processor = processor  # ví dụ PoseProcessor()
        self.loop = loop
        self.cap = None
        self.timer = QTimer(self.mw)
        self.timer.setInterval(interval_ms)
        self.timer.timeout.connect(self._tick)

    def set_source(self, path: str):
        # mở là chạy, không cần nút
        self.stop()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.cap = None
            return
        # hiển thị khung đầu (nếu muốn)
        ok, f = self.cap.read()
        if ok and f is not None:
            f = self._process(f)
            self._display(f)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.timer.start()

    def stop(self):
        self.timer.stop()
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _tick(self):
        if not (self.cap and self.cap.isOpened()):
            self.timer.stop(); return
        ok, f = self.cap.read()
        if not ok or f is None:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                self.stop()
            return
        f = self._process(f)
        self._display(f)

    def _process(self, bgr):
        # vẽ khung xương bằng processor của bạn (an toàn theo frame)
        try:
            out, _ = self.processor.process(bgr) if self.processor else (bgr, None)
            return out if out is not None else bgr
        except Exception:
            return bgr

    def _display(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(pix)
