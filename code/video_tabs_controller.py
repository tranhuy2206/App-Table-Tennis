import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFileDialog, QApplication

class VideoTabController:
    def __init__(self, parent_window, *, video_label, btn_open, btn_start, btn_pause, processor):
        self.mw = parent_window
        self.video_label = video_label
        self.processor = processor

        self.cap = None
        self.timer = QTimer(self.mw)
        self.timer.timeout.connect(self._update_frame)

        self.is_paused = False
        self.current_source = None
        self.interval_ms = 33
        self.last_qimg = None

        if btn_open is not None:
            btn_open.clicked.connect(self._open_video)
        if btn_start is not None:
            btn_start.clicked.connect(self.start_or_resume)
        if btn_pause is not None:
            btn_pause.clicked.connect(self.pause)


        QApplication.instance().aboutToQuit.connect(self.stop)

    # --- Buttons ---
    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self.mw, "Chọn video", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
        )
        if path:
            self.start(path)

    def start_or_resume(self):
        if self.cap and self.is_paused:
            self.is_paused = False
            self.timer.start(self.interval_ms)
            return
        if self.current_source is None:
            self._open_video()
        else:
            self.start(self.current_source)

    def pause(self):
        if self.cap and self.timer.isActive():
            self.timer.stop()
            self.is_paused = True

    # --- Core ---
    def start(self, source):
        if self.cap and self.is_paused and source == self.current_source:
            self.is_paused = False
            self.timer.start(self.interval_ms)
            return

        self.stop()  # release nguồn cũ
        self.cap = cv2.VideoCapture(source)
        self.current_source = source

        if not self.cap.isOpened():
            self.video_label.setText("Không mở được video/camera")
            self.cap = None
            self.current_source = None
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 1:
            fps = 30
        self.interval_ms = int(1000 / fps)
        self.is_paused = False
        self.timer.start(self.interval_ms)

    def stop(self):
        self.timer.stop()
        if self.cap:
            try: self.cap.release()
            except: pass
        self.cap = None
        self.is_paused = False
        # để nguyên current_source: sau này Start có thể phát lại cùng nguồn

    def _update_frame(self):
        if not (self.cap and self.cap.isOpened()):
            self.timer.stop(); return

        ok, frame_bgr = self.cap.read()
        if not ok:
            self.timer.stop(); return

        frame_out, _ = self.processor.process(frame_bgr)  # thuật toán tuỳ tab
        frame_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.last_qimg = qimg.copy()
        self._show(qimg)

    # --- Helpers ---
    def _show(self, qimg):
        scaled = qimg.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(QPixmap.fromImage(scaled))

    def on_resize(self):
        if self.last_qimg is not None:
            self._show(self.last_qimg)
