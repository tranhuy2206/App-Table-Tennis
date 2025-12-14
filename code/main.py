import sys, os, warnings   
from PySide6 import QtUiTools
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QTabWidget
from video_tabs_controller import VideoTabController
from processor_ball import BallProcessor
from processor_pose import PoseProcessor
from compare_tab_controller import CompareTabController


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mw = QtUiTools.QUiLoader().load("ui/app.ui")
    mw.setWindowIcon(QIcon("ui/icon.png"))
    if not mw: raise RuntimeError("Không load được UI")

    # --- Tab Ball ---
    ctrl_ball = VideoTabController(
        mw,
        video_label = mw.findChild(QLabel, "videoLabelTracking"),
        btn_open    = mw.findChild(QPushButton, "btnOpenTracking"),
        btn_start   = mw.findChild(QPushButton, "btnStartTracking"),
        btn_pause   = mw.findChild(QPushButton, "btnPauseTracking"),
        processor   = BallProcessor()
    )

    # --- Tab Pose ---
    ctrl_pose = VideoTabController(
        mw,
        video_label = mw.findChild(QLabel, "videoLabelPose"),
        btn_open    = mw.findChild(QPushButton, "btnOpenPose"),
        btn_start   = mw.findChild(QPushButton, "btnStartPose"),
        btn_pause   = mw.findChild(QPushButton, "btnPausePose"),
        processor   = PoseProcessor()
    )

    # ---------- TAB: Pose Compare ----------
    cmp_ctrl = CompareTabController(mw)

    # Dừng controller không hoạt động khi đổi tab (tránh chiếm camera)
    tabw = mw.findChild(QTabWidget, "tabWidget")
    if tabw:
        def on_tab_changed(idx):
            if idx == 0:        
                ctrl_pose.stop()  
            elif idx == 1:      
                ctrl_ball.stop()
            elif idx == 2:
                cmp_ctrl.stop_previews()
        tabw.currentChanged.connect(on_tab_changed)

    # scale lại khi resize
    old_resize = mw.resizeEvent
    def resizeEvent(e):
        ctrl_ball.on_resize()
        ctrl_pose.on_resize()
        if old_resize: old_resize(e)
    mw.resizeEvent = resizeEvent

    mw.show()
    sys.exit(app.exec())
