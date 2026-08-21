import os
from PySide6.QtWidgets import QFileDialog, QMessageBox, QLabel, QPushButton, QTextEdit, QListWidget, QLCDNumber, QListWidgetItem, QAbstractItemView, QComboBox
from PySide6.QtGui import QTextCursor
from PySide6.QtCore import QThread, Qt
from compare_worker import CompareWorker
from processor_pose import PoseProcessor
from autoplay_preview import AutoplayPreview
import Compare_Pose as CP

class CompareTabController:
    def __init__(self, mw):
        self.mw = mw
        
        self.btnPickRef    = mw.findChild(QPushButton, "btnPickRef")
        self.btnPickStu    = mw.findChild(QPushButton, "btnPickStu")
        self.btnRunCompare = mw.findChild(QPushButton, "btnRunCompare")
        self.listErrors = self.mw.findChild(QListWidget, "listErrors")
        self._tune_error_list()

        self.lblRefPath = mw.findChild(QLabel, "lblRefPath")
        self.lblStuPath = mw.findChild(QLabel, "lblStuPath")

        self.lcdScore    = mw.findChild(QLCDNumber, "lcdScore")
        self.videoTeacher = mw.findChild(QLabel, "videoTeacher")
        self.videoStudent = mw.findChild(QLabel, "videoStudent")
        self.comboAction = mw.findChild(QComboBox, "comboAction")
        self.console = mw.findChild(QTextEdit, "outputConsole")
        
        # Setup action combo box
        self._setup_action_combo()

        # --- State ---
        self.ref_path = None
        self.stu_path = None
        self._thread = None
        self._worker = None
        self.cap_teacher = None 
        self.cap_student = None
        self.use_3d = True  # Mặc định bật 3D để xử lý tốt hơn khi góc quay khác nhau

        self.teacher_preview = AutoplayPreview(
            mw, self.videoTeacher, processor=PoseProcessor(), interval_ms=33, loop=True
        )

        self.student_preview = AutoplayPreview(
            mw, self.videoStudent, processor=PoseProcessor(), interval_ms=33, loop=True
        )

        # --- Wire events ---
        if self.btnPickRef:    self.btnPickRef.clicked.connect(self.pick_ref)
        if self.btnPickStu:    self.btnPickStu.clicked.connect(self.pick_stu)
        if self.btnRunCompare: self.btnRunCompare.clicked.connect(self.run_compare)

    

    # ---------------- Actions ----------------
    def pick_ref(self): 
        path, _ = QFileDialog.getOpenFileName(self.mw, "Select Reference Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if not path: return
        self.ref_path = path
        if self.lblRefPath:
            self.lblRefPath.setText(os.path.basename(path))
            self.lblRefPath.setToolTip(path)
        self.teacher_preview.set_source(path)

    def pick_stu(self):
        path, _ = QFileDialog.getOpenFileName(self.mw, "Select Target Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if not path: return
        self.stu_path = path
        if self.lblStuPath:
            self.lblStuPath.setText(os.path.basename(path))
            self.lblStuPath.setToolTip(path)
        self.student_preview.set_source(path)

    def run_compare(self):
        if not self.ref_path or not self.stu_path:
            QMessageBox.warning(self.mw, "Missing file", "Please select both files before comparing.")
            return
        
        self.stop_previews()

        # Disable pick/start during run
        for btn in (self.btnPickRef, self.btnPickStu, self.btnRunCompare):
            if btn: btn.setEnabled(False)
        if self.comboAction:
            self.comboAction.setEnabled(False)

        # Get selected action
        action_name = None
        if self.comboAction:
            current_index = self.comboAction.currentIndex()
            if current_index >= 0:
                action_name = self.comboAction.itemData(current_index)
        
        # Start worker thread
        self._thread = QThread(self.mw)
        self._worker = CompareWorker(self.ref_path, self.stu_path, n_points=80, weights=None, action_name=action_name, use_3d=self.use_3d)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._log)

        def on_done(result: dict):
            # 1) Log summary
            score = result.get("overall_score", result.get("weighted_score"))

            if self.lcdScore is not None and score is not None:
                try:
                    self.lcdScore.display(int(round(float(score))))
                except Exception:
                    self.lcdScore.display(0)

            self._errors_clear()
            errs = result.get("error_messages", [])
            if errs:
                for m in errs:
                    self._errors_add(m)
            else:
                self._errors_add("Great alignment overall. Keep it up!")
            for warning in result.get("quality", {}).get("warnings", []):
                self._errors_add(f"Data quality: {warning}")

            # Re-enable buttons & cleanup thread
            for btn in (self.btnPickRef, self.btnPickStu, self.btnRunCompare):
                if btn: btn.setEnabled(True)
            if self.comboAction:
                self.comboAction.setEnabled(True)
            self._thread.quit()
            self._thread.wait()
            self._worker.deleteLater()
            self._thread.deleteLater()

        def on_fail(msg: str):
            self._errors_clear()
            self._errors_add("ERROR:")
            self._errors_add(msg.splitlines()[-1] if msg else "Unknown error")

            for btn in (self.btnPickRef, self.btnPickStu, self.btnRunCompare):
                if btn: btn.setEnabled(True)
            if self.comboAction:
                self.comboAction.setEnabled(True)
            self._thread.quit()
            self._thread.wait()
            self._worker.deleteLater()
            self._thread.deleteLater()

        self._worker.finished.connect(on_done)
        self._worker.failed.connect(on_fail)
        self._thread.start()

    def stop_previews(self):
        if hasattr(self, "teacher_preview") and self.teacher_preview:
            self.teacher_preview.stop()
        if hasattr(self, "student_preview") and self.student_preview:
            self.student_preview.stop()
    
    def _errors_clear(self):
        if self.listErrors:
            self.listErrors.clear()
    
    def _errors_add(self, text: str):
        if not self.listErrors:
            return
        item = QListWidgetItem(text)
        item.setToolTip(text)  
        self.listErrors.addItem(item)
    
    def _tune_error_list(self):
        lw = self.listErrors
        if not lw: 
            return
        
        f = lw.font(); f.setPointSize(12)   
        lw.setFont(f)
        
        lw.setWordWrap(True)                                   
        lw.setTextElideMode(Qt.ElideNone)                      
        lw.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff) 
        lw.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
    
    def _setup_action_combo(self):
        """Setup ComboBox với danh sách các động tác có sẵn"""
        if not self.comboAction:
            return
        
        # Lấy danh sách động tác
        actions = CP.list_available_actions()
        
        # Thêm vào ComboBox với tên hiển thị
        for action in actions:
            display_name = CP.get_action_display_name(action)
            self.comboAction.addItem(display_name, action)
        
        # Set default selection
        default_index = 0
        for i in range(self.comboAction.count()):
            if self.comboAction.itemData(i) == "default":
                default_index = i
                break
        self.comboAction.setCurrentIndex(default_index)

    # ---------------- Helpers ----------------
    def _log(self, s: str):
        if not getattr(self, "console", None):
            return
        self.console.append(s)
        self.console.moveCursor(QTextCursor.End)

    def clear_log(self):
        if getattr(self, "console", None):
            self.console.clear()
