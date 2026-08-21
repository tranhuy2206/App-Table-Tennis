from PySide6.QtCore import QObject, Signal
import Compare_Pose as CP

# Worker chạy trong thread nền
class CompareWorker(QObject):
    progress = Signal(str)          
    finished = Signal(dict)         
    failed   = Signal(str)

    def __init__(self, ref_path, stu_path, n_points=100, weights=None, action_name=None, use_3d=False):
        super().__init__()
        self.ref_path = ref_path
        self.stu_path = stu_path
        self.n_points = n_points
        self.weights = weights
        self.action_name = action_name
        self.use_3d = use_3d

    def run(self):
        try:
            result = CP.compare_videos(
                self.ref_path,
                self.stu_path,
                n_points=self.n_points,
                weights=self.weights,
                action_name=self.action_name,
                use_3d=self.use_3d,
                lang="en",
                progress_callback=self.progress.emit,
            )
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))
