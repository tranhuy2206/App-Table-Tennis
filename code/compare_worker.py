from PySide6.QtCore import QObject, QThread, Signal
import Compare_Pose as CP

# Worker chạy trong thread nền
class CompareWorker(QObject):
    progress = Signal(str)          
    finished = Signal(dict)         
    failed   = Signal(str)

    def __init__(self, ref_path, stu_path, n_points=100, weights=None, action_name=None):
        super().__init__()
        self.ref_path = ref_path
        self.stu_path = stu_path
        self.n_points = n_points
        self.weights = weights
        self.action_name = action_name

    def run(self):
        try:
            featsA = CP.extract_features(self.ref_path, draw=False)

            featsB = CP.extract_features(self.stu_path, draw=False)

            A_rs = CP.resample_features(featsA, n=self.n_points)
            B_rs = CP.resample_features(featsB, n=self.n_points)

            result = CP.compare_features_DTW(A_rs, B_rs, weights=self.weights, action_name=self.action_name, window_ratio=0.1)
            
            # Lấy weights thực tế đã được sử dụng
            if self.weights is None:
                actual_weights = CP.get_weights_for_action(self.action_name)
            else:
                actual_weights = self.weights

            per_feature_err, T = CP.compute_per_frame_errors(A_rs, B_rs, result["paths"], actual_weights)

            agg_err = CP.aggregate_error(per_feature_err, actual_weights)

            feedback = CP.build_feedback_messages(
                agg_err, per_feature_err, actual_weights, fps=30,
                overall_threshold=0.35, overall_min_duration=0.50, overall_min_coverage_ratio=0.10,
                per_feature_threshold=0.35, per_feature_min_duration=0.30,
                lang="en"  
                )
            
            result["feedback"] = feedback                   
            result["error_messages"] = feedback["messages"] if feedback.get("should_warn") else []

            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))
