class ProcessorBase:
    def process(self, frame_bgr):
        """
        Input: frame_bgr (numpy, OpenCV)
        Output: (frame_out_bgr, side_data)
        """
        raise NotImplementedError