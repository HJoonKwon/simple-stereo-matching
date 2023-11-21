from simple_stereo_matching.metrics.metric import Metric


class NCC(Metric):
    """Normalized Cross Correlation (NCC)"""

    def __init__(self):
        super().__init__()

    def compute(self, left_img, right_img, max_disparity, window_size):
        pass
