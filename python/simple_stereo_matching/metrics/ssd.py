from simple_stereo_matching.metrics.metric import Metric


class SSD(Metric):
    """Sum of Squared Differences (SSD)"""

    def __init__(self):
        super().__init__()

    def compute(self, left_img, right_img, max_disparity, window_size):
        pass
