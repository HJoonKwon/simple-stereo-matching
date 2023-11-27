import torch
from simple_stereo_matching.metrics.metric import Metric


class SSD(Metric):
    """Sum of Squared Differences (SSD)"""

    def __init__(self):
        super().__init__()

    def compute(self, left_img, right_img, max_disparity, window_size):
        assert (
            left_img.shape == right_img.shape
        ), f"left_img and right_img must have the same shape"
        height, width = left_img.shape[-2:]
        cost_volume = torch.zeros(
            (max_disparity, height, width), device=left_img.device
        )

        left_windows = self._get_windows(left_img, window_size)
        right_windows = self._get_windows(right_img, window_size)

        assert (
            len(left_windows.shape) == 4
            and left_windows.shape[0] == 1
            and left_windows.shape[-2] == height
            and left_windows.shape[-1] == width
        ), f"left_windows shape: {left_windows.shape}"

        assert (
            len(right_windows.shape) == 4
            and right_windows.shape[0] == 1
            and right_windows.shape[-2] == height
            and right_windows.shape[-1] == width
        ), f"left_windows shape: {right_windows.shape}"

        for d in range(max_disparity):
            # roll in width dimension (horizontal shift)
            shifted_right_windows = torch.roll(right_windows, d, dims=-1)
            shifted_right_windows[..., :d] = 0

            ssd = torch.sum((left_windows - shifted_right_windows) ** 2, dim=1)
            cost_volume[d] = ssd
        
        return cost_volume