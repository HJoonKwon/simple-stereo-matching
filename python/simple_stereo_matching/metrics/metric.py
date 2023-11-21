import torch
from abc import ABC, abstractmethod


class Metric(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def compute(
        self,
        left_img: torch.Tensor,
        right_img: torch.Tensor,
        max_disparity: int,
        window_size: int,
    ):
        """Compute the metric for stereo matching.

        Args:
            left_img (torch.Tensor): image from the left camera
            right_img (torch.Tensor): image from the right camera
            max_disparity (int): maximum disparity value (in pixels)
            window_size (int): window size for computing the metric
        """
        pass
