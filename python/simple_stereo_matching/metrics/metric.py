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

    @staticmethod
    def _get_windows(img: torch.Tensor, window_size: int):
        """Get the windows of the image.

        Args:
            img (torch.Tensor): image
            window_size (int): window size

        Returns:
            torch.Tensor: windows of the image
        """
        assert (
            len(img.shape) == 2
        ), f"Image shape {img.shape} should be (H, W) grayscale"
        img = img.unsqueeze(0)
        height, width = img.shape[-2], img.shape[-1]
        unfold_ops = torch.nn.Unfold(kernel_size=window_size, padding=window_size // 2)
        windows = unfold_ops(img).view(-1, window_size**2, height, width)
        return windows

    @staticmethod
    def get_best(cost_volume: torch.Tensor) -> torch.Tensor:
        assert len(cost_volume.shape) == 3
        best_matrix = torch.argmin(cost_volume, dim=0)
        assert len(best_matrix.shape) == 2
        return best_matrix
