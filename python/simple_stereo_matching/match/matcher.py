import torch
from simple_stereo_matching.metrics import METRIC_CLASSES


class Matcher:
    def __init__(self):
        super().__init__()

    def compute_disparity(
        self,
        left_img: torch.Tensor,
        right_img: torch.Tensor,
        max_disparity: int,
        window_size: int,
        metric_name: str = "ssd",
    ) -> torch.Tensor:
        """Computes the disparity map for a pair of images.

        Args:
            left_img (torch.Tensor): Left image
            right_img (torch.Tensor): Right image
            max_disparity (int): maximum available disparity
            window_size (int): size of the window used to compute the cost volume

        Returns:
            torch.Tensor: Disparity map
        """

        metric_name = metric_name.lower()
        metric_class = METRIC_CLASSES.get(metric_name)
        if metric_class:
            metric = metric_class()
            cost_volume = metric.compute(
                left_img, right_img, max_disparity, window_size
            )
            best_disparity_map = metric.get_best(cost_volume)
        else:
            raise NotImplementedError(f"Metric {metric_name} is not supported")

        return best_disparity_map
