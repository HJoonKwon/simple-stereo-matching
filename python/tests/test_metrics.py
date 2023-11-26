import torch
from simple_stereo_matching.metrics.ssd import SSD


def test_ssd():
    ssd = SSD()
    left_img = torch.randn((32, 32))
    rigth_img = torch.randn((32, 32))
    max_disparity = 10
    ssd_cost_volume = ssd.compute(
        left_img, rigth_img, max_disparity=max_disparity, window_size=3
    )
    assert (
        ssd_cost_volume.shape[0] == max_disparity
        and ssd_cost_volume.shape[1] == left_img.shape[0]
        and ssd_cost_volume.shape[2] == left_img.shape[1]
    )
    best_ssd = ssd.get_best(ssd_cost_volume)
    assert (
        best_ssd.shape[0] == left_img.shape[0]
        and best_ssd.shape[1] == left_img.shape[1]
    )
