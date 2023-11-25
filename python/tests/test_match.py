import torch 
from simple_stereo_matching.match.matcher import Matcher

def test_matcher():
    matcher = Matcher()
    left_img = torch.randn((1, 32, 32))
    right_img = torch.randn((1, 32, 32))
    max_disparity = 10
    window_size = 3 
    metric_name = 'ssd'
    disparity_map = matcher.compute_disparity(left_img, right_img, max_disparity, window_size, metric_name)
    assert (
        disparity_map.shape[0] == left_img.shape[1]
        and disparity_map.shape[1] == left_img.shape[2]
    )