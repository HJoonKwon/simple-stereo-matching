import torch 
from simple_stereo_matching.match.matcher import Matcher

def test_matcher():
    left_img = Matcher.read_img_as_tensor('../data/Adirondack_left.png')
    right_img = Matcher.read_img_as_tensor('../data/Adirondack_right.png')
    assert len(left_img.shape) == 2 and len(right_img.shape) == 2
    max_disparity = left_img.shape[1]
    window_size = 3 
    metric_name = 'ssd'
    disparity_map = Matcher.compute_disparity(left_img, right_img, max_disparity, window_size, metric_name)
    assert (
        disparity_map.shape[0] == left_img.shape[0]
        and disparity_map.shape[1] == left_img.shape[1]
    )