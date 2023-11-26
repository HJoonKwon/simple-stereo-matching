import sys, os

sys.path.append(os.path.realpath(os.path.join(__file__, "../../python")))

import matplotlib.pyplot as plt
from simple_stereo_matching.match.matcher import Matcher


def main():
    left_img = Matcher.read_img_as_tensor("../data/Adirondack_left.png")
    right_img = Matcher.read_img_as_tensor("../data/Adirondack_right.png")
    disparity_map = Matcher.compute_disparity(
        left_img,
        right_img,
        max_disparity=60,
        window_size=3,
        metric_name="ssd",
    )
    disparity_map += 1
    depth_map = 1 / disparity_map.float() * 255
    plt.imshow(depth_map, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
