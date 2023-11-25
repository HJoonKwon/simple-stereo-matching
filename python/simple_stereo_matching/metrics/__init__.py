from .ssd import SSD
from .sad import SAD
from .ncc import NCC    

METRIC_CLASSES = {"ssd": SSD, "sad": SAD, "ncc": NCC}
