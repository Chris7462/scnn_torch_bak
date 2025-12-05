# Updated to use PyTorch's native TensorBoard support
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class TensorBoard(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        for i, img in enumerate(images):
            # Convert to (C, H, W) format if needed
            if len(img.shape) == 3 and img.shape[2] in [1, 3, 4]:
                img = np.transpose(img, (2, 0, 1))
            self.writer.add_image('%s/%d' % (tag, i), img, step)
        
    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram(tag, values, step)
