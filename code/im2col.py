import numpy as np

def _pair(x):
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return int(x[0]), int(x[1])
    return int(x), int(x)

def im2col(x, kH, kW, stride=1, padding=0, dilation=1):
    """TODO: Convert 2D convolution into matrix multiplication via columnization.
    Args:
        x: (C, H, W) input
        kH, kW: kernel height/width
        stride: int or (sH, sW)
        padding: int or (pH, pW)
        dilation: int or (dH, dW)
    Returns:
        cols: (C*kH*kW, outH*outW)
    Steps to implement:
      1) Parse stride/padding/dilation with _pair (provided).
      2) Compute effective kernel size with dilation.
      3) Compute out spatial dims (H_out, W_out) and pad the input.
      4) Extract each sliding window into a column and stack.
    """
    # TODO: your code here
    raise NotImplementedError("Implement im2col as described in the docstring.")
