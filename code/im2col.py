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
    # Step 1: Parse stride, padding, and dilation parameters
    # Convert single int values to (value, value) tuples for height and width
    sH, sW = _pair(stride)  # stride in height and width dimensions
    pH, pW = _pair(padding)  # padding in height and width dimensions
    dH, dW = _pair(dilation)  # dilation in height and width dimensions

    # Extract input dimensions
    C, H, W = x.shape

    # Step 2: Compute effective kernel size with dilation
    # Dilation creates gaps between kernel elements
    # Effective size = (kernel_size - 1) * dilation + 1
    kH_eff = (kH - 1) * dH + 1  # effective kernel height
    kW_eff = (kW - 1) * dW + 1  # effective kernel width

    # Step 3: Compute output spatial dimensions
    # Formula: out_size = (in_size + 2*padding - effective_kernel_size) / stride + 1
    H_out = (H + 2 * pH - kH_eff) // sH + 1
    W_out = (W + 2 * pW - kW_eff) // sW + 1

    # Pad the input tensor
    # np.pad adds pH rows of zeros at top/bottom and pW columns at left/right
    # ((0, 0), ...) means no padding on the channel dimension
    x_padded = np.pad(
        x, ((0, 0), (pH, pH), (pW, pW)), mode="constant", constant_values=0
    )

    # Step 4: Extract each sliding window into a column
    # Initialize output array to hold all columns
    # Shape: (C * kH * kW, H_out * W_out)
    cols = np.zeros((C * kH * kW, H_out * W_out), dtype=x.dtype)

    # Column index counter
    col_idx = 0

    # Iterate over each output position
    for i_out in range(H_out):
        for j_out in range(W_out):
            # Calculate the top-left position of the sliding window in the padded input
            # Using stride to determine starting position
            i_start = i_out * sH  # starting row in padded input
            j_start = j_out * sW  # starting column in padded input

            # Extract the receptive field (sliding window) for this output position
            # For each kernel position (ki, kj), extract values with dilation
            patch = []
            for c in range(C):  # for each input channel
                for ki in range(kH):  # for each kernel row
                    for kj in range(kW):  # for each kernel column
                        # Calculate actual position in padded input with dilation
                        i_actual = i_start + ki * dH
                        j_actual = j_start + kj * dW

                        # Extract the value at this position
                        patch.append(x_padded[c, i_actual, j_actual])

            # Store the flattened patch as a column
            # Shape of patch: (C * kH * kW,)
            cols[:, col_idx] = patch
            col_idx += 1

    return cols
