import numpy as np
from im2col import im2col


def conv2d_im2col(img, weight, bias=0.0, stride=1, padding=0, dilation=1):
    # Extract dimensions from the weight tensor
    # C_out: number of output channels (filters)
    # kH, kW: kernel height and width
    C_out, _, kH, kW = weight.shape
    H, W = img.shape

    # Add a channel dimension to the grayscale image: (H, W) -> (1, H, W)
    img_with_channel = img[np.newaxis, :, :]  # Shape: (1, H, W)

    # cols shape: (1*kH*kW, outH*outW) = (kH*kW, outHW)
    cols = im2col(
        img_with_channel, kH, kW, stride=stride, padding=padding, dilation=dilation
    )

    # This flattens each filter into a row vector
    weight_reshaped = weight.reshape(C_out, -1)  # Shape: (C_out, kH*kW)

    # Each row in weight_reshaped is one filter, and we convolve all filters at once
    out_flat = weight_reshaped @ cols  # Shape: (C_out, outHW)

    # If bias is a scalar, it broadcasts across all channels
    # If bias is (C_out,), each channel gets its corresponding bias value
    out_flat = (
        out_flat + bias[:, np.newaxis]
        if isinstance(bias, np.ndarray)
        else out_flat + bias
    )

    # Compute output spatial dimensions
    # Formula: out_size = (in_size + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1
    H_out = (H + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (kW - 1) - 1) // stride + 1

    # Reshape from (C_out, outHW) to (C_out, H_out, W_out)
    out = out_flat.reshape(C_out, H_out, W_out)

    return out


def kernels():
    gaussian_3x3 = (1 / 16.0) * np.array(
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32
    )
    sharpen_3x3 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    def pack(*mats):
        return np.stack([m[np.newaxis, :, :] for m in mats], axis=0)

    return {
        "gaussian_3x3": pack(gaussian_3x3),
        "sharpen_3x3": pack(sharpen_3x3),
        "sobel_x": pack(sobel_x),
        "sobel_y": pack(sobel_y),
    }
