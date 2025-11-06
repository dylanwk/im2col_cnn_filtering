import numpy as np
from code.im2col import im2col

def test_im2col_shape():
    x = np.arange(1*5*5).reshape(1,5,5).astype(np.float32)
    cols = im2col(x, 3, 3, stride=1, padding=0)
    assert cols.shape == (9, 9), f"Expected (9,9), got {cols.shape}"
