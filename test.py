from PIL import Image
import numpy as np
import torch
from jump_flooding import jump_flooding


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def visualize_flow(flow: np.ndarray):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow (np.ndarray): Flow UV image of shape [..., H, W, 2]

    Returns:
        np.ndarray: Flow visualization image of shape [H, W, 3]
    """
    assert flow.ndim >= 3, 'input flow must have at least three dimensions'
    assert flow.shape[-1] == 2, 'input flow must have shape [..., H, W, 2]'
    u = flow[..., 0]
    v = flow[..., 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad, axis=(-2, -1), keepdims=True)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    flow_image = np.zeros((*flow.shape[:-1], 3), dtype=flow.dtype)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(3):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        flow_image[..., i] = col

    return flow_image


img = np.array(Image.open('test.png').convert('L')).astype(np.float32) / 255.0
img = -img + 0.5
data = torch.from_numpy(img).cuda().unsqueeze(-1)
result = jump_flooding(data)
print(result.min(), result.max())
result_np = result.cpu().numpy()
vis = visualize_flow(result_np) * 255.0
Image.fromarray(vis.astype(np.uint8)).save("result.png")