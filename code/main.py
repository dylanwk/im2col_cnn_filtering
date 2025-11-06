import argparse, os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from im2col import im2col
from filters import conv2d_im2col, kernels

def to_gray01(img):
    if img.mode != 'L':
        img = img.convert('L')
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr

def save_img(path, arr):
    arr = np.clip(arr, 0, 1)
    Image.fromarray((arr*255).astype('uint8')).save(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default='../data/sample.png')
    ap.add_argument('--save_dir', default='../results')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--padding', type=int, default=1, help='Try 0 or 1 for 3x3 kernels')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    img = Image.open(args.image)
    g = to_gray01(img)

    K = kernels()

    print(K['gaussian_3x3'].shape)

    blur = conv2d_im2col(g, K['gaussian_3x3'], stride=args.stride, padding=args.padding)[0]
    sharp = conv2d_im2col(g, K['sharpen_3x3'], stride=args.stride, padding=args.padding)[0]
    gx = conv2d_im2col(g, K['sobel_x'], stride=args.stride, padding=args.padding)[0]
    gy = conv2d_im2col(g, K['sobel_y'], stride=args.stride, padding=args.padding)[0]
    mag = np.sqrt(gx*gx + gy*gy)

    save_img(os.path.join(args.save_dir, 'blur.png'), blur)
    save_img(os.path.join(args.save_dir, 'sharpen.png'), sharp)
    save_img(os.path.join(args.save_dir, 'sobel_x.png'), gx)
    save_img(os.path.join(args.save_dir, 'sobel_y.png'), gy)
    save_img(os.path.join(args.save_dir, 'edges.png'), mag)

    # Grid
    fig = plt.figure(figsize=(10,7))
    ax1 = fig.add_subplot(2,3,1); ax1.imshow(g, cmap='gray'); ax1.set_title('Original'); ax1.axis('off')
    ax2 = fig.add_subplot(2,3,2); ax2.imshow(blur, cmap='gray'); ax2.set_title('Gaussian blur'); ax2.axis('off')
    ax3 = fig.add_subplot(2,3,3); ax3.imshow(sharp, cmap='gray'); ax3.set_title('Sharpen'); ax3.axis('off')
    ax4 = fig.add_subplot(2,3,4); ax4.imshow(gx, cmap='gray'); ax4.set_title('Sobel X'); ax4.axis('off')
    ax5 = fig.add_subplot(2,3,5); ax5.imshow(gy, cmap='gray'); ax5.set_title('Sobel Y'); ax5.axis('off')
    ax6 = fig.add_subplot(2,3,6); ax6.imshow(mag, cmap='gray'); ax6.set_title('Edge magnitude'); ax6.axis('off')
    fig.tight_layout()
    grid_path = os.path.join(args.save_dir, 'grid.png')
    fig.savefig(grid_path, dpi=150)
    print(f"Saved outputs to {args.save_dir}")

if __name__ == '__main__':
    main()
