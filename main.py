import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from compression import perform_compression, get_levels
import argparse


def main(args):
    img = cv.imread(args.img)
    img_compressed = perform_compression(img, factor = args.threshold)
    n_levels = get_levels(img_compressed)
    img_c = img_compressed.convert2np(plot_edges= args.plot_edges != 0)

    stacked = np.hstack([img, img_c])

    if args.plot:
        plt.imshow(stacked[:, :, ::-1])
        plt.show()

    if args.save:
        cv.imwrite("data/output.jpg", stacked)

    print(f"Done | Reduced from {np.size(img)} --> {n_levels}")

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Convert the image to Quadtrees and back')
    parser.add_argument('--img', type=str, default= "data/test.png", help='Path to the image')
    parser.add_argument('--threshold', type=int, default = 50, help='Threshold for Quadtrees')
    parser.add_argument('--plot_edges', type=int, default = 0, help='Plot edgges to visualize the quadtree structure | 0 for False else True')
    parser.add_argument('--save', type=int, default = 1, help='Save the output into jpeg file | 0 for False else True')
    parser.add_argument('--plot', type=int, default = 0, help='Plot the output | 0 for False else True')
    args = parser.parse_args()
    main(args)