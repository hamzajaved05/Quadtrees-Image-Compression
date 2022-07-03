import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from compression import perform_compression, get_levels


def main():
    img = cv.imread("test.png")
    img_compressed = perform_compression(img, factor = 20)
    n_levels = get_levels(img_compressed)
    img_c = img_compressed.convert2np(plot_edges= False)
    
    stacked = np.hstack([img, img_c])
    plt.imshow(stacked[:, :, ::-1])
    plt.show()
    print(f"Done | Reduced from {np.size(img)} --> {n_levels}")

if __name__ =="__main__":
    main()