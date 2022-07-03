import numpy as np
import cv2 as cv
from Quadtree import Quadtree

def perform_compression(img, *args, **kwargs):
    img_quaded = Quadtree(img, *args, **kwargs)
    return img_quaded

def get_levels(quad):
    count = np.array([0])
    levels = iter_levels(quad)
    return levels

def iter_levels(quad):
    if quad.val is None:
        c0 = iter_levels(quad.child[0])
        c1 = iter_levels(quad.child[1])
        c2 = iter_levels(quad.child[2])
        c3 = iter_levels(quad.child[3])
        return c0 + c1 + c2 + c3
    else:
        return 1