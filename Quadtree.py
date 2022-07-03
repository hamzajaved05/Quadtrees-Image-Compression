import numpy as np
import cv2 as cv

class Quadtree:
    def __init__(self, data, factor = 15):
        self.child = None
        self.val = None
        self.compress(data, factor)
        
    def get_quadrants(self, data):
        shape = data.shape[:2]
        siz = shape[0] // 2
        return data[:siz, :siz], data[:siz, siz:], data[siz:, :siz], data[siz:, siz:]

    def compress(self, data, factor):
        shape = data.shape[:2]
        lin_data = data.reshape(-1, data.shape[-1])
        var = np.linalg.norm(lin_data.max(0) - lin_data.min(0))
        if var >= factor:
            res = []
            Q = self.get_quadrants(data)
            for q in Q:
                res.append(Quadtree(q, factor = factor))
            self.child = res
        else:
            avg = lin_data.mean(0).astype("uint8")
            self.val = avg

    def convert2np(self, size = 512, plot_edges = False):
        img = np.zeros((size, size, 3))
        img = convert(self, img, x = [0, size], y = [0, size], plot_edges = plot_edges)
        return img.astype(int)


def convert(quad, img, x, y, plot_edges = False):
    if quad.val is None:
            x_n = (x[0] + x[1]) // 2
            y_n = (y[0] + y[1]) // 2
            convert(quad.child[0], img, [x[0], x_n], [y[0], y_n], plot_edges = plot_edges)
            convert(quad.child[1], img, [x_n, x[1]], [y[0], y_n], plot_edges = plot_edges)
            convert(quad.child[2], img, [x[0], x_n], [y_n, y[1]], plot_edges = plot_edges)
            convert(quad.child[3], img, [x_n, x[1]], [y_n, y[1]], plot_edges = plot_edges)
    else:
        img[y[0]:y[1], x[0]:x[1]] = quad.val
        if plot_edges:
            img = cv.rectangle(img, (x[0], y[0]), (x[1], y[1]), (0,0,0), 1)
    return img
