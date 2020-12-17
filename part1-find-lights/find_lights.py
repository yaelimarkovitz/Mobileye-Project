try:
    print("Elementary imports: ")
    import os
    import json
    import glob
    import argparse

    print("numpy/scipy imports:")
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter

    print("PIL imports:")
    from PIL import Image

    print("matplotlib imports:")
    import matplotlib.pyplot as plt

    print("skimage imports:")
    from skimage.feature import peak_local_max
except ImportError:
    print("Need to fix the installation")
    raise
print("All imports okay. Yay!")


# ==============Auxiliary Functions===========================
def find_lights(image: np.ndarray, kernel: np.ndarray, c_image: np.ndarray):
    result = sg.convolve2d(image, kernel)
    max_dots = peak_local_max(result, min_distance=20, num_peaks=10)
    x = []
    y = []
    for i in max_dots:
        x.append(i[1])
        y.append(i[0])
    return x, y


# ====================== find lights ========================
def find_tfl_lights(c_image: np.ndarray, **kwargs):

    kernel_r = np.array(
        [[-0.5, -0.5, -0.5],
         [-0.5, -0.5, -0.5],
         [-0.5, -0.5, -0.5],
         [-0.5, -0.5, -0.5],
         [-0.5, -0.5, -0.5],
         [-0.5, -0.5, -0.5],
         [-0.5, -0.5, -0.5],
         [-0.5, -0.5, -0.5],
         [-0.5, -0.5, -0.5],
         [1, 1, -0.5],
         [1, 2, 1],
         [1, 2, 1],
         [1, 2, 1], [1, 1, 1]])
    kernel_g = np.array(
        [[1, 1, 1],
         [1, 2, 1],
         [1, 2, 1],
         [1, 2, 1],
         [1, 1, -0.5],
         [1, 1, -0.5],
         [-0.5, -0.5, -0.5],
         [-0.5, -0.5, -0.5],
         [-0.5, -0.5, -0.5],
         [-0.5, -0.5, -0.5],
         [-0.5, -0.5, -0.5],
         [-0.5, -0.5, -0.5],
         [-0.5, -0.5, -0.5],
         [-0.5, -0.5, -0.5],
         [-0.5, -0.5, -0.5], ])
    x_green, y_green = find_lights(c_image[:, :, 1], kernel_g, c_image)
    x_red, y_red = find_lights(c_image[:, :, 0], kernel_r, c_image)
    candidates = [[x,y] for x,y in zip(x_red,y_red)]
    auxiliary = [1]*len(x_red)
    candidates += [[x,y] for x ,y, in zip(x_green,y_green)]
    auxiliary += [0]*len(x_green)

    return candidates, auxiliary


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


# ==============================TEST=====================================
def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]
    show_image_and_gt(image, objects, fig_num)
    red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def main(argv=None):
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = 'pictures'
    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)
    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)

#
# if __name__ == '__main__':
#     main()
