import json
import matplotlib.pyplot as plt

modelname = "xlan"

with open("results-{}.json".format(modelname), "r") as file:
    resultsmap = json.load(file)

c = 0
ofile = open("results-{}.csv".format(modelname), "w")
for k in resultsmap.keys():
    maxval = -100
    imaxval = ""
    minval = 100
    iminval = ""
    nval = 100
    inval = ""
    for r in resultsmap[k]:
        parts = r.split(":")
        pdiff = float(parts[1])
        if pdiff > maxval:
            maxval = pdiff
            imaxval = parts[0]
        if pdiff < minval:
            minval = pdiff
            iminval = parts[0]
        if abs(pdiff) < nval:
            nval = pdiff
            inval = parts[0]
    ofile.write(
        k
        + ","
        + "{:0.3f}".format(maxval)
        + ","
        + "{:0.3f}".format(minval)
        + ","
        + "{:0.3f}".format(nval)
        + ","
        + imaxval
        + ","
        + iminval
        + ","
        + inval
        + "\n"
    )
    c = c + 1
    # if c>1:
    #    break
ofile.close()


# IMAGE GRID TO DISPLAY EXPLANATIONS

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img_width, img_height = 400, 300


def resize_img_to_array(img, img_shape=(244, 244)):
    img_array = np.array(img.resize(img_shape))

    return img_array


def image_grid(fn_images: list, text: list = [], top: int = 8, per_row: int = 4):
    """
    fn_images is a list of image paths.
    text is a list of annotations.
    top is how many images you want to display
    per_row is the number of images to show per row.
    """
    for i in range(len(fn_images[:top])):
        if i % 4 == 0:
            _, ax = plt.subplots(
                1, per_row, sharex="col", sharey="row", figsize=(24, 6)
            )
        j = i % 4
        image = Image.open(fn_images[i])
        image = resize_img_to_array(image, img_shape=(img_width, img_height))

        ax[j].imshow(image)
        ax[j].axis("off")
        if text:
            ax[j].annotate(
                text[i],
                (0, 0),
                (0, -32),
                xycoords="axes fraction",
                textcoords="offset points",
                va="top",
            )


# IMAGE GRID OF EXPLANATIONS FROM RESULTS FILE


def makeimgstr(val):
    N = 12
    v = N - len(str(val))
    s = ""
    for i in range(0, v, 1):
        s = s + "0"
    s = s + str(val)
    return s


curr_row = 0
trainprefix = "./trainimages/train2014/COCO_train2014_"
testprefix = "./sample_images/COCO_val2014_"

C = 0
ifile = open("results-{}.csv".format(modelname))
res = []
for ln in ifile:
    res.append(ln.strip())
imagelist = []
for i in range(0, len(res), 1):
    col = i
    parts = res[i].split(",")
    if len(parts) < 7:
        continue
    found = True
    for p in parts:
        if len(p) < 2:
            found = False
            break
    if not found:
        continue
    T = testprefix + makeimgstr(parts[0]) + ".jpg"
    Mx = trainprefix + makeimgstr(parts[4]) + ".jpg"
    Mn = trainprefix + makeimgstr(parts[5]) + ".jpg"
    Nt = trainprefix + makeimgstr(parts[6]) + ".jpg"
    imagelist.append(T)
    imagelist.append(Mx)
    imagelist.append(Mn)
    imagelist.append(Nt)

    """
    At = plt.imread(T)
    AMx = plt.imread(Mx)
    AMn = plt.imread(Mn)
    ANt = plt.imread(Nt)
    axarr[col,curr_row].imshow(At)
    axarr[col,curr_row+1].imshow(AMx)
    axarr[col,curr_row+2].imshow(AMn)
    axarr[col,curr_row+3].imshow(ANt)
    
    curr_row = 0
    """
    C = C + 1
    if C > 10:
        break

ifile.close()

image_grid(imagelist, [], 32)
