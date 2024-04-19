# ACM-MM-TRAINING
# CREATES THE WEIGHTS FROM TRAINING DATA

import math
import utilities as ut
from collections import defaultdict


"""
def neglog(c,r):
    v1 = -1*math.log(1 + math.exp(10*(0.7-c)))
    v2 = -1*math.log(1 + math.exp(1*(0.7-r)))
    return math.exp(v1)+math.exp(v2)
"""
multicaptionclipfile = "allCaptionTrain.json"
allTrainCscoreTrainfile = "./allTripClipScore/allTripCscoreTrain.json"


def neglog(c):
    v1 = -1 * math.log(1 + math.exp(10 * (0.7 - c)))
    return math.exp(v1)


def getCapClipScore():
    capclipscores = {}
    multicaptionclips = ut.readJson(multicaptionclipfile)
    for k in multicaptionclips.keys():
        parts = k.split("_")
        img = parts[len(parts) - 1].lstrip("0")
        capclipscores[img] = multicaptionclips[k]["CLIPScore"]

    return capclipscores


def train():

    rclip_contents = ut.readJson(allTrainCscoreTrainfile)
    ofile = open("hmln-trainingdata.txt", "w")
    TrainedFormulas = defaultdict(list)
    capclipscores = getCapClipScore()
    formulaimgmap = ut.readJson("formulaimgmap.json")

    for k in formulaimgmap.keys():
        parts = k.split(":")
        P1 = parts[0]
        P2 = parts[1]

        t = formulaimgmap[k][0]
        C = capclipscores[t]

        L = int(len(rclip_contents[t]) / 2)

        p1del = 0
        p2del = 0
        altdel = 0
        altcnt = 0

        for j in range(0, L, 1):
            ix = L + j
            if rclip_contents[t][j] == P1:
                p1del = neglog(rclip_contents[t][ix])
            elif rclip_contents[t][j] == P2:
                p2del = neglog(rclip_contents[t][ix])
            else:
                altdel = altdel + neglog(rclip_contents[t][ix])
                altcnt = altcnt + 1

        if altcnt != 0:
            TrainedFormulas[k] = [(altdel / altcnt), p1del, p2del, neglog(C)]
        else:
            TrainedFormulas[k] = [neglog(0.25), neglog(0.25), neglog(0.25), neglog(C)]

    ofile.close()

    ut.writeJson("trainedFormulas.json", TrainedFormulas)
