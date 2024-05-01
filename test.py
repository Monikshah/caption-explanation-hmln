# PERFORM INFERENCE

import json
import os
from collections import defaultdict
import math
import random
import pandas as pd
import numpy as np


def neglog(c):
    v1 = -1 * math.log(1 + math.exp(10 * (0.7 - c)))
    return math.exp(v1)


def isSameTriple(F, T):
    P = F.split()
    R = T.split()
    if P[0] == R[0] and P[len(P) - 1] == R[len(R) - 1]:
        return True
    else:
        return False


# models: xlan, sgae, aoa, m2
modelname = "xlan"

# IMPLEMENTS GIBBS SAMPLING FOR COMPUTING MARGINALS FOR PREDICATES AND FORMULAS


def gibbssampler(Vars, Edges, Pots):
    allkeys = []
    varindex = defaultdict(list)
    margs = {}
    margsform = defaultdict(list)
    margformcnts = {}
    cnts = {}
    if len(Vars.keys()) == 0:
        return {}, {}
    for k in Vars.keys():
        allkeys.append(k)
        margs[k] = 0
        cnts[k] = 0
        Vars[k] = random.randint(0, 1)
    for i, e in enumerate(Edges):
        varindex[e[0]].append(i)
        varindex[e[1]].append(i)
        for j in range(0, 4, 1):
            margsform[e[0] + "," + e[1]].append(0)
        margformcnts[e[0] + "," + e[1]] = 0
    T = 20000
    burnin = 500
    statemap = {"0:0": 0, "0:1": 1, "1:0": 2, "1:1": 3}
    for t in range(0, T, 1):
        r = random.randint(0, len(allkeys) - 1)
        # K = Vars[allkeys[r]]
        K = allkeys[r]
        VALP = 1
        VALN = 1
        for j in varindex[K]:
            v = [0, 0]
            v[0] = Edges[j][0]
            v[1] = Edges[j][1]
            if v[0] == K:
                state0 = "0:" + str(Vars[v[1]])
                state1 = "1:" + str(Vars[v[1]])
                val0 = Pots[j][statemap[state0]]
                val1 = Pots[j][statemap[state1]]
                VALP = VALP * val1
                VALN = VALN * val0
            elif v[1] == K:
                state0 = str(Vars[v[0]]) + ":0"
                state1 = str(Vars[v[0]]) + ":1"
                val0 = Pots[j][statemap[state0]]
                val1 = Pots[j][statemap[state1]]
                VALP = VALP * val1
                VALN = VALN * val0
        p = VALP / (VALP + VALN)
        r1 = random.random()
        # origkey = allkeys[K]
        if r1 < p:
            Vars[K] = 1
            if t > burnin:
                margs[K] = margs[K] + 1
                cnts[K] = cnts[K] + 1
        else:
            Vars[K] = 0
            cnts[K] = cnts[K] + 1
        if t > burnin:
            for j in varindex[K]:
                edgetup = Edges[j]
                parts = [0, 0]
                parts[0] = edgetup[0]
                parts[1] = edgetup[1]
                edgekey = edgetup[0] + "," + edgetup[1]
                if Vars[parts[0]] == 0 and Vars[parts[1]] == 0:
                    margsform[edgekey][0] = margsform[edgekey][0] + 1
                elif Vars[parts[0]] == 0 and Vars[parts[1]] == 1:
                    margsform[edgekey][1] = margsform[edgekey][1] + 1
                elif Vars[parts[0]] == 1 and Vars[parts[1]] == 0:
                    margsform[edgekey][2] = margsform[edgekey][2] + 1
                elif Vars[parts[0]] == 1 and Vars[parts[1]] == 1:
                    margsform[edgekey][3] = margsform[edgekey][3] + 1
                margformcnts[edgekey] = margformcnts[edgekey] + 1

    for k in margs:
        if cnts[k] > 0:
            margs[k] = margs[k] / cnts[k]
        else:
            margs[k] = 0
    for k in margsform:
        if margformcnts[k] > 0:
            for i in range(0, 4, 1):
                margsform[k][i] = margsform[k][i] / margformcnts[k]
        else:
            for i in range(0, 4, 1):
                margsform[k][i] = 0
    return margs, margsform


def test():
    groundingmapfile = "./data/groundingmap-{}.json".format(modelname)
    with open(groundingmapfile, "r") as file:
        groundingmap = json.load(file)

    captionmapfile = "./data/captionmap-{}.json".format(modelname)
    with open(captionmapfile, "r") as file:
        captionmap = json.load(file)

    testrel2file = "./data/testrel2formsclip-{}.json".format(modelname)
    with open(testrel2file, "r") as file:
        testrel2f = json.load(file)

    rclipscorefile = "./allTripClipScore/allTripCscore_" + modelname + ".json"
    trainedfile = "./data/trainedFormulas.json"
    formulamapfile = "./data/formulaimgmap.json"

    csvfolder_path = "./data/csv_imageLabelCscore_test_" + modelname

    CapClipScores = {}
    files = os.listdir(csvfolder_path)
    for file in files:
        if file.find(".csv") < 0:
            continue
        data = pd.read_csv(os.path.join(csvfolder_path, file), sep="\t")
        clipcol = "CLIPScore"
        if modelname == "aoa" or modelname == "xlan":
            clipcol = "clipScore"

        imgcol = "imageId"
        if modelname == "sgae" or modelname == "m2":
            imgcol = "image_id"

        for index, row in data.iterrows():
            tmp = int(row[imgcol])
            CapClipScores[str(tmp)] = row[clipcol]

    with open(formulamapfile, "r") as file:
        formulaimgmap = json.load(file)

    with open(trainedfile, "r") as file:
        TrainedFormulas = json.load(file)

    with open(
        "allTripClipScore/missingclips-{}-clipscore.json".format(modelname), "r"
    ) as file:
        missingsamples = json.load(file)

    ct = 0
    cnt = 0
    norels = 0
    # MISSINGCLIPS = defaultdict(list)
    RESULTS = defaultdict(list)
    for gkey in groundingmap.keys():
        if gkey not in CapClipScores:
            continue
        CS = CapClipScores[gkey]

        VARS = []
        POTS = []
        POTSMod = []
        uniquevars = {}
        allgroundings = groundingmap[gkey]
        groundings = []

        for k in allgroundings:
            if k not in testrel2f:
                continue
            else:
                if gkey not in testrel2f[k]:
                    continue
            groundings.append(k)
        missingclipscores = defaultdict(list)
        if len(groundings) == 0:
            if gkey in missingsamples:
                L1 = int(len(missingsamples[gkey]) / 2)
                for i in range(0, L1 - 1, 2):
                    ky = missingsamples[gkey][i] + ":" + missingsamples[gkey][i + 1]
                    missingclipscores[ky].append(missingsamples[gkey][L1 + i])
                    missingclipscores[ky].append(missingsamples[gkey][L1 + i + 1])

        if len(groundings) > 50:
            groundings = random.sample(allgroundings, 50)
        ncr = 0

        for k in groundings:
            parts = k.split(":")
            V1 = parts[0]
            V2 = parts[1]
            if V1 == V2:
                continue
            uniquevars[V1] = 1
            uniquevars[V2] = 1
            if (V1, V2) in VARS:
                ix = VARS.index((V1, V2))
                traindeltas = []
                for tx in TrainedFormulas[k]:
                    traindeltas.append(tx)
                TD = np.array(traindeltas)
                POTS[ix] = (POTS[ix] + TD) / 2.0
            elif (V2, V1) in VARS:
                ix = VARS.index((V2, V1))
                traindeltas = []
                for tx in TrainedFormulas[k]:
                    traindeltas.append(tx)
                TD = np.array(traindeltas)
                POTS[ix] = (POTS[ix] + TD) / 2.0
            else:
                VARS.append((V1, V2))
                traindeltas = []
                for tx in TrainedFormulas[k]:
                    traindeltas.append(tx)
                TD = np.array(traindeltas)
                POTS.append(TD)
                TD1 = np.array(traindeltas)
                if k in testrel2f:
                    if gkey in testrel2f[k]:
                        if testrel2f[k][gkey][0] != 0:
                            n1 = neglog(testrel2f[k][gkey][0])
                            TD1[2] = (TD1[2] + n1) / 2.0
                        if testrel2f[k][gkey][1] != 0:
                            n2 = neglog(testrel2f[k][gkey][1])
                            TD1[1] = (TD1[1] + n2) / 2.0
                elif k in missingclipscores:
                    # print(k+" "+str(missingclipscores[k]))
                    n1 = neglog(missingclipscores[k][0])
                    n2 = neglog(missingclipscores[k][1])
                    TD1[2] = (TD1[2] + n1) / 2.0
                    TD1[1] = (TD1[1] + n2) / 2.0
                POTSMod.append(TD1)

        cnt = cnt + 1
        margs, margsform = gibbssampler(uniquevars, VARS, POTS)
        margs, margsformmod = gibbssampler(uniquevars, VARS, POTSMod)
        result = []
        for m in margsform:
            fp = m.split(",")
            v1 = margsform[m][1]
            if margsform[m][2] > v1:
                v1 = margsform[m][2]
            if m in margsformmod:
                v2 = margsformmod[m][1]
                if margsformmod[m][2] > v2:
                    v2 = margsformmod[m][2]
            result.append(formulaimgmap[fp[0] + ":" + fp[1]][0] + ":" + str(v1 - v2))
        RESULTS[gkey] = result

    with open("resultsChecking-{}.json".format(modelname), "w") as fp:
        json.dump(RESULTS, fp)


test()
