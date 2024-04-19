import nltk
import json
import os
from collections import defaultdict
import utilities as ut
from collections import defaultdict
import numpy as np
import pandas as pd
import random

# ACM-MM-CREATEMAPOFCAPTIONS
# CREATES A MAP OF CAPTIONS FOR A TEST IMAGE


def getCapClipScoreTest():
    csvfolder_path = "./csv_imageLabelCscore_test_" + modelname
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

    return CapClipScores


modelname = "xlan"
CapClipScores = getCapClipScoreTest()


def getTestCaptionMap():
    dirname = "output_test_{}/".format(modelname)
    files = os.listdir(dirname)
    doneprocessing = {}
    groundingmap = defaultdict(list)
    captionmap = {}

    for filename in files:
        if filename.find(".ipynb_checkpoints") >= 0:
            continue
        separator = "Caption"
        if not os.path.isfile(dirname + filename):
            continue
        ifile = open(dirname + filename)
        for ln in ifile:
            if ln.find(separator) == 0:
                cparts = ln.strip().split(":")
                ln1 = ifile.readline()
                p1 = ln1.find("Imageid: ")
                p2 = ln1.find(",")
                parts = ln1[p1:p2].split(":")
                if len(parts) < 2:
                    continue
                imageid = parts[1].strip()
                if str(imageid) in doneprocessing:
                    continue
                captionmap[imageid] = cparts[1]
                doneprocessing[str(imageid)] = 1
    with open("captionmap-{}.json".format(modelname), "w") as fp:
        json.dump(captionmap, fp)


# ACM-MM-CREATEGROUNDINGMAP
# CREATES A MAP OF FORMULAS TO BE GROUNDED FOR A TEST IMAGE


# extripclipfile = "allTripClipScore/extractedTripAoa.json"
# with open(extripclipfile, 'r') as file:
#    extripclipscores = json.load(file)


def groundingForTestImg():
    modelname = "xlan"
    formulaimgmap = ut.readJson("formulaimgmap.json")

    img2tripfile = "img2trip-{}.json".format(modelname)
    with open(img2tripfile, "r") as file:
        img2triples = json.load(file)

    extripclipfile = "allTripClipScore/extractedTripXlan-new.json"
    with open(extripclipfile, "r") as file:
        extripclipscores = json.load(file)

    captionsfile = "captionmap-{}.json".format(modelname)
    with open(captionsfile, "r") as file:
        captionmap = json.load(file)

    # dirname = "output_test_"+modelname+"/"
    # files = os.listdir(dirname)

    groundingmap = defaultdict(list)
    testrel2formclips = defaultdict(dict)
    cnt = 0
    ctmp = 0
    for imgid in img2triples.keys():
        ctmp = ctmp + 1
        #    if ctmp>5:
        #        break
        imageid = str(imgid)
        clipscorestripls = {}
        extr = []
        for i in range(0, len(img2triples[imageid]), 1):
            extr.append(img2triples[imageid][i])
        L1 = int(len(extripclipscores[imageid]) / 2)
        for i in range(0, len(extr), 1):
            # print(imageid)
            # print(extripclipscores[imageid])
            # print('*****', extr[i])
            ix = extripclipscores[imageid].index(extr[i])
            clipscorestripls[extr[i]] = extripclipscores[imageid][L1 + ix]
        # if len(extr)!=L1:
        #    print(imageid+" "+str(extr)+" "+str(extripclipscores[imageid][:L1]))
        # continue
        pairs = []
        pairclipscores = {}
        newobjs = []
        if len(extr) == 1:
            tokens = nltk.word_tokenize(captionmap[imageid])
            tagged = nltk.pos_tag(tokens)
            sparts = extr[0].split()
            for tg in tagged:
                if tg[1] == "NN":
                    if tg[0] != sparts[0] and tg[0] != sparts[len(sparts) - 1]:
                        newobjs.append(tg[0])
            P1 = extr[0].split()
            pairs.append(P1[0])
            pairs.append(P1[len(P1) - 1])
            pairclipscores[P1[0] + ":" + P1[len(P1) - 1]] = clipscorestripls[extr[0]]
        elif len(extr) == 2:
            # C2=C2+1
            P1 = extr[0].split()
            P2 = extr[1].split()
            pairs.append(P1[0])
            pairs.append(P1[len(P1) - 1])
            pairs.append(P2[0])
            pairs.append(P2[len(P2) - 1])
            pairclipscores[P1[0] + ":" + P1[len(P1) - 1]] = clipscorestripls[extr[0]]
            pairclipscores[P2[0] + ":" + P2[len(P2) - 1]] = clipscorestripls[extr[1]]
        else:
            modtriples = []
            for e in extr:
                parts = e.split()
                captionwords = captionmap[imageid].split()
                w1 = False
                w2 = False
                for c in captionwords:
                    if parts[0] == c:
                        w1 = True
                    if parts[len(parts) - 1] == c:
                        w2 = True
                if w1 and w2:
                    modtriples.append(e)

            if len(modtriples) == 0:
                # C1=C1+1
                tokens = nltk.word_tokenize(captionmap[imageid])
                tagged = nltk.pos_tag(tokens)
                newobjs = []
                for tg in tagged:
                    if tg[1] == "NN":
                        newobjs.append(tg[0])
            if len(modtriples) >= 2:
                for m in modtriples:
                    P1 = m.split()
                    pairs.append(P1[0])
                    pairs.append(P1[len(P1) - 1])
                    pairclipscores[P1[0] + ":" + P1[len(P1) - 1]] = clipscorestripls[m]

        newobjs = list(set(newobjs))
        listofforms = []
        for fr in formulaimgmap:
            parts = fr.split(":")
            gr = 0
            P1 = parts[0].split()
            P2 = parts[1].split()

            tmp = []
            tmp.append(P1[0])
            tmp.append(P1[len(P1) - 1])
            tmp.append(P2[0])
            tmp.append(P2[len(P2) - 1])
            # if len(set(tmp))<3:
            #    continue

            m1 = False
            m2 = False

            for i in range(0, len(pairs) - 1, 2):
                if P1[0] == pairs[i] and P1[len(P1) - 1] == pairs[i + 1]:
                    m1 = True
                    break
            for i in range(0, len(pairs) - 1, 2):
                if P2[0] == pairs[i] and P2[len(P2) - 1] == pairs[i + 1]:
                    m2 = True
                    break
            if m1 and m2:
                v0 = pairclipscores[P1[0] + ":" + P1[len(P1) - 1]]
                v1 = pairclipscores[P2[0] + ":" + P2[len(P2) - 1]]
                testrel2formclips[fr][imageid] = (v0, v1)
                listofforms.append(fr)
            else:
                if m1 and ((P2[0] in newobjs) or (P2[len(P2) - 1] in newobjs)):
                    v0 = pairclipscores[P1[0] + ":" + P1[len(P1) - 1]]
                    v1 = 0
                    testrel2formclips[fr][imageid] = (v0, v1)
                    listofforms.append(fr)
                if m2 and ((P1[0] in newobjs) or (P1[len(P1) - 1] in newobjs)):
                    v0 = 0
                    v1 = pairclipscores[P2[0] + ":" + P2[len(P2) - 1]]
                    testrel2formclips[fr][imageid] = (v0, v1)
                    listofforms.append(fr)
        if len(listofforms) == 0:
            cnt = cnt + 1
        groundingmap[str(imageid)] = listofforms

    with open("groundingmap-{}.json".format(modelname), "w") as fp:
        json.dump(groundingmap, fp)
    with open("testrel2formsclip-{}.json".format(modelname), "w") as fp:
        json.dump(testrel2formclips, fp)


def groundingMapForTest():

    modelname = "xlan"
    formulaimgmap = ut.readJson("formulaimgmap.json")

    with open("trainedFormulas.json", "r") as fp:
        TrainedFormulas = json.load(fp)

    # ACM-MM-REFINEGROUNDINGMAP
    # REFINES THE MAP OF FORMULAS TO BE GROUNDED FOR A TEST IMAGE TO MINIMIZE UNGROUNDED IMAGES

    groundingmapfile = "groundingmap-{}.json".format(modelname)

    with open(groundingmapfile, "r") as file:
        groundingmap = json.load(file)

    captionmapfile = "captionmap-{}.json".format(modelname)
    with open(captionmapfile, "r") as file:
        captionmap = json.load(file)
    missingkeys = defaultdict(list)
    missing = defaultdict(list)
    ps = nltk.PorterStemmer()
    for itr in range(0, 2, 1):
        for M1 in missing.keys():
            gr = False
            for fr in formulaimgmap:
                cnt = 0
                for m in missing[M1]:
                    if m in fr:
                        cnt = cnt + 1
                if cnt >= 2:
                    missingkeys[M1].append(fr)
        # cnt=0
        ct = 0
        for gkey in groundingmap.keys():
            if gkey not in CapClipScores:
                continue
            CS = CapClipScores[gkey]
            VARS = []
            POTS = []
            uniquevars = {}
            groundings = groundingmap[gkey]

            if gkey in missingkeys:
                groundings = missingkeys[gkey]

            for k in groundings:
                parts = k.split(":")
                objs = parts[0].split()
                V1 = objs[0] + ":" + objs[len(objs) - 1]
                objs1 = parts[1].split()
                V2 = objs1[0] + ":" + objs1[len(objs1) - 1]

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
                else:
                    VARS.append((V1, V2))
                    traindeltas = []
                    for tx in TrainedFormulas[k]:
                        traindeltas.append(tx)
                    TD = np.array(traindeltas)
                    POTS.append(TD)
            cnt = cnt + 1
            if len(VARS) == 0:
                ct = ct + 1
                tokens = nltk.word_tokenize(captionmap[gkey])
                tagged = nltk.pos_tag(tokens)
                tmp = []
                for i in range(0, len(tagged), 1):
                    if tagged[i][1] == "NN":
                        # ofile.write(tagged[i][0]+",")
                        tmp.append(tagged[i][0])
                    if tagged[i][1] == "NNS":
                        s1 = tagged[i][0]
                        tmp.append(ps.stem(s1))
                        if s1 == "people":
                            tmp.append("person")
                        if s1 == "men":
                            tmp.append("man")
                        if s1 == "women":
                            tmp.append("woman")
                        if s1 == "children":
                            tmp.append("child")

                missing[gkey] = tmp
    for gkey in groundingmap.keys():
        if gkey in missingkeys:
            groundingmap[gkey] = missingkeys[gkey]

    with open("groundingmap-updated-{}.json".format(modelname), "w") as fp:
        json.dump(groundingmap, fp)


def addMissingGroundingMap():
    # MISSING_GROUNDINGS_MAP

    missingmap = {}
    groundingmap = ut.readJson("groundingmap-updated-{}.json".format(modelname))

    for gkey in groundingmap.keys():
        if gkey not in CapClipScores:
            continue
        CS = CapClipScores[gkey]
        allgroundings = groundingmap[gkey]
        groundings = []
        for k in allgroundings:
            # print(k)
            if k not in testrel2f:
                continue
            else:
                if gkey not in testrel2f[k]:
                    continue
            groundings.append(k)
        if len(groundings) == 0:
            if len(allgroundings) > 0:
                if len(allgroundings) < 10:
                    missingmap[gkey] = allgroundings
                else:
                    missingmap[gkey] = random.sample(allgroundings, 10)
            # GET THE CLIP SCORES FOR 10 SAMPLED FORMULAS FROM allgroundings for image gkey

    with open("missingclips-{}.json".format(modelname), "w") as fp:
        json.dump(missingmap, fp)
