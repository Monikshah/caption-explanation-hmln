import nltk
import json
import os
from collections import defaultdict
import utilities as ut


# This function generates grounding map, caption map, and image 2 triplets
def generateMaps():

    formulaimgmap = ut.readJson("formulaimgmap.json")

    modelname = "xlan"
    dirname = "output_test_" + modelname + "/"
    files = os.listdir(dirname)

    doneprocessing = {}
    groundingmap = defaultdict(list)
    captionmap = {}
    img2triples = defaultdict(list)
    testrel2formclips = defaultdict(list)
    separator = "Caption"

    for filename in files:

        if filename.find(".ipynb_checkpoints") >= 0:
            continue

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

                p1 = ln1.find("extracted triplets: [")
                p2 = ln1.find("]")
                parts = ln1[p1:p2].split("[")

                if len(parts) < 2:
                    continue
                extr = parts[1].split(",")

                for i in range(0, len(extr), 1):
                    extr[i] = extr[i].strip()
                    extr[i] = extr[i][1 : len(extr[i]) - 1]

                if imageid in img2triples:
                    continue

                for e1 in extr:
                    img2triples[imageid].append(e1)

                pairs, newobjs = [], []

                if len(extr) == 1:
                    tokens = nltk.word_tokenize(cparts[1])
                    tagged = nltk.pos_tag(tokens)
                    sparts = extr[0].split()
                    for tg in tagged:
                        if tg[1] == "NN":
                            if tg[0] != sparts[0] and tg[0] != sparts[len(sparts) - 1]:
                                newobjs.append(tg[0])

                    P1 = extr[0].split()
                    pairs.append(P1[0])
                    pairs.append(P1[len(P1) - 1])
                elif len(extr) == 2:
                    P1 = extr[0].split()
                    P2 = extr[1].split()
                    pairs.append(P1[0])
                    pairs.append(P1[len(P1) - 1])
                    pairs.append(P2[0])
                    pairs.append(P2[len(P2) - 1])
                else:
                    modtriples = []
                    for e in extr:
                        parts = e.split()
                        captionwords = cparts[1].split()
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
                        tokens = nltk.word_tokenize(cparts[1])
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

                cnt = 0
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
                    if len(set(tmp)) < 3:
                        continue

                    m1, m2 = False, False

                    for i in range(0, len(pairs) - 1, 2):
                        if P1[0] == pairs[i] and P1[len(P1) - 1] == pairs[i + 1]:
                            m1 = True
                            break
                    for i in range(0, len(pairs) - 1, 2):
                        if P2[0] == pairs[i] and P2[len(P2) - 1] == pairs[i + 1]:
                            m2 = True
                            break

                    if m1 and m2:
                        cnt = cnt + 1
                        listofforms.append(fr)
                    else:
                        if m1 and ((P2[0] in newobjs) and (P2[len(P2) - 1] in newobjs)):
                            cnt = cnt + 1
                            listofforms.append(fr)
                        if m2 and ((P1[0] in newobjs) and (P1[len(P1) - 1] in newobjs)):
                            cnt = cnt + 1
                            listofforms.append(fr)

                groundingmap[str(imageid)] = listofforms
                doneprocessing[str(imageid)] = 1
                captionmap[str(imageid)] = cparts[1]
        ifile.close()

    ut.writeJson("groundingmap-{}.json".format(modelname), groundingmap)
    ut.writeJson("captionmap-{}.json".format(modelname), captionmap)
    ut.writeJson("img2trip-{}.json".format(modelname), img2triples)
    # ut.writeJson("testrel2formsclip-{}.json".format(modelname), testrel2formclips)
