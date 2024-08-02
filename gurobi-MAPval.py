from gurobipy import *
from scipy.stats import norm
import json
import pandas as pd
import spacy

# HYPERPARAMETERS
# (A1,B1)
ABTup = [
    (0.01, 0.1),
    (0.1, 0.01),
    (0.1, 0.05),
    (0.1, 0.1),
    (0.1, 0.25),
    (0.1, 0.4),
    (0.25, 0.05),
    (0.3, 0.05),
    (0.35, 0.05),
    (0.25, 0.50),
]

# from pycocotools.coco import COCO
nlp = spacy.load("en_core_web_lg")

# modelnames = ["aoa", "sgae", "m2", "xlan", blip]
modelname = "m2"

csvfolder_path = "./data/scores_{}.csv".format(modelname)

Clip = {}
names = [
    "Bleu_1",
    "Bleu_2",
    "Bleu_3",
    "Bleu_4",
    "METEOR",
    "ROUGE",
    "CIDEr",
    "SPICE",
    "CLIPScore",
]
Bleu_1 = {}
Bleu_2 = {}
Bleu_3 = {}
Bleu_4 = {}
Meteor = {}
Rogue = {}
Cider = {}
Spice = {}
CLIPScore = {}


data = pd.read_csv(csvfolder_path, sep=",")

for index, row in data.iterrows():
    tmp = int(row["imageId"])
    Bleu_1[str(tmp)] = row["Bleu_1"]
    Bleu_2[str(tmp)] = row["Bleu_2"]
    Bleu_3[str(tmp)] = row["Bleu_3"]
    Bleu_4[str(tmp)] = row["Bleu_4"]
    Meteor[str(tmp)] = row["METEOR"]
    Rogue[str(tmp)] = row["ROUGE"]
    Cider[str(tmp)] = row["CIDEr"]
    Spice[str(tmp)] = row["SPICE"]
    Clip[str(tmp)] = row["CLIPScore"]

with open("allTripClipScore/allTripCscore_" + modelname + ".json", "r") as file:
    blipcontents = json.load(file)

with open("./../clipscore/alltripletScores/tripletGTtest_csore.json", "r") as file1:
    gtcontents = json.load(file1)

for A1, B1 in ABTup:
    print("running for {},{}".format(A1, B1))
    cnt = 0
    cnt1 = 0
    allmapvals = {}
    mapscore = []
    otherscore = []
    for k in gtcontents.keys():
        if k not in blipcontents:
            continue
        if str(k) not in Clip:
            continue
        # model = Model(env=env)
        model = Model("ObjMatch")
        model.Params.LogToConsole = 0
        L = int(len(gtcontents[k]) / 2)
        L1 = int(len(blipcontents[k]) / 2)

        modelvars = {}
        g_univars = {}
        m_univars = {}

        for p in range(0, L1, 1):
            v1 = float(blipcontents[k][L1 + p])
            m_univars[str(p)] = model.addVar(
                vtype=GRB.BINARY, obj=A1 * (v1 * v1), name="M:" + str(p)
            )
        for p1 in range(0, L, 1):
            v2 = float(gtcontents[k][L + p1])
            g_univars[str(p1)] = model.addVar(
                vtype=GRB.BINARY, obj=A1 * (v2 * v2), name="G:" + str(p1)
            )
        for p in range(0, L1, 1):
            v1 = float(blipcontents[k][L1 + p])
            parts = blipcontents[k][p].split()
            for p1 in range(0, L, 1):
                parts1 = gtcontents[k][p1].split()
                P1 = parts[0]
                P2 = parts[len(parts) - 1]
                G1 = parts1[0]
                G2 = parts1[len(parts1) - 1]
                tokens = nlp(P1 + " " + G1)
                SM = tokens[0].similarity(tokens[1])
                tokens = nlp(P2 + " " + G2)
                if tokens[0].similarity(tokens[1]) < SM:
                    SM = tokens[0].similarity(tokens[1])
                DM = 1 - SM
                v2 = float(gtcontents[k][L + p1])
                modelvars[str(p) + ":" + str(p1)] = model.addVar(
                    vtype=GRB.BINARY,
                    obj=-1 * (v1 - v2) * (v1 - v2) - B1 * DM * DM,
                    name="P:" + str(p) + ":" + str(p1),
                )

        for p in range(0, L1, 1):
            for p1 in range(0, L, 1):
                model.addConstr(
                    modelvars[str(p) + ":" + str(p1)]
                    + (1 - m_univars[str(p)])
                    + (1 - g_univars[str(p1)])
                    >= 1
                )
                model.addConstr(
                    (1 - modelvars[str(p) + ":" + str(p1)]) + m_univars[str(p)] >= 1
                )
                model.addConstr(
                    (1 - modelvars[str(p) + ":" + str(p1)]) + g_univars[str(p1)] >= 1
                )

        for p in range(0, L1, 1):
            model.addConstr(
                quicksum(modelvars[str(p) + ":" + str(c)] for c in range(0, L, 1)) >= 1
            )

        model.modelSense = GRB.MAXIMIZE
        model.update()
        model.optimize()

        if model.Status != GRB.OPTIMAL:
            print("Not solvable " + str(k))
            continue

        mapval = model.objVal
        allmapvals[k] = mapval

        cnt = cnt + 1

    with open(
        "./output/MAPSCORES-" + str(A1) + "-" + str(B1) + "-" + modelname + ".json", "w"
    ) as fp:
        json.dump(allmapvals, fp)
