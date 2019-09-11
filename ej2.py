import numpy as np
from bisect import bisect as binsearch

def train_data_for(data, gender, ranges, xbin, ybin):
    hist, xedges, yedges = np.histogram2d(
        x = data["Peso"],
        y = data["Altura"],
        bins = (xbin, ybin),
        range = ranges)
    # need these to use the model later
    return {
        "gender": gender,
        "hist": hist,
        "xedg": xedges,
        "yedg": yedges
    }

def eval_model_for(train_sets, row, xbin, ybin):
    weight = row["Peso"]
    height = row["Altura"]
    result = train_sets[0]
    hits = -1
    for s in train_sets:
        # edges go over the size of the histogram, clamp accordingly
        wi = max(0, min(xbin - 1, binsearch(s["xedg"], weight) - 1))
        hi = max(0, min(ybin - 1, binsearch(s["yedg"], height) - 1))
        # fetch sample count that fit in the cell
        count = s["hist"][wi, hi]
        hits = max(hits, count)
        if (count == hits):
            # model has the max hits for this case
            result = s
    return result["gender"]

def segregate(data):
    women = data["Genero"] == "Mujer"
    return {"f":data[women], "m":data[~women]}