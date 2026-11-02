import numpy as np


def q275(centers, scales, ratios):
    """Generate xyxy anchors at each supplied (x,y) center for every scale and width/height
    ratio. A scale denotes the square root of anchor area.
    """
    boxes = []
    for x, y in centers:
        for scale in scales:
            for ratio in ratios:
                w = scale * np.sqrt(ratio)
                h = scale / np.sqrt(ratio)
                boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
    return np.asarray(boxes, float).reshape(-1, 4)
