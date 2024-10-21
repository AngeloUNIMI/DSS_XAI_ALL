import numpy as np
from util import pause

def getClassCount(trainLoader):

    classCounts = np.zeros(4)

    for _, target in trainLoader:

        binC = np.bincount(target)

        if len(binC) < 4:  # no element for class 3
            binC = np.append(binC, np.zeros(4-len(binC)))

        classCounts += binC


    return classCounts