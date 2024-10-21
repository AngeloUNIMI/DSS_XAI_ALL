import os
import matplotlib.pyplot as plt
from util.pause import pause
import csv


def dbToDataStore(dirIn, dirOut, extOrig, extNew, log):

    # create dir for class 0
    os.makedirs(os.path.join(dirOut, '0'), exist_ok=True)

    # display
    if log:
        print("Transforming DB...")

    # transform db
    for name in os.listdir(dirIn):
        if name.endswith(extOrig):
            # display
            #if log:
                #print("\tProcessing: " + name)
            # read name
            pre, ext = os.path.splitext(name)
            # get label
            C = pre.split('_')

            # increase class number +1
            newClass = str(int(C[1]) + 1)

            # create dir with label
            # dirOutLabel = os.path.join(dirOut, C[1])
            dirOutLabel = os.path.join(dirOut, newClass)
            # newname
            newName = pre + '.' + extNew
            newPath = os.path.join(dirOutLabel, newName)

            # print(newPath)

            # if already present skip
            if os.path.exists(newPath):
                continue
            # create directory if not present
            if not os.path.exists(dirOutLabel):
                os.makedirs(dirOutLabel)
            # read
            img = plt.imread(os.path.join(dirIn, name))

            # display
            """
            print(newName)
            plt.imshow(img)
            plt.show()
            pause()
            """

            # write
            plt.imsave(newPath, img, format=extNew)

            #pause()

    print()


def getClass(columnNames, rowF, classesADP):
    classes = list()
    for className in classesADP['classesNames']:
        classes.append(rowF[columnNames.index(className)])
    #print(classes)
    #classOne = torch.max(classes, 1)
    # cast
    classesInt = [int(i) for i in classes]
    return classesInt


def getAllClassesVec(classesADP, csvFileFull, log):
    # open csv
    allClasses = list()
    allFileNames = list()
    with open(csvFileFull) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                columnNames = row
            else:
                #print(row)
                fileName = row[0]
                classVec = getClass(columnNames, row, classesADP)
                allClasses.append(classVec)
                allFileNames.append(fileName)
                #print(classVec)
                #pause()
            line_count += 1
        print('Processed {0} lines.'.format(line_count))
    return allClasses, allFileNames, columnNames
