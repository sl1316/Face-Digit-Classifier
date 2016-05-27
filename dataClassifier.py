import linear_svm
import naiveBayes
import perceptron
import samples
import sys
import util
import timeit
import numpy as np
import random
from scipy.ndimage import uniform_filter
TEST_SET_SIZE = 50
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


def basicFeatureExtractorDigit(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(DIGIT_DATUM_HEIGHT):
    for y in range(DIGIT_DATUM_WIDTH):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0

  return features
def basicFeatureExtractorFace(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  a = datum.getPixels()

  features = util.Counter()

  for x in range(FACE_DATUM_HEIGHT):
    for y in range(FACE_DATUM_WIDTH):
      if datum.getPixel(x, y) == 1:
        features[(x, y)] = 1
      elif datum.getPixel(x, y) == 2:
        features[(x, y)] = 1
      else:
        features[(x, y)] = 0
  return features
def enhancedFeatureExtractorDigit(datum):
    features = basicFeatureExtractorDigit(datum)

    def addFeature(name, value, time):
        features[name] = value
        for i in xrange(time):
            features[str(i) + '--' + name] = features[name]

    TP = DIGIT_DATUM_WIDTH * DIGIT_DATUM_HEIGHT

    pixels = datum.getPixels()
    upper = sum([pixels[row][col] > 0 for row in xrange(DIGIT_DATUM_HEIGHT / 2)
                 for col in xrange(DIGIT_DATUM_WIDTH)])
    lower = sum([pixels[row][col] > 0 for row in xrange(DIGIT_DATUM_HEIGHT / 2, DIGIT_DATUM_HEIGHT)
                 for col in xrange(DIGIT_DATUM_WIDTH)])
    left = sum([pixels[row][col] > 0 for row in xrange(DIGIT_DATUM_WIDTH)
                for col in xrange(DIGIT_DATUM_WIDTH / 2)])
    right = sum([pixels[row][col] > 0 for row in xrange(DIGIT_DATUM_WIDTH)
                 for col in xrange(DIGIT_DATUM_WIDTH / 2, DIGIT_DATUM_WIDTH)])

    addFeature('left', left > right, 2)
    addFeature('upper', upper > lower, 2)

    connectedRegions = getPartitionNumDigit(pixels)
    for i in xrange(1, 5):
        addFeature('regions' + str(i), connectedRegions == i, 6)

    for x in range(DIGIT_DATUM_HEIGHT):
        pixelsInbinary = [bool(datum.getPixel(x, y)) for y in xrange(DIGIT_DATUM_WIDTH)]
        nBPixels = sum(pixelsInbinary)
        addFeature('empty' + str(x), nBPixels > 0, 3)

        leftEdge = ((DIGIT_DATUM_WIDTH - 1) - pixelsInbinary[::-1].index(True)) if nBPixels else 0
        rightEdge = pixelsInbinary.index(True) if nBPixels else 0
        width = leftEdge - rightEdge
        addFeature('hole' + str(x), width + (width > 1) > nBPixels, 2)

        # horizontal symmetrical
        hs = sum([pixels[x][y] > 0 and pixels[x][DIGIT_DATUM_WIDTH - 1 - y] > 0 for y in xrange(DIGIT_DATUM_WIDTH / 2)])
        addFeature('hs' + str(x), hs > 0.3 * TP / 2, 2)
    return features

def getPartitionNumDigit(pixels):

    from collections import Counter

    def pointInImage(x, y):
        return (0 <= x < DIGIT_DATUM_HEIGHT) and (0 <= y < DIGIT_DATUM_WIDTH)

    def neighbours(x, y):
        candidates_neighbours = ((x, y-1), (x, y+1), (x+1, y), (x-1, y), (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1))
        return {(x, y) for x, y in candidates_neighbours if pointInImage(x, y)}

    def unexplored_neighbours(x, y, explored_set):
        return neighbours(x, y) - explored_set

    def pointOnEdge(x, y):
        return pixels[x][y] > 0

    def pointNotOnEdge(x, y):
        return not pixels[x][y] > 0

    def bfs(explored_set, start_point, spaceIndex, partition):
        # use bfs to color the image
        queue = [start_point]
        showEnqueue = pointOnEdge if pointOnEdge(*start_point) else pointNotOnEdge
        while len(queue) > 0:
            x, y = current = queue.pop(0)
            if ((x, y) not in explored_set) and showEnqueue(x, y):
                partition[current] = spaceIndex
                explored_set.add(current)
                queue.extend(unexplored_neighbours(x, y, explored_set))
        return partition

    def partitionIsComplete(partition):
        for x in xrange(DIGIT_DATUM_HEIGHT):
            for y in xrange(DIGIT_DATUM_WIDTH):
                if (x, y) not in partition:
                    return False, (x, y)
        return True, None

    exploredPoints = set()
    partition = {}
    spaceIndex = 0
    isComplete, startPoint = partitionIsComplete(partition)
    while not isComplete:
        partition = bfs(exploredPoints, startPoint, spaceIndex, partition)
        spaceIndex += 1
        isComplete, startPoint = partitionIsComplete(partition)
    c = Counter(partition.values()).items()
    return len(filter(lambda t: t[1] > 3, c))

def getPartitionNumFace(pixels):
    '''
    :param pixels: a 2-dim array of pixels representing an image
    :return: number of regions of  the image
    '''
    from collections import Counter
    def pointInImage(x, y):
        return (0 <= x < FACE_DATUM_HEIGHT) and (0 <= y < FACE_DATUM_WIDTH)

    def neighbours(x, y):
        '''
        :param x: x coordinate of point
        :param y: y coordinate of point
        :return: a set of numbering points of point (x, y)
        '''
        candidates_neighbours = ((x, y-1), (x, y+1), (x+1, y), (x-1, y), (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1))
        return {(x, y) for x, y in candidates_neighbours if pointInImage(x, y)}

    def unexplored_neighbours(x, y, explored_set):
        '''
        :param x: x coordinate of point
        :param y: y coordinate of point
        :param explored_set: a set containing explored points
        :return: a set of non-explored sets
        '''
        return neighbours(x, y) - explored_set

    def pointOnEdge(x, y):
        return pixels[x][y] > 0

    def pointNotOnEdge(x, y):
        return not pixels[x][y] > 0

    def bfs(explored_set, start_point, spaceIndex, partition):
        # use bfs to color the image
        queue = [start_point]
        showEnqueue = pointOnEdge if pointOnEdge(*start_point) else pointNotOnEdge
        while len(queue) > 0:
            x, y = current = queue.pop(0)
            if ((x, y) not in explored_set) and showEnqueue(x, y):
                partition[current] = spaceIndex
                explored_set.add(current)
                queue.extend(unexplored_neighbours(x, y, explored_set))
        return partition

    def partitionIsComplete(partition):
        for x in xrange(FACE_DATUM_HEIGHT):
            for y in xrange(FACE_DATUM_WIDTH):
                if (x, y) not in partition:
                    return False, (x, y)
        return True, None

    exploredPoints = set()
    partition = {}
    spaceIndex = 0
    isComplete, startPoint = partitionIsComplete(partition)
    while not isComplete:
        partition = bfs(exploredPoints, startPoint, spaceIndex, partition)
        spaceIndex += 1
        isComplete, startPoint = partitionIsComplete(partition)
    c = Counter(partition.values()).items()
    return len(filter(lambda t: t[1] > 3, c))

def enhancedFeatureExtractorFace(datum):
    features =  basicFeatureExtractorFace(datum)

    def addFeature(name, value, num):
        features[name] = value
        for i in xrange(num):
            features[str(i) + '--' + name] = features[name]

    pixels = datum.getPixels()

    TP = FACE_DATUM_WIDTH * FACE_DATUM_HEIGHT

    upper = sum([pixels[row][col] > 0 for row in xrange(FACE_DATUM_HEIGHT / 2)
                 for col in xrange(FACE_DATUM_WIDTH)])
    lower = sum([pixels[row][col] > 0 for row in xrange(FACE_DATUM_HEIGHT / 2, FACE_DATUM_HEIGHT)
                 for col in xrange(FACE_DATUM_WIDTH)])
    left = sum([pixels[row][col] > 0 for row in xrange(FACE_DATUM_WIDTH)
                for col in xrange(FACE_DATUM_WIDTH / 2)])
    right = sum([pixels[row][col] > 0 for row in xrange(FACE_DATUM_WIDTH)
                 for col in xrange(FACE_DATUM_WIDTH / 2, FACE_DATUM_WIDTH)])
    addFeature('left', left > right, 2)
    addFeature('upper', upper > lower, 2)

    connectedRegions = getPartitionNumFace(pixels)
    for i in xrange(1, 5):
        addFeature('regions'+str(i), connectedRegions == i, 6)

    for x in range(FACE_DATUM_HEIGHT):
        pixelsInbinary = [bool(datum.getPixel(x, y)) for y in xrange(FACE_DATUM_WIDTH)]
        nBPixels = sum(pixelsInbinary)
        addFeature('empty'+str(x), nBPixels > 0, 3)

        leftEdge = ((FACE_DATUM_WIDTH-1)-pixelsInbinary[::-1].index(True)) if nBPixels else 0
        rightEdge = pixelsInbinary.index(True) if nBPixels else 0
        width = leftEdge - rightEdge
        addFeature('hole'+str(x), width + (width > 1) > nBPixels, 2)

        # horizontal symmetrical
        hs = sum([pixels[x][y] > 0 and pixels[x][FACE_DATUM_WIDTH - 1 -y] > 0 for y in xrange(FACE_DATUM_WIDTH / 2)])
        addFeature('hs' + str(x), hs > 0.3*TP/2, 2)
    return features

def basicExtractFace(datum):
    a = datum.getPixels()

    basicFeatures =np.zeros((FACE_DATUM_HEIGHT,FACE_DATUM_WIDTH))
    for x in range(FACE_DATUM_HEIGHT):
        for y in range(FACE_DATUM_WIDTH):
            if datum.getPixel(x, y) >0:
                basicFeatures[x,y]=1
            else:
                basicFeatures[x, y] = 0
    return basicFeatures
def HogFeatureFaceImg(datum):
    img = basicExtractFace(datum)
    size_x, size_y = img.shape
    cx, cy = (5, 5)
    bx, by = (3, 3)
    orientations = 9;  # number of gradient bins
    gx = np.zeros(img.shape)
    gy = np.zeros(img.shape)
    gx[:, :-1] = np.diff(img, n=1, axis=1)
    gy[:-1, :] = np.diff(img, n=1, axis=0)  # compute gradient on y-direction
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)  # gradient magnitude
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90  # gradient orientation
    n_cellsx = int(np.floor(size_x / cx))  # number of cells in x
    n_cellsy = int(np.floor(size_y / cy))  # number of cells in y
    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    # print n_cellsx,n_cellsy
    # print "histogram"
    # print orientation_histogram
    for i in range(orientations):
        # create new integral image for this orientation
        # isolate orientations in this range
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                            grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i,
                            temp_ori, 0)
        # select magnitudes for those orientations
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, grad_mag, 0)
        # print temp_mag
        orientation_histogram[:, :, i] = uniform_filter(temp_mag, size=(cx, cy))[cx / 2::cx, cy / 2::cy]
    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1
    normalised_blocks = np.zeros((n_blocksx, n_blocksy,
                                  bx, by, orientations))

    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = orientation_histogram[x:x + bx, y:y + by, :]
            eps = 1e-5
            normalised_blocks[x, y, :] = block / np.sqrt(block.sum() ** 2 + eps)

    return normalised_blocks.ravel()
def basicExtractDigit(datum):
    a = datum.getPixels()

    basicFeatures =np.zeros((DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT))
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) ==1:
                basicFeatures[x, y] = 1
            elif datum.getPixel(x, y) ==2:
                basicFeatures[x, y] = 1
            else:
                basicFeatures[x, y] = 0
    return basicFeatures
def HogFeatureImgDigit(datum):
    img=basicExtractDigit(datum)
    #img=np.sqrt(img
    size_x,size_y=img.shape
    cx,cy=(4,4)
    #print cx,cy
    bx, by=(3,3)
    orientations=9;#number of gradient bins
    gx = np.zeros(img.shape)
    gy = np.zeros(img.shape)
    gx[:, :-1] = np.diff(img, n=1, axis=1)
    gy[:-1, :] = np.diff(img, n=1, axis=0)  # compute gradient on y-direction
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)  # gradient magnitude
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90  # gradient orientation
    n_cellsx = int(np.floor(size_x / cx))  # number of cells in x
    n_cellsy = int(np.floor(size_y / cy))  # number of cells in y
    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    #print n_cellsx,n_cellsy
    #print "histogram"
    #print orientation_histogram
    for i in range(orientations):
        # create new integral image for this orientation
        # isolate orientations in this range
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                            grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i,
                            temp_ori, 0)
        # select magnitudes for those orientations
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, grad_mag, 0)
        #print temp_mag
        orientation_histogram[:, :, i] = uniform_filter(temp_mag, size=(cx, cy))[cx / 2::cx, cy / 2::cy].T
    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1
    normalised_blocks = np.zeros((n_blocksx, n_blocksy,
                                  bx, by, orientations))

    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = orientation_histogram[x:x + bx, y:y + by, :]
            eps = 1e-5
            normalised_blocks[x, y, :] = block / np.sqrt(block.sum() ** 2 + eps)

    return normalised_blocks.ravel()

    #rint orientation_histogram.shape
    #print grad_ori.shape
    #print gx[:, :-1]
def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
  """
  This function is called after learning.
  Include any code that you want here to help you analyze your results.
  
  Use the printImage(<list of pixels>) function to visualize features.
  
  An example of use has been given to you.
  
  - classifier is the trained classifier
  - guesses is the list of labels predicted by your classifier on the test set
  - testLabels is the list of true labels
  - testData is the list of training datapoints (as util.Counter of features)
  - rawTestData is the list of training datapoints (as samples.Datum)
  - printImage is a method to visualize the features 
  (see its use in the odds ratio part in runClassifier method)
  
  This code won't be evaluated. It is for your own optional use
  (and you can modify the signature if you want).
  """
  
  # Put any code here...
  # Example of use:
  for i in range(len(guesses)):
      prediction = guesses[i]
      truth = testLabels[i]
      if (prediction != truth):
          print "==================================="
          print "Mistake on example %d" % i 
          print "Predicted %d; truth is %d" % (prediction, truth)
          print "Image: "
          print rawTestData[i]
          break

class ImagePrinter:
    def __init__(self, width, height):
      self.width = width
      self.height = height

    def printImage(self, pixels):
      """
      Prints a Datum object that contains all pixels in the 
      provided list of pixels.  This will serve as a helper function
      to the analysis function you write.
      
      Pixels should take the form 
      [(2,2), (2, 3), ...] 
      where each tuple represents a pixel.
      """
      image = samples.Datum(None,self.width,self.height)
      for pix in pixels:
        try:
            # This is so that new features that you could define which 
            # which are not of the form of (x,y) will not break
            # this image printer...
            x,y = pix
            image.pixels[x][y] = 2
        except:
            print "new features:", pix
            continue
      print image
def default(str):
  return str + ' [Default: %default]'

def readCommand( argv ):
  "Processes the command used to run from the command line."
  from optparse import OptionParser  
  parser = OptionParser(USAGE_STRING)
  
  parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['linear_svm', 'nb', 'naiveBayes', 'perceptron'], default='linear_svm')
  parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], default='faces')
  parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
  parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=0.1)
  parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
  parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
  parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=50, type="int")

  options, otherjunk = parser.parse_args(argv)
  if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
  args = {}
  
  # Set up variables according to the command line input.
  print "Doing classification"
  print "--------------------"
  print "data:\t\t" + options.data
  print "classifier:\t\t" + options.classifier
  print "training set size:\t" + str(options.training)
  print "testing set size:\t"+str(options.test)
  if(options.data=="digits"):
    printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage

  elif(options.data=="faces"):
    printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
  else:
    print "Unknown dataset", options.data
    print USAGE_STRING
    sys.exit(2)
    
  if(options.data=="digits"):
    legalLabels = range(10)
  else:
    legalLabels = range(2)
    
  if options.training <= 0:
    print "Training set size should be a positive integer (you provided: %d)" % options.training
    print USAGE_STRING
    sys.exit(2)
    
  if options.smoothing <= 0:
    print "Please provide a positive number for smoothing (you provided: %f)" % options.smoothing
    print USAGE_STRING
    sys.exit(2)

  if(options.classifier == "linear_svm"):
    classifier = linear_svm.LinearClassifier(options.data)
  elif(options.classifier == "naiveBayes" or options.classifier == "nb"):
    classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
    classifier.setSmoothing(options.smoothing)
    if (options.autotune):
        print "using automatic tuning for naivebayes"
        classifier.automaticTuning = True
    else:
        print "using smoothing parameter k=%f for naivebayes" %  options.smoothing
  elif(options.classifier == "perceptron"):
    classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
  else:
    print "Unknown classifier:", options.classifier
    print USAGE_STRING
    sys.exit(2)

  args['classifier'] = classifier
  args['printImage'] = printImage
  return args, options

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default linear svm classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -k 2.5 -s 100
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the 100 test examples
                 """
def randomSample(data,label,n):
    indices=[i for i in range(len(label))]
    if(n!=len(label)):
        sample_indices=np.array(random.sample(indices,n))
    else:
        sample_indices = [i for i in range(len(label))]
    dataSample=np.array(data)[sample_indices].tolist()
    labelSample=np.array(label)[sample_indices].tolist()
    return dataSample,labelSample
def runClassifier(args, options):
  classifier = args['classifier']
  printImage = args['printImage']
  # Load data  
  numTraining = options.training
  numTest = options.test
  if(options.data=="faces"):
    print "loading face data set"
    rawTrainingData = samples.loadDataFile("facedata/facedatatrain",FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels")
    rawValidationData = samples.loadDataFile("facedata/facedatavalidation",FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("facedata/facedatavalidationlabels")
    rawTestData = samples.loadDataFile("facedata/facedatatest", FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("facedata/facedatatestlabels")
    rawTrainingData,trainingLabels=randomSample(rawTrainingData,trainingLabels,numTraining)
    rawTestData,testLabels=randomSample(rawTestData,testLabels,numTest)
  else:
    print "loading digit data set"
    rawTrainingData = samples.loadDataFile("digitdata/trainingimages",DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels")
    rawValidationData = samples.loadDataFile("digitdata/validationimages",DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("digitdata/validationlabels")
    rawTestData = samples.loadDataFile("digitdata/testimages",DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels")
    rawTrainingData, trainingLabels = randomSample(rawTrainingData, trainingLabels, numTraining)
    rawTestData, testLabels = randomSample(rawTestData, testLabels, numTest)
  print "Extracting features..."
  if (options.classifier == "linear_svm"):
        if (options.data == "faces"):
            featureFunction = HogFeatureFaceImg
        else:
            featureFunction=HogFeatureImgDigit
        trainingData = map(featureFunction, rawTrainingData)
        trainingData=np.array(trainingData).transpose()
        validationData=map(featureFunction, rawValidationData)
        validationData = np.array(validationData).transpose()
        testData=map(featureFunction, rawTestData)
        testData = np.array(testData).transpose()
  else:
      if (options.data == "faces"):
          featureFunction = enhancedFeatureExtractorFace
      else:
          featureFunction = enhancedFeatureExtractorDigit
      trainingData = map(featureFunction, rawTrainingData)
      validationData = map(featureFunction, rawValidationData)
      testData = map(featureFunction, rawTestData)
  print "Training..."
  start = timeit.default_timer()
  classifier.train(trainingData, trainingLabels, validationData, validationLabels)
  stop = timeit.default_timer()
  print  stop - start, " s"
  print "Validating..."
  guesses = classifier.classify(validationData)
  correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
  print str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels))
  print "Testing..."
  guesses = classifier.classify(testData)
  correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
  print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
  analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)
  



if __name__ == '__main__':
  # Read input
  args, options = readCommand( sys.argv[1:] ) 
  # Run classifier
  runClassifier(args, options)
