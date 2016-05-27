
import os.path
from linear_svm import *
from dataClassifier import *
class LinearClassifier:
    def __init__(self,datatype):
        self.W = None
        self.Train = True
        self.dataType=datatype
        input = raw_input("choose 1. train    or      2. load weight\n")
        if (input.isdigit()):
            if int(input) == 2:
                if(self.dataType=="digits"):
                    if os.path.isfile("weight_digits.dat"):
                        self.Train = False
                    else:
                        print "weight file do not exist, begin training....."
                elif(self.dataType=="faces"):
                    if os.path.isfile("weight_faces.dat"):
                        self.Train = False
                    else:
                        print "weight file do not exist, begin training....."
        else:
            print "invalid input"
            exit()

    def train(self, X, y, valid_x, valid_y, learning_rate=5e-2, reg=5e-4, num_iters=50000,  # 10000
              batch_size=2, verbose=False):

        """
    Train this linear classifier using stochastic gradient descent.
    Inputs:
    - X: D x N array of training data. Each training point is a D-dimensional
         column.
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.
    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
        if (self.Train):
            if self.dataType=="digits":
                num_iters=50000
            #print "training....."
            print learning_rate, reg, num_iters, batch_size
            y = np.array(y)
            dim, num_train = X.shape
            num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is number of classes
            if self.W is None:
                # lazily initialize W
                self.W = np.random.randn(num_classes, dim) * 0.001
            bestLoss = float("inf")
            bestError = float("inf")
            # Run stochastic gradient descent to optimize W
            for it in xrange(num_iters):
                X_batch = None
                y_batch = None
                # generate random indices
                rand_idx = np.random.choice(num_train, batch_size).tolist()
                X_batch = X[:, rand_idx]
                y_batch = y[rand_idx]
                # evaluate loss and gradient
                loss, grad = self.loss(X_batch, y_batch, reg)
                weight = self.W
                self.W += -1 * learning_rate * grad
                predictLable = self.validation_predict(valid_x, self.W)
                curError = self.error(predictLable, valid_y)
                # Update the weights using the gradient and the learning rate.
                if (curError <= bestError):
                    bestError = curError
                    bestLoss = loss
                    #print it
                    #print "error", bestError
                    #print "loss", loss
                else:
                    self.W = weight
                if verbose and it % 100 == 0:
                    print 'iteration %d / %d: loss %f' % (it, num_iters, loss)
            if self.dataType=="digits":
                print "saving weight file..."
                np.array(self.W).dump("weight_digits.dat")
            elif self.dataType=="faces":
                print "saving weight file..."
                np.array(self.W).dump("weight_faces.dat")
        else:
            self.reloadWeight()
    def validation_predict(self, X, weight):
        """
      Use the trained weights of this linear classifier to predict labels for
      data points.
      Inputs:
      - X: D x N array of training data. Each column is a D-dimensional point.
      Returns:
      - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is an integer giving the predicted
        class.
      """
        scores = np.dot(weight, X)
        y_pred = scores.argmax(axis=0)
        return y_pred

    def classify(self, X):
        """
    Use the trained weights of this linear classifier to predict labels for
    data points.
    Inputs:
    - X: D x N array of training data. Each column is a D-dimensional point.
    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
        scores = np.dot(self.W, X)
        y_pred = scores.argmax(axis=0)
        return y_pred

    def reloadWeight(self):
        print "reload weight....."
        if self.dataType=="digits":
            self.W = np.load("weight_digits.dat")
        elif self.dataType=="faces":
            self.W=np.load("weight_faces.dat")

    def error(self, predictLableSet, originLableSet):
        errorNum = 0
        for i in range(predictLableSet.shape[0]):
            if (predictLableSet[i] != originLableSet[i]):
                errorNum += 1
        errorPercentage = float(errorNum) / predictLableSet.shape[0]
        return errorPercentage

    def loss(self, X_batch, y_batch, reg):
        dW = np.zeros(self.W.shape)  # initialize the gradient as zero
        # compute the loss and the gradient
        num_classes = self.W.shape[0]
        num_train = X_batch.shape[1]
        loss = 0.0
        for i in xrange(num_train):
            scores = self.W.dot(X_batch[:, i])
            correct_class_score = scores[y_batch[i]]
            for j in xrange(num_classes):
                if j == y_batch[i]:  # If correct class
                    continue
                margin = (scores[j] - correct_class_score + 1)  # note delta = 1
                if margin > 0:
                    loss += margin
                    dW[j, :] += X_batch[:, i]
                    dW[y_batch[i], :] -= X_batch[:, i]

        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.
        loss /= num_train
        # Average gradients as well
        dW /= num_train
        # Add regularization to the loss.
        loss += 0.5 * reg * np.sum(self.W * self.W)
        # Add regularization to the gradient
        return loss, dW


def errorRate(predictLableSet, originLableSet):
    errorNum = 0
    for i in range(predictLableSet.shape[0]):
        if (predictLableSet[i] != originLableSet[i]):
            errorNum += 1
    errorPercentage = float(errorNum) / predictLableSet.shape[0]
    print "error percentage: ", errorPercentage * 100
    return errorPercentage


# if __name__ == "__main__":
#     TrainING = True
#     #traindataSet,trainlableSet,validationDataSet,validationLableSet,testdataSet,testlableSet=HogFeatureSetFace()
#     traindataSet = np.array(traindataSet.transpose())
#     validationDataSet = np.array(validationDataSet.transpose())
#     validationLableSet = np.array(validationLableSet)
#     print traindataSet.shape
#     trainlableSet = np.array(trainlableSet)
#     testdataSet = np.array(testdataSet.transpose())
#     testlableSet = np.array(testlableSet)
#     for i in range(1):
#         svm = LinearClassifier()
#         if (TrainING):
#             execTime = i
#             svm.train(traindataSet, trainlableSet, validationDataSet, validationLableSet)
#         else:
#             svm.reloadWeight()
#         predictLableSet = svm.classify(testdataSet)
#         print errorRate(predictLableSet, testlableSet)
