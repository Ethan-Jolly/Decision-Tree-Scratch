from collections import Counter
import random


# Class which represents the individual nodes of the decision tree. This class contains information for the label if
# it is a leaf node, the midpoint and which feature to compare if it is a decision node, and pointers for whatever
# comes left and right after the node.
class Node:
    def __init__(self, label=None, left=None, right=None, midpoint=None, feature=None):
        self.label = label
        self.midpoint = midpoint
        self.feature = feature
        self.left = left
        self.right = right


# Class which creates the decision tree. This class is initialised with a data array and a label array, and supports
# the parameters max depth, min samples split and min samples leaf, which have the defaults none, 2 and 1 respectfully
class DecisionTree:
    def __init__(self, data, labels, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.data = data  # The input data
        self.labels = labels  # The input labels
        self.Tree = None  # The tree of nodes which will be created
        self.maxDepth = max_depth  # The maximum depth at which the tree will be allowed to reach
        self.minSamplesSplit = min_samples_split  # The minimum number of samples for a split on the samples to be
        # allowed
        self.minSamplesleaf = min_samples_leaf  # The minimum number of samples for a that need to be left after a
        # split for a leaf node

    # Method which takes an array of new points, goes through each one at a time, and using the Tree which has been
    # trained to traverse down the tree using the decision nodes, until a leaf node is reached and the label at that
    # leaf node is assigned to the new point.
    def predict(self, newpoints):
        preds = []
        # For loop going through the newpoints and finding predictions for them
        for n in newpoints:
            pred = self.rec2(self.Tree, n)
            preds.append(pred)
        return preds

    # Method which recursively goes down the tree by comparing the new point to the midpoint until a leaf node is found
    def rec2(self, tree, newpoint):
        # print(tree.midpoint)
        if tree.midpoint is None:
            return tree.label
        else:
            # print(newpoint[tree.feature])
            if newpoint[tree.feature] <= tree.midpoint:
                # print('hi')
                pred = self.rec2(tree.left, newpoint)
            else:
                pred = self.rec2(tree.right, newpoint)
        return pred

    # Method which creates the decision tree of the data, by recursive splitting the data in the left and right
    # according to a purity measure, until a max depth is reached or all points are in a leaf node (depending on the
    # parameters set).
    def train(self):
        split = self.findBestSplit(self.data, self.labels)
        Tree = self.rec(split, 0)
        self.Tree = Tree

    # Method which is used as the recursive element to continuously split the data in to left and right, and assign
    # the nodes to the tree.
    def rec(self, split, depth):
        # Creating a new node for the start of each recursion.
        node = Node()

        # Checking if the split failed, so we create a leaf node
        if type(split) is int:
            # print(split)
            node.label = split
            return node
        # Checking if there is a max depth, and if we have reached it, then create a leaf on the current branch.
        if self.maxDepth is not None:
            if depth >= self.maxDepth:
                # print(split[5])
                node.label = split[5]
                # print('hi')
                return node
        # Creating the separate arrays for left and right
        # print(split)
        left = [i[0] for i in split[2]]
        leftlabels = [i[1] for i in split[2]]
        right = [i[0] for i in split[3]]
        rightlabels = [i[1] for i in split[3]]

        node.midpoint = split[1]
        node.feature = split[4]

        # Recursively going left down from the current node, increasing the depth value
        # print("left")
        split = self.findBestSplit(left, leftlabels)
        newnode = self.rec(split, depth + 1)
        node.left = newnode

        # Recursively going right down from the current node, increasing the depth value
        # print("right")
        split = self.findBestSplit(right, rightlabels)
        newnode = self.rec(split, depth + 1)
        node.right = newnode
        return node

    # Method which finds the split from data and labels, and returns the split information, or the most common label
    # from the data for a leaf node if split failed.
    def findBestSplit(self, data, labels):
        split = self.CalGiniOfSplits(data, labels)
        if not split:
            tuples = list(zip(data, labels))
            sorteddata = sorted(tuples, key=lambda x: x[0])
            return self.maxLabel(sorteddata)
        return split

    # Method which takes the data and labels, and calculates the gini purity for all the possible splits, finding and
    # returning the split which optimizes the gini. False is returned if a split is not found, the number of points
    # in the subset is less than the minimum points for a split, or if the number of different labels is less than 2.
    def CalGiniOfSplits(self, subset, labels):
        if len(set(labels)) < 2 or len(subset) < self.minSamplesSplit:
            return False
        tuples = list(zip(subset, labels))
        sorteddata = sorted(tuples, key=lambda x: x[0])

        minGini = 1  # Setting values to find min gini
        minIndex = 0
        Index = 0
        splits = []
        rows = len(sorteddata)
        cols = len(sorteddata[0][0])
        a = list(range(cols))
        # Randomly arranging array of cols so that if minimum ginis from different features are the same it is random
        # which is chosen.
        random.shuffle(a)
        # print(sorteddata)
        # print(cols)
        for col in a:
            # Sorting the data by the current column of interest
            sorteddata = sorted(tuples, key=lambda x: x[0][col])
            for i in range(rows):
                if i < rows - 1:
                    # Calculating the midpoint between two features
                    midpoint = (sorteddata[i][0][col] + sorteddata[i + 1][0][col]) / 2
                    # Separating the features in to left and right arrays
                    left = []
                    right = []
                    # Creating left and right arrays
                    for j in sorteddata:
                        # Checking if points are above or below the midpoint
                        if j[0][col] < midpoint:
                            left.append(j)
                        else:
                            right.append(j)
                    # Checking if the split has been found or not, if not don't calculate a gini
                    if len(left) != rows and len(right) != rows and len(left) >= self.minSamplesleaf and len(
                            right) >= self.minSamplesleaf:
                        # Calculating the ginis of left and right separately using separate probability calculation
                        # method
                        leftgini = 1 - self._SquareProbabilitiesOfLabels(left)
                        rightgini = 1 - self._SquareProbabilitiesOfLabels(right)
                        # Finding the complete gini at this split
                        gini = ((len(left) / rows) * leftgini) + ((len(right) / rows) * rightgini)
                        splits.append((round(gini, 5), midpoint, list(map(list, left)), list(map(list, right)), col,
                                       self.maxLabel(sorteddata)))
                        # Checking if current gini is the smallest gini that has been found
                        if gini < minGini:
                            # Updating values
                            minIndex = Index
                            minGini = gini
                        Index += 1
        # if no splits were found then return that the split failed.
        if len(splits) == 0:
            return False
        # Returning the gini, midpoint, left and right at minimum gini
        return splits[minIndex]

    # Method which takes a sorted array of tuples of the data and labels, then finds the most common label for the
    # data, and returns it.
    def maxLabel(self, data):
        labels = [data[i][1] for i in range(len(data))]
        totals = Counter(labels)
        totals = {k: totals[k] for k in sorted(totals)}
        return max(totals, key=totals.get)

    # Method which  takes a sorted array of tuples of the data and labels, finds the total number of each label in
    # the data, and uses this to find the square probabilities of the labels in the data.
    def _SquareProbabilitiesOfLabels(self, data):
        labels = [data[i][1] for i in range(len(data))]
        totals = Counter(labels).values()

        sp = 0
        for i in totals:
            sp += (i / len(data)) ** 2
        return sp
