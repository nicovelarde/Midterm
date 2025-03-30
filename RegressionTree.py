
import numpy

# Each node in the tree is represented by this class
class NodeTree:
    """
    @:parameter
    feature_index : which feature it splits on
    threshold : split threshold
    left : left child
    right : right child
    value : for leaf nodes only. Average of y values
    """
    def __init__(self, feature_index=None , threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left=left
        self.right=right
        self.value=value


class RegressionTree:
    def __init__(self,X, y, max_depth=None, min_samples_leaf=None, control_by=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.control_by = control_by #this one determines the stopping condition
        self.root= self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y,depth):
        if self.control_by == 'depth' and self.max_depth==depth:
            return NodeTree(value=numpy.mean(y))

        if self.control_by=='leaf' and self.min_samples_leaf is not None and  len(y) <= self.min_samples_leaf:
            return NodeTree(value=numpy.mean(y))

        #Variables to track the best split (best_SSE, best_feature, best_threshold)
        best_sse = float('inf') #Infinity because we want to minimize the sse
        best_feature= None
        best_threshold = None

        for feature_index in range(X.shape[1]):
            # Get unique values of this feature as potential thresholds
            thresholds = numpy.unique(X[:,feature_index])

            for threshold in thresholds:
                #Split data into left and right based on threshold
                left_mask = X[:,feature_index] < threshold
                right_mask = X[:, feature_index] >= threshold
                #If either side is empty then skip the threshold
                if numpy.sum(left_mask) == 0 or numpy.sum(right_mask)==0:
                    continue
                #contains all target values y for the samples that went to the left side
                y_left = y[left_mask]
                # contains all target values y for the samples that went to the right side
                y_right = y[right_mask]

                #Calculate SSE for left and right
                #You need y_left and y_right because the split is based on X but the error is calculated on Y
                sse_left = numpy.sum((y_left - numpy.mean(y_left))**2)
                sse_right = numpy.sum((y_right - numpy.mean(y_right))**2)
                total_sse = sse_left + sse_right

                #If total SSE is lower than best_SSE then update best_SSE and save the split
                if total_sse < best_sse:
                    best_sse = total_sse
                    best_feature = feature_index
                    best_threshold = threshold

        # If no valid split is found then return a leaf
        # Else, recursively split left and right with the best split found
        if best_feature is None:
            return NodeTree(value=numpy.mean(y))
        left_mask = X[:, best_feature] < best_threshold
        right_mask = X[:, best_feature] >= best_threshold

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return NodeTree(feature_index=best_feature,threshold=best_threshold, left=left_subtree, right=right_subtree)


    def _predict_sample(self, x ,node):
        """
        Helper function that walks one sample down the entire tree
        """
        if node.value is not None:
            return node.value
        else:
            if x[node.feature_index] < node.threshold:
                return self._predict_sample(x,node.left)
            else:
                return self._predict_sample(x,node.right)

    def predict(self, X):
        """
        Returns predictions for each sample in X by walking down the tree.
        """
        return [self._predict_sample(sample, self.root) for sample in X]


    def decision_path(self, x):
        """
        Returns a list of rules (splits) followed to reach the prediction.
        Helpful to understand how the tree makes decisions.
        """
        path = []
        self._decision_path(x, self.root, path)
        return path

    def _decision_path(self, x, node, path):
        if node.value is not None:
            path.append(f"Predict: {node.value}")
            return
        if x[node.feature_index] < node.threshold:
            path.append(f"X[{node.feature_index}] < {node.threshold}")
            self._decision_path(x, node.left, path)
        else:
            path.append(f"X[{node.feature_index}] >= {node.threshold}")
            self._decision_path(x, node.right, path)

    #Implemented for task 2
    def height(self):
        """
        Returns the depth of the tree
        """
        return self._calculate_height(self.root)

    def _calculate_height(self, node):
        if node is None or node.value is not None:
            return 0
        return 1 + max(self._calculate_height(node.left), self._calculate_height(node.right))
