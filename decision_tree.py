import csv
import random

## TODO Assignment:
# Build the functions learntree and classify of the DecisionTree class
#

def read_data(csv_path):
    """Read in the training data from a csv file.
    
    The examples are returned as a list of Python dictionaries, with column names as keys.
    """
    examples = []
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for example in csv_reader:
            for k, v in example.items():
                if v == '':
                    example[k] = None
                else:
                    try:
                        example[k] = float(v)
                    except ValueError:
                         example[k] = v
            examples.append(example)
    return examples


def train_test_split(examples, test_perc):
    """Randomly data set (a list of examples) into a training and test set."""
    test_size = round(test_perc*len(examples))    
    shuffled = random.sample(examples, len(examples))
    return shuffled[test_size:], shuffled[:test_size]


class TreeNodeInterface():
    """Simple "interface" to ensure both types of tree nodes must have a classify() method."""
    def classify(self, example): 
        raise NotImplementedError


class DecisionNode(TreeNodeInterface):
    """Class representing an internal node of a decision tree."""

    def __init__(self, test_attr_name, test_attr_threshold, child_lt, child_ge, miss_lt):
        """Constructor for the decision node.  Assumes attribute values are continuous.

        Args:
            test_attr_name: column name of the attribute being used to split data
            test_attr_threshold: value used for splitting
            child_lt: DecisionNode or LeafNode representing examples with test_attr_name
                values that are less than test_attr_threshold
            child_ge: DecisionNode or LeafNode representing examples with test_attr_name
                values that are greater than or equal to test_attr_threshold
            miss_lt: True if nodes with a missing value for the test attribute should be 
                handled by child_lt, False for child_ge                 
        """    
        self.test_attr_name = test_attr_name  
        self.test_attr_threshold = test_attr_threshold 
        self.child_ge = child_ge
        self.child_lt = child_lt
        self.miss_lt = miss_lt

    def classify(self, example):
        """Classify an example based on its test attribute value.
        
        Args:
            example: a dictionary { attr name -> value } representing a data instance

        Returns: a class label and probability as tuple
        """
        test_val = example[self.test_attr_name]
        if test_val is None:
            return self.child_miss.classify(example)
        elif test_val < self.test_attr_threshold:
            return self.child_lt.classify(example)
        else:
            return self.child_ge.classify(example)

    def __str__(self):
        return "test: {} < {:.4f}".format(self.test_attr_name, self.test_attr_threshold) 


class LeafNode(TreeNodeInterface):
    """Class representing a leaf node of a decision tree.  Holds the predicted class."""

    def __init__(self, pred_class, pred_class_count, total_count):
        """Constructor for the leaf node.

        Args:
            pred_class: class label for the majority class that this leaf represents
            pred_class_count: number of training instances represented by this leaf node
            total_count: the total number of training instances used to build the leaf node
        """    
        self.pred_class = pred_class
        self.pred_class_count = pred_class_count
        self.total_count = total_count
        self.prob = pred_class_count / total_count  # probability of having the class label

    def classify(self, example):
        """Classify an example.
        
        Args:
            example: a dictionary { attr name -> value } representing a data instance

        Returns: a class label and probability as tuple as stored in this leaf node.  This will be
            the same for all examples!
        """
        return self.pred_class, self.prob

    def __str__(self):
        return "leaf {} {}/{}={:.2f}".format(self.pred_class, self.pred_class_count, 
                                             self.total_count, self.prob)


class DecisionTree:
    """Class representing a decision tree model."""

    def __init__(self, examples, id_name, class_name, min_leaf_count=1):
        """Constructor for the decision tree model.  Calls learn_tree().

        Args:
            examples: training data to use for tree learning, as a list of dictionaries
            id_name: the name of an identifier attribute (ignored by learn_tree() function)
            class_name: the name of the class label attribute (assumed categorical)
            min_leaf_count: the minimum number of training examples represented at a leaf node
        """
        self.id_name = id_name
        self.class_name = class_name
        self.min_leaf_count = min_leaf_count

        # build the tree!
        self.root = self.learn_tree(examples)  

    def learn_tree(self, examples):
        """Build the decision tree based on entropy and information gain.
        
        Args:
            examples: training data to use for tree learning, as a list of dictionaries.  The
                attribute stored in self.id_name is ignored, and self.class_name is consided
                the class label.
        
        Returns: a DecisionNode or LeafNode representing the tree

        Recursive function and at each level, it should:
        1) Use a list of examples to identify the best attribute and threshold that to split on
        2) Using that split criteria, then divide the examples and perform the same procedure on that subset
        3) miss_lt parameter (bool) determines the child node to be used to classify examples with a missing value,
        and should specify whichever subtree has more examples in the training data.


        """

        # STEP 1)

        # Check if all class labels are in the example set
        # If they are, return a LeafNode with that class label (the label that this class mostly represents)

        for example in examples:
            example_class_name = example[self.class_name]
            for example_label in example_class_name:
                pred_class = examples[0][self.class_name]
                if example_label == pred_class:
                    # LeafNode(pred_class, pred_class_count, total_count)
                    return LeafNode(pred_class, len(examples), len(examples))

        # STEP 2)

        # Check if this example set is at the minimum leaf count
        # If it is, return a LeafNode with that class label

        if len(examples) == self.min_leaf_count:
            labels = []
            for example in examples:
                labels.append(example[self.class_name])
            pred_class = max(labels, key=labels.count)
            return LeafNode(pred_class, labels.count(pred_class), len(examples))

        # STEP 3)

        # Those were our base/ best case situations. Now we have the bulk of the code
        # Iterate through the examples and go through every attribute and every possible threshold value
        # Best threshold value, we will split on
        # Calculate the info gain that for the iteration's attribute and threshold value
        # Store that value and store the children that have the highest info gain


        else:
            # Get all the possible attributes to split on
            attributes = self.get_attributes(examples)

            # Init a dictionary that holds characteristics of the best attribute, update this with the
            # Best attribute during each iteration
            optimal_attribute = {
                "name":None,
                "threshold":0,
                "info-gain":0,
                "children":(None,None), }

            # BEGIN THE GREAT ITERATION THROUGH THE ATTRIBUTES!!! HAHAHAHHAHAHAHHAAHOHOHOOHEHEHEHHEEHE
            for attribute in attributes:
                # Skip the unnecessary attribute
                if (attribute == self.class_name) or (attribute == "town"):
                    continue

                # GET THRESHOLDS
                thresholds = []
                for example in examples:
                    if example[attribute] != None:
                        thresholds.append(example[attribute])

                for threshold in thresholds:
                    # Split examples into children examples on this threshold and attribute
                    # This way we can optimally analyze each example based on attribute and threshold
                    children = self.split_examples(examples, attribute, threshold)

                 # check if split is valid
                # Avoid the first two attrivutes again because they are unnecessary
                if len(children[0]) < self.min_leaf_count or len(children[1]) < self.min_leaf_count:
                    continue

                # NOW WE GET THE INFO GAIN FOR THIS CHILD USING THIS ITERATION'S ATTRIBUTE AND THRESHOLD
                info_gain = self.gain(examples, children)

                # See if this new info gain is better than the prev optimal
                if info_gain > optimal_attribute["info-gain"]:
                    # If optimal, update our optimal attribute dictionary
                    optimal_attribute["info-gain"] = info_gain
                    optimal_attribute["name"] = attribute
                    optimal_attribute["threshold"] = threshold
                    optimal_attribute["children"] = children

                #
                # Handles if no possbible split exists for this attribute; returns leaf node with all current examples
                if optimal_attribute["name"] is None:
                    labels = []
                    for example in examples:
                        labels.append(example[self.class_name])

                    pred_class = max(labels, key=labels.count)
                    return LeafNode(pred_class, labels.count(pred_class), len(examples))

                    # STEP 4)
                    # Begin recursive phase
                    # Go through the children of the split sections and create decision nodes based on the attributes


                    nodes = []

                    for child in optimal_attribute["children"]: #optimal_attribute["children"] = (Child, Child)
                        node = self.learn_tree(child) # RECURSION
                        nodes.append(node)

                    # TODO May need to flip these around
                    # creates decision node and sets child_miss to child_lt or ge depending on size of lt and ge
                    if len(optimal_attribute["children"][1]) <= len(optimal_attribute["children"][0]):

                        decision_node = DecisionNode(optimal_attribute["name"], optimal_attribute["threshold"], nodes[0], nodes[1],
                                                     nodes[0])
                    else:
                        decision_node = DecisionNode(optimal_attribute["name"], optimal_attribute["threshold"], nodes[0], nodes[1],
                                                     nodes[1])

                return decision_node

    def entrophy(self,examples):
        return None

    def gain(self, examples):
        return None

    def get_attributes(self, example):
        return list(example[0].keys())

    def split(self, examples, splitting_attribute, threshold):
        return None
    
    def classify(self, example):
        """Perform inference on a single example.

        Args:
            example: the instance being classified

        Returns: a tuple containing a class label and a probability

        Notes from assignment worksheet:
        Should use the classify() method defined in other tree nodes
        """
        #
        # fill in the function body here!
        #
        return self.root.classify(example)

    def __str__(self):
        """String representation of tree, calls _ascii_tree()."""
        ln_bef, ln, ln_aft = self._ascii_tree(self.root)
        return "\n".join(ln_bef + [ln] + ln_aft)

    def _ascii_tree(self, node):
        """Super high-tech tree-printing ascii-art madness."""
        indent = 6  # adjust this to decrease or increase width of output 
        if type(node) == LeafNode:
            return [""], "leaf {} {}/{}={:.2f}".format(node.pred_class, node.pred_class_count, node.total_count, node.prob), [""]  
        else:
            child_ln_bef, child_ln, child_ln_aft = self._ascii_tree(node.child_ge)
            lines_before = [ " "*indent*2 + " " + " "*indent + line for line in child_ln_bef ]            
            lines_before.append(" "*indent*2 + u'\u250c' + " >={}----".format(node.test_attr_threshold) + child_ln)
            lines_before.extend([ " "*indent*2 + "|" + " "*indent + line for line in child_ln_aft ])

            line_mid = node.test_attr_name
            
            child_ln_bef, child_ln, child_ln_aft = self._ascii_tree(node.child_lt)
            lines_after = [ " "*indent*2 + "|" + " "*indent + line for line in child_ln_bef ]
            lines_after.append(" "*indent*2 + u'\u2514' + "- <{}----".format(node.test_attr_threshold) + child_ln)
            lines_after.extend([ " "*indent*2 + " " + " "*indent + line for line in child_ln_aft ])

            return lines_before, line_mid, lines_after


def test_model(model, test_examples, label_ordering):
    """Test the tree on the test set and see how we did."""
    correct = 0
    almost = 0  # within one level of correct answer
    test_act_pred = {}
    for example in test_examples:
        actual = example[model.class_name]
        pred, prob = model.classify(example)
        print("{:30} pred {:15} ({:.2f}), actual {:15} {}".format(example[model.id_attr_name] + ':', 
                                                            "'" + pred + "'", prob, 
                                                            "'" + actual + "'",
                                                            '*' if pred == actual else ''))
        if pred == actual:
            correct += 1
        if abs(label_ordering.index(pred) - label_ordering.index(actual)) < 2:
            almost += 1
        test_act_pred[(actual, pred)] = test_act_pred.get((actual, pred), 0) + 1 

    acc = correct/len(test_examples)
    near_acc = almost/len(test_examples)
    return acc, near_acc, test_act_pred


def confusion4x4(labels, vals):
    """Create an normalized predicted vs. actual confusion matrix for four classes."""
    n = sum([ v for v in vals.values() ])
    abbr = [ "".join(w[0] for w in lab.split()) for lab in labels ]
    s =  ""
    s += " actual ___________________________________  \n"
    for ab, labp in zip(abbr, labels):
        row = [ vals.get((labp, laba), 0)/n for laba in labels ]
        s += "       |        |        |        |        | \n"
        s += "  {:^4s} | {:5.2f}  | {:5.2f}  | {:5.2f}  | {:5.2f}  | \n".format(ab, *row)
        s += "       |________|________|________|________| \n"
    s += "          {:^4s}     {:^4s}     {:^4s}     {:^4s} \n".format(*abbr)
    s += "                     predicted \n"
    return s


#############################################

if __name__ == '__main__':

    path_to_csv = 'town_growth_data.csv'
    id_attr_name = 'town'
    class_attr_name = 'growth'
    label_ordering = ['negative', 'small', 'medium', 'large']  # used to count "almost" right
    min_examples = 10  # minimum number of examples for a leaf node

    # read in the data
    examples = read_data(path_to_csv)
    train_examples, test_examples = train_test_split(examples, 0.25)

    # learn a tree from the training set
    tree = DecisionTree(train_examples, id_attr_name, class_attr_name, min_examples)

    # test the tree on the test set and see how we did
    acc, near_acc, test_act_pred = test_model(tree, test_examples, label_ordering)

    # print some stats
    print("\naccuracy: {:.2f}".format(acc))
    print("almost:   {:.2f}\n".format(near_acc))

    # visualize the results and tree in sweet, 8-bit text
    print(confusion4x4(label_ordering, test_act_pred))
    print(tree) 
