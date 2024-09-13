# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Feature matrix
y = iris.target  # Target vector
feature_names = iris.feature_names
class_names = iris.target_names

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X, y)

# Create a figure and axes
plt.figure(figsize=(12, 8))

# Plot the decision tree
plot_tree(clf,
          feature_names=feature_names,
        #   class_names=class_names,
          filled=True,
          rounded=True,
          impurity=True,
          fontsize=12)

# Display the plot
plt.savefig('tree.png')
