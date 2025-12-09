# load the feature data(sepal,petal length)in X and labels(flower name) in y
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data      # shape (150, 4)
y = iris.target    # shape (150,)
print(iris.feature_names, iris.target_names)


#split the data in x and Y for test and training
#test size indicates the percentage split from whole data 
#Adding a value to random_state brings same data for every run
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)


#training a model with selected algorithm
from sklearn.tree import DecisionTreeClassifier
import joblib
model = DecisionTreeClassifier(random_state=42)
#train the model with modelfit function
model.fit(X_train, y_train)
#save the model to a file
with open('output/model.joblib', 'a'):
 joblib.dump(model,'output/model.joblib')


#after model is trained it can be used to test.
y_pred = model.predict(X_test)
print(y_pred)
#verify model prediction is same as test results
with open('output/output.txt', 'a') as f:
 print('\n',"Predictions:", y_pred[:6],file=f)
 print('\n',"True labels:", y_test[:6],file=f)

#evaluate the model by checking the accuracy with test and predicted data.
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
with open('output/output.txt', 'a') as f:
 print('\n'"Accuracy:",accuracy, file=f)

#evaluate the model overall performance with confusion matrix predicted vs true labels as below
#Predicted →    Setosa  Versicolor  Virginica
#True ↓
#Setosa           x         0        0
#Versicolor       0         y        y
#Virginica        0         z        z
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
matrix = confusion_matrix(y_test, y_pred)
print("matrix:", matrix)
cm_display = ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels = ['setosa', 'versicolor','virginica'])
cm_display.plot()
plt.savefig("output/Confusion_Matrix.png")
plt.show()
plt.clf()

#visualize the entire structure with plot tree model 
from sklearn.tree import DecisionTreeClassifier, plot_tree
plot_tree(model) 
plt.savefig("output/Plot_tree.png")
plt.clf()


#check the accuracy with different algorithm
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print("k-NN accuracy:", accuracy_score(y_test, y_pred2))

#tune parameter as below, the depth of tree shall be selected in DecisionTreeClassifier
model3 = DecisionTreeClassifier(max_depth=3, random_state=42)
model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)
print("k-NN accuracy:", accuracy_score(y_test, y_pred3))
