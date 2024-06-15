import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


diabetes = pd.read_csv('diabetes.csv')


print(diabetes.head())
print(diabetes.info())


sns.countplot(x="Outcome", data=diabetes)


plt.figure(figsize=(10, 8))
sns.heatmap(diabetes.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


X = diabetes.drop(columns='Outcome', axis=1)
Y = diabetes['Outcome']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


model = LogisticRegression()


model.fit(X_train, Y_train)


Y_pred = model.predict(X_test)


conf_matrix = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:\n", conf_matrix)

accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

class_report = classification_report(Y_test, Y_pred)
print("Classification Report:\n", class_report)
