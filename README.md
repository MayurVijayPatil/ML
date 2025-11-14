import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

data = pd.read_csv('student_results.csv')
data.head()

X = data[['StudyHours', 'Attendance']]
y = data['Result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics_table = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [round(acc, 2), round(prec, 2), round(rec, 2), round(f1, 2)]
})

print("✅ Model Evaluation Metrics\n")
display(metrics_table)  

print("\n✅ Detailed Classification Report\n")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
display(report_df.round(2))

plt.figure(figsize=(6,4))
sns.heatmap(cm,
            annot=True,       
            fmt='d',          
            cmap='BuGn',    
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix (Logistic Regression)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

new_student = np.array([[5.5, 85]])
prediction = model.predict(new_student)
print("Prediction (1=Pass, 0=Fail):", prediction[0])
