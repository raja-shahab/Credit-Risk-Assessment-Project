from django.shortcuts import render
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load the dataset and train the model
data = pd.read_csv("E:\\Django Projects 2\\credit_risk_assesment\\data\\credit_risk_dataset.csv")
X = data[['cb_person_cred_hist_length', 'person_income', 'loan_percent_income']]
y = data['cb_person_default_on_file']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

def home(request):
    prediction = None
    if 'credit_score' in request.GET and 'income' in request.GET and 'debt_to_income_ratio' in request.GET:
        credit_score = float(request.GET['credit_score'])
        income = float(request.GET['income'])
        debt_to_income_ratio = float(request.GET['debt_to_income_ratio'])

        new_data = pd.DataFrame({
            'cb_person_cred_hist_length': [credit_score],
            'person_income': [income],
            'loan_percent_income': [debt_to_income_ratio]
        })

        prediction = model.predict(new_data)[0]

        risk_level = "High Risk" if prediction == 'Y' else "Low Risk"
        return render(request, 'home.html', {'prediction': risk_level})

    return render(request, 'home.html', {'prediction': prediction})
