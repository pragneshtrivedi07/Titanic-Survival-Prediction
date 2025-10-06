import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ----------------------
# Load & Preprocess Data
# ----------------------
data = pd.read_csv('train.csv')

# Drop unnecessary columns
data = data.drop(columns=['Cabin', 'PassengerId', 'Name', 'Ticket'])

# Fill missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Encode categorical variables
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)  # Drop first to avoid dummy trap

# Features & target
X = data.drop('Survived', axis=1)
y = data['Survived']

# ----------------------
# Train Models
# ----------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X, y)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# ----------------------
# Streamlit Dashboard
# ----------------------
st.title("Titanic Survival Prediction Dashboard")
st.write("Enter passenger details below:")

# User Inputs
pclass = st.selectbox("Passenger Class (1=First, 2=Second, 3=Third)", [1,2,3])
sex = st.selectbox("Sex", ["female","male"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.slider("Number of Siblings/Spouses aboard", 0, 10, 0)
parch = st.slider("Number of Parents/Children aboard", 0, 10, 0)
fare = st.slider("Fare", 0.0, 500.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Map inputs to model features
sex = 0 if sex=="female" else 1
emb_C = 1 if embarked=="C" else 0
emb_Q = 1 if embarked=="Q" else 0
# Embarked_S is implicit: if emb_C=0 and emb_Q=0 → Embarked=S

# Create input DataFrame matching training features
input_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, emb_C, emb_Q]],
                          columns=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_C','Embarked_Q'])

# Prediction buttons
if st.button("Predict with Logistic Regression"):
    prediction = log_model.predict(input_data)[0]
    prob = log_model.predict_proba(input_data)[0][1]
    st.write(f"Prediction: {'Survived' if prediction==1 else 'Did Not Survive'}")
    st.write(f"Survival Probability: {prob:.2f}")

if st.button("Predict with Random Forest"):
    prediction = rf_model.predict(input_data)[0]
    prob = rf_model.predict_proba(input_data)[0][1]
    st.write(f"Prediction: {'Survived' if prediction==1 else 'Did Not Survive'}")
    st.write(f"Survival Probability: {prob:.2f}")
