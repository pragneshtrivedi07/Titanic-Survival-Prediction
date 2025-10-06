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
data = pd.get_dummies(data, columns=['Embarked'])  # No drop_first this time!

# Ensure all Embarked columns exist
for col in ['Embarked_C', 'Embarked_Q', 'Embarked_S']:
    if col not in data.columns:
        data[col] = 0

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
st.title("🚢 Titanic Survival Prediction Dashboard")
st.write("Enter passenger details below:")

# User Inputs
pclass = st.selectbox("Passenger Class (1=First, 2=Second, 3=Third)", [1, 2, 3])
sex = st.selectbox("Sex", ["female", "male"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.slider("Number of Siblings/Spouses aboard", 0, 10, 0)
parch = st.slider("Number of Parents/Children aboard", 0, 10, 0)
fare = st.slider("Fare", 0.0, 500.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Map inputs to model features
sex = 0 if sex == "female" else 1
emb_C = 1 if embarked == "C" else 0
emb_Q = 1 if embarked == "Q" else 0
emb_S = 1 if embarked == "S" else 0

# Create input DataFrame matching training features exactly
input_data = pd.DataFrame(
    [[pclass, sex, age, sibsp, parch, fare, emb_C, emb_Q, emb_S]],
    columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
)

# ----------------------
# Predictions
# ----------------------
import matplotlib.pyplot as plt

# ----------------------
# Prediction + Visualization Function
# ----------------------
def show_prediction(model_name, model):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader(f"🎯 Prediction using {model_name}")
    st.write(f"Result: {'🟩 Survived' if prediction == 1 else '🟥 Did Not Survive'}")
    st.write(f"Probability of Survival: **{prob:.2%}**")

    # Probability bar visualization
    fig, ax = plt.subplots(figsize=(5, 0.4))
    ax.barh([""], [prob], color="green" if prob > 0.5 else "red")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("Probability of Survival")
    ax.set_yticks([])
    ax.set_facecolor("#f5f5f5")
    plt.tight_layout()
    st.pyplot(fig)


# ----------------------
# Separate Buttons with Unique Keys
# ----------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("Predict with Logistic Regression", key="log_btn"):
        show_prediction("Logistic Regression", log_model)

with col2:
    if st.button("Predict with Random Forest", key="rf_btn"):
        show_prediction("Random Forest", rf_model)
import numpy as np

# ----------------------
# Feature Importance Visualization
# ----------------------
st.markdown("---")
st.header("📊 Feature Importance (Random Forest)")

# Get feature importances from trained Random Forest model
importances = rf_model.feature_importances_
feature_names = X.columns

# Create a sorted DataFrame for plotting
feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

# Plot using matplotlib
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(feat_imp['Feature'], feat_imp['Importance'], color='skyblue')
ax.invert_yaxis()  # Highest importance at the top
ax.set_xlabel('Importance Score')
ax.set_title('Top Features Influencing Survival')
st.pyplot(fig)

# Optional summary below chart
st.write("🧠 **Insights:**")
st.write("- Higher importance means the feature strongly influences the model’s prediction.")
st.write("- Typically, factors like **Sex**, **Pclass**, and **Fare** are the most impactful for survival.")

