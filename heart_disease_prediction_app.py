import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


model = joblib.load('./logistic_model.pkl')
# Function to map probability to risk zone
def get_model(model_choice, model):
    if (model_choice == 'Logistic Regression'):
        model = joblib.load('./logistic_model.pkl')
    elif (model_choice == 'svm'):
        model = joblib.load('./svm_model.pkl')
def get_risk_zone(probability):
    if probability <= 0.25:
        return "Green Zone", "low risk", "green"
    elif probability <= 0.50:
        return "Yellow Zone", "medium-low risk", "#DAA520"
    elif probability <= 0.75:
        return "Orange Zone", "medium-high risk", "orange"
    else:
        return "Red Zone", "high risk", "red"

def get_status(value, normal_range):
    """"Return arrow prompts based on the relationship of the value to the normal range"""
    try:
        if float(value) < normal_range[0]:
            return f"<span style='color: red;'>⬇️ Below Normal</span>"
        elif float(value) > normal_range[1]:
            return f"<span style='color: red;'>⬆️ Above Normal</span>"
        else:
            return "✅ Normal"
    except ValueError:
        return ""

def plot_input_data(age, sex, restingecg, st_slope, chestpain, exerciseangina, bp, cholesterol, fastingbs, maxhr, oldpeak):
    normal_ranges = {
        "RestingBP": [60, 130],
        "Cholesterol": [125, 200],
        "FastingBS": [70, 100],
        "MaxHR": [60, 200],
        "Oldpeak": [-5, 5]
    }

    features_data = [
        {"Feature": "Age", "Value": age},
        {"Feature": "Sex", "Value": sex},
        {"Feature": "RestingECG", "Value": restingecg},
        {"Feature": "ST Slope", "Value": st_slope},
        {"Feature": "Chest Pain Type", "Value": chestpain},
        {"Feature": "Exercise Angina", "Value": exerciseangina},
        {"Feature": "RestingBP", "Value": f"{bp} {get_status(bp, normal_ranges['RestingBP']) if 'RestingBP' in normal_ranges else ''}"},
        {"Feature": "Cholesterol", "Value": f"{cholesterol} {get_status(cholesterol, normal_ranges['Cholesterol']) if 'Cholesterol' in normal_ranges else ''}"},
        {"Feature": "FastingBS", "Value": f"{fastingbs} {get_status(fastingbs, normal_ranges['FastingBS']) if 'FastingBS' in normal_ranges else ''}"},
        {"Feature": "MaxHR", "Value": f"{maxhr} {get_status(maxhr, normal_ranges['MaxHR']) if 'MaxHR' in normal_ranges else ''}"},
        {"Feature": "Oldpeak", "Value": f"{oldpeak} {get_status(oldpeak, normal_ranges['Oldpeak']) if 'Oldpeak' in normal_ranges else ''}"},
    ]

    # create DataFrame
    df = pd.DataFrame(features_data).set_index('Feature')
    transposed_df = df.T

    st.subheader("User Input Data")
    st.write(f"<p style='font-size: 14px'>",transposed_df.to_html(escape=False), unsafe_allow_html=True)

# Function to predict the heart disease probability and calculate feature contributions
def predict_heart_disease_and_contributions(age, sex, bp, cholesterol, fastingbs, maxhr, restingecg, oldpeak, st_slope, chestpain, exerciseangina):
    # Convert fasting blood sugar to binary (1 if fastingbs > 120, else 0)
    fastingbs = 1 if float(fastingbs) > 120 else 0

    # Data Conversion for RestingECG
    if restingecg == 'ST-T wave abnormality':
        restingecg = 'ST'
    elif restingecg == 'Left ventricular hypertrophy':
        restingecg = 'LVH'

        # Data Conversion for ChestPainType (convert to standard labels)
    chestpain_map = {
        'Typical Angina': 'TA',  # Typical Angina
        'Atypical Angina': 'ATA',  # Atypical Angina
        'Non-Anginal Pain': 'NAP',  # Non-Anginal Pain
        'Asymptomatic': 'ASY'  # Asymptomatic
    }
    chestpain = chestpain_map.get(chestpain, chestpain)
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'RestingBP': [bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fastingbs],
        'MaxHR': [maxhr],
        'RestingECG': [restingecg],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope],
        'ChestPainType': [chestpain],
        'ExerciseAngina': [exerciseangina]
    })

    # Preprocess using the pipeline
    preprocessor = model.named_steps['preprocessor']
    sample_processed = preprocessor.transform(input_data)

    # Get model coefficients and intercept
    coefficients = model.named_steps['model'].coef_[0]
    intercept = model.named_steps['model'].intercept_[0]

    # Calculate contributions of each feature
    contributions = sample_processed.flatten() * coefficients
    log_odds = contributions.sum() + intercept  # log-odds sum

    # Calculate the predicted probability using the logistic function (sigmoid)
    predicted_probability = 1 / (1 + np.exp(-log_odds))

    # Get risk zone and color
    risk_zone, risk_description, color = get_risk_zone(predicted_probability)

    # Get feature names (same as used during training)
    categorical_features = ['Sex', 'ChestPainType', 'ExerciseAngina', 'ST_Slope']
    cat_feature_names = preprocessor.transformers_[1][1].get_feature_names_out(categorical_features)
    numerical_features = ['Age', 'FastingBS', 'MaxHR', 'Oldpeak']
    feature_names = numerical_features + list(cat_feature_names)

    # Create DataFrame of feature names and their contributions
    contrib_df = pd.DataFrame({
        'Feature': feature_names,
        'Contribution': contributions
    })

    # Remove features with zero contribution
    contrib_df = contrib_df[contrib_df['Contribution'] != 0]

    # Sort by absolute contribution (optional)
    contrib_df['AbsContribution'] = np.abs(contrib_df['Contribution'])
    contrib_df = contrib_df.sort_values(by='AbsContribution', ascending=False)

    return contrib_df, predicted_probability, risk_zone, risk_description, color

# Function to plot feature contributions as bar charts
def plot_feature_contributions(contrib_df):
    # Sort by absolute contribution (from high to low)
    contrib_df['AbsContribution'] = np.abs(contrib_df['Contribution'])
    contrib_df = contrib_df.sort_values(by='AbsContribution', ascending=True)

    # Create a figure and plot the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(contrib_df['Feature'], contrib_df['Contribution'], color='skyblue')

    # Set the labels and title
    ax.set_xlabel('Contribution Value')
    ax.set_title('Feature Contributions to Predicted Log-Odds of Heart Disease')

    # Show the plot in Streamlit
    st.pyplot(fig)  # Display the plot in Streamlit
    # Identify features with contribution > 0.5, excluding 'Age' and 'Sex'
    significant_features = contrib_df[(contrib_df['Contribution'] > 0.5) &
                                      ~contrib_df['Feature'].isin(['Age', 'Sex_M', 'Sex_F'])]

    if not significant_features.empty:
        st.write("### Features with High Contribution:")
        for index, row in significant_features.iterrows():
            feature = row['Feature']
            contribution = row['Contribution']
            st.write(
                f"- **{feature}**: This feature has a significant contribution of {contribution:.2f} to the predicted log-odds of heart disease. Please pay special attention to this feature.")

# Main function for Streamlit app
def predict_proba(model_choice):
    # Load the pre-trained model
    get_model(model_choice, model)
    # Page Layout
    st.title("Heart Disease Prediction using ML")

    # Information about the tool
    st.markdown("""
           ### Descriptions:
           To predict your heart disease status, simply follow the steps below:
           Enter the parameters that best describe you in an honest manner,
           Press the **"Predict"** button at the end and wait for your result.
           This model predicts the probability of heart disease based on input data. 
           The probability is then categorized into risk zones for better understanding:
           - **Green Zone** (0-25%): Low risk
           - **Yellow Zone** (25-50%): Medium-low risk
           - **Orange Zone** (50-75%): Medium-high risk
           - **Red Zone** (75-100%): High risk

           **Note**: This model is for educational purposes only. Always consult with a healthcare professional for accurate diagnosis.
       """)

    # Sidebar for input data
    st.sidebar.header("Provide Input Data")

    # Input Fields
    age = st.sidebar.number_input("**Age** (years)", min_value=0, max_value=120, value=0)
    sex = st.sidebar.selectbox("**Sex** (M: Male, F: Female)", ["M", "F"], index=0)
    bp = st.sidebar.number_input("**RestingBP** : resting blood pressure (mm Hg)", min_value=0, max_value=250, value=0)
    cholesterol = st.sidebar.number_input("**Cholesterol** : serum cholesterol (mg/dL)", min_value=0, max_value=800,
                                          value=0)
    fastingbs = st.sidebar.number_input("**FastingBS** : fasting blood sugar (mg/dL)", min_value=0, max_value=250,
                                        value=0)
    maxhr = st.sidebar.number_input("**MaxHR** : maximum heart rate achieved (bpm)", min_value=0, max_value=220,
                                    value=0)
    restingecg = st.sidebar.selectbox("Resting ECG",
                                      ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"], index=0)
    oldpeak = st.sidebar.number_input("**Oldpeak** : ST depression induced by exercise", min_value=-5.0, max_value=5.0,
                                      value=0.0)
    st_slope = st.sidebar.selectbox("**ST Slope** : the slope of the peak exercise ST segment", ["Up", "Flat", "Down"],
                                    index=0)
    chestpain = st.sidebar.selectbox("**Chest Pain Type** : chest pain type",
                                     ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"], index=0)
    exerciseangina = st.sidebar.selectbox("**Exercise Angina** : exercise-induced angina [Yes: Y, No: N]", ["Y", "N"],
                                          index=0)

    # Predict when user clicks "Predict" button
    if st.sidebar.button("Predict"):
        if age == 0 or bp == 0 or cholesterol == 0 or fastingbs == 0 or maxhr == 0:
            st.warning("Please enter valid numerical values. All numerical inputs cannot be zero.")
        else:
            # Get feature contributions and predicted probability
            contrib_df, predicted_probability, risk_zone, risk_description, color = predict_heart_disease_and_contributions(
                age, sex, bp, cholesterol, fastingbs, maxhr, restingecg, oldpeak, st_slope, chestpain, exerciseangina
            )

            # Display prediction result with color-coded text
            result_text = f"The probability that you will have heart disease is {predicted_probability * 100:.2f}%. You are in the {risk_zone}."
            st.markdown(f"<p style='font-size: 28px; color:{color};'><strong>{result_text}</strong></p>", unsafe_allow_html=True)

            # Provide advice based on the risk zone
            if risk_zone == "Green Zone":
                st.write("You are at a low risk of heart disease. Maintain a healthy lifestyle!")
            elif risk_zone == "Yellow Zone":
                st.write("You have a medium-low risk of heart disease. Consider regular checkups.")
            elif risk_zone == "Orange Zone":
                st.write("You have a medium-high risk of heart disease. Please consult a healthcare provider.")
            else:
                st.write("You are at high risk of heart disease. Immediate consultation with a doctor is recommended.")

            # Check if the predicted probability is greater than 75%
            if predicted_probability > 0.75:
                # Display warning message in red and bold
                st.markdown(
                    "<p style='font-size: 24px; color: red;'><strong>Based on your input data, your risk of heart disease is relatively high. Please visit a hospital for a check-up and medical consultation as soon as possible.</strong></p>",
                    unsafe_allow_html=True
                )
                # Display hospital image
                st.image("./pic_hospital.png", caption="Hospital")
                # Provide a link to the hospital's website
                st.markdown(
                    "<p style='font-size: 16px;'>Click the link to visit the hospital's website: <a href='http://www.srrsh.com/' target='_blank'>Hospital Website</a></p>",
                    unsafe_allow_html=True
                )

            # Check if the predicted probability is less than 0.25
            elif predicted_probability < 0.25:
                #Display congratulatory message with a thumbs-up image
                st.markdown(
                    "<p style='font-size: 24px; color: green;'><strong>Great job! Your risk of heart disease is very low. Keep up the good work!</strong></p>",
                    unsafe_allow_html=True
                )
                # Display thumbs-up image
                st.image("./pic_good.png", caption="Well Done!",  width=200)


            # Display feature contributions as bar chart
            plot_feature_contributions(contrib_df)

            # Display User Input Data
            plot_input_data(age, sex, restingecg, st_slope, chestpain, exerciseangina, bp, cholesterol, fastingbs, maxhr, oldpeak)

if __name__ == "__main__":
    predict_proba()
