import streamlit as st
from dataplot import run_data_analysis
from heart_disease_prediction_app import predict_proba
from model_training import display_model_results, model_save

def main():
    model_save()
    page = st.sidebar.selectbox("Choose a page:",
                                ["Home", "Dataset Visualization and Analysis",
                                 "Heart Disease Prediction and Feature Contribution Analysis",
                                 "Understand the model training process"])


    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = "Logistic Regression"
    if page == "Home":

        st.title("Heart Disease Prediction System")

        st.image("./pic1.png",  use_container_width=True)

        st.markdown("""
            ## About the Project:
            This is a heart disease prediction system. The model predicts the probability of heart disease based on various input factors.
            You can explore the following sections:
            - **Dataset Visualization and Analysis**: Explore the dataset and see various visualizations.
            - **Heart Disease Prediction and Feature Contribution Analysis**: Input your data and get a prediction along with feature contributions.
            - **Understand the Model Training Process**: Learn about how the model was trained, the scores, and the final performance.
        """)
        st.sidebar.markdown("""
        ## Author Information
        **Name**: Yiling Chen

        **Introduction**:  
        I am specializing in the development of heart disease prediction models.  I am committed to improving medical diagnostics through machine learning technologies.  
        **Project Introduction**:  
        Design and implementation of heart failure prediction software based on Python  
        [Heart Failure Prediction Dataset](https://www.kaggle.com/fedesoriano/heart-failure-prediction/code)  
        Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5 CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. This software uses machine learning algorithms to develop a model that can predict heart failure in patients and improve people's health.  

        **Contact Information**:  
        - Email: 1160039933@qq.com  
        - University of Sussex-Contact me'yc521@sussex.ac.uk'""")

    elif page == "Dataset Visualization and Analysis":
        # "Dataset Visualization and Analysis"页面
        st.sidebar.markdown("""
            ### Aim :
            - To classify / predict whether a patient is prone to heart failure depending on multiple attributes.
            - It is a **binary classification** with multiple numerical and categorical features.

            ### Dataset Attributes :
            Source of datasets： https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data
    
            - **Age** : age of the patient [years]
            - **Sex** : sex of the patient [M: Male, F: Female]
            - **ChestPainType** : chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
            - **RestingBP** : resting blood pressure [mm Hg]
            - **Cholesterol** : serum cholesterol [mm/dl]
            - **FastingBS** : fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
            - **RestingECG** : resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
            - **MaxHR** : maximum heart rate achieved [Numeric value between 60 and 202]
            - **ExerciseAngina** : exercise-induced angina [Y: Yes, N: No]
            - **Oldpeak** : oldpeak = ST [Numeric value measured in depression]
            - **ST_Slope** : the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
            - **HeartDisease** : output class [1: heart disease, 0: Normal]

        """)
        st.title("Dataset Visualization and Analysis")
        run_data_analysis()  # 调用dataplot.py中的分析和可视化函数
    elif page == "Understand the model training process":
        # "Understand the model training process"页面
        st.title("Understand the Model Training Process")

        # Show model training scores and evaluation
        st.markdown("## Model Training and Evaluation:")
        st.markdown(
            "The model was trained using logistic regression on a dataset that includes features like age, cholesterol levels, resting blood pressure, etc.")
        st.markdown(
            "The performance of the model was evaluated using metrics such as accuracy, precision, recall, and AUC (Area Under the Curve).")
        # 在边栏添加模型选择单选按钮
        model_choice = st.sidebar.radio("Choose a model:", ["Logistic Regression", "SVM"])
        st.session_state.model_choice = model_choice  # 保存用户选择到会话状态

        st.sidebar.markdown("""
                   ## **Recommendation**:
                   If you are more concerned with accurately identifying sick people (positive class), then SVM and Logistic Regression would be the superior choice.

                   If the focus is on balanced performance of the model (i.e., balance of precision and recall), then the combined performance of SVM would be the superior choice.
               """)
        display_model_results()

    elif page == "Heart Disease Prediction and Feature Contribution Analysis":
        st.sidebar.write("You selected:", st.session_state.model_choice)
        predict_proba(st.session_state.model_choice)




if __name__ == "__main__":
    main()
