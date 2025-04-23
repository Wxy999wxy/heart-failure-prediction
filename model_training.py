import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import RocCurveDisplay

def load_and_process_data():
    data = pd.read_csv("./heart.csv")
    data['Cholesterol'] = data['Cholesterol'].replace(0, np.nan)
    mean_cholesterol_male = data[data['Sex'] == 'M']['Cholesterol'].mean()
    mean_cholesterol_female = data[data['Sex'] == 'F']['Cholesterol'].mean()
    data.loc[(data['Sex'] == 'M') & (data['Cholesterol'].isna()), 'Cholesterol'] = mean_cholesterol_male
    data.loc[(data['Sex'] == 'F') & (data['Cholesterol'].isna()), 'Cholesterol'] = mean_cholesterol_female
    data = data[data['RestingBP'] != 0]
    data = data[(data['MaxHR'] >= 60) & (data['MaxHR'] <= 202)]
    data = data[(data['FastingBS'] == 0) | (data['FastingBS'] == 1)]
    data = data[data['Sex'].isin(["M", "F"])]
    return data

def get_preprocessor():
    numerical_features = ['Age', 'FastingBS', 'MaxHR', 'Oldpeak']
    categorical_features = ['Sex', 'ChestPainType',  'ExerciseAngina', 'ST_Slope']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    return preprocessor


# 训练并保存逻辑回归模型
def train_logistic_regression_and_save(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000, C=0.1, solver='liblinear')
    preprocessor = get_preprocessor()
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # 训练模型
    pipeline.fit(X_train, y_train)

    # 获取预测概率
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # 风险级别分类
    def get_risk_level(prob):
        if prob <= 0.25:
            return "Green Zone"
        elif prob <= 0.50:
            return "Yellow Zone"
        elif prob <= 0.75:
            return "Orange Zone"
        else:
            return "Red Zone"

    # 显示前10个样本的预测概率和风险等级
    for i, prob in enumerate(probabilities[:10]):
        risk_level = get_risk_level(prob)


    # 保存模型
    model_path = "./logistic_model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"Logistic Regression Model saved to {model_path}")

    # 评估模型
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # print(f"Logistic Regression: Accuracy={accuracy:.4f}, F1 Score={f1:.4f}, ROC-AUC={roc_auc:.4f}")
    # print("Confusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))
    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred))


# 训练并保存SVM模型
def train_svm_and_save(X_train, X_test, y_train, y_test):
    model = SVC(probability=True, kernel='linear', C=0.1)
    preprocessor = get_preprocessor()
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # 训练模型
    pipeline.fit(X_train, y_train)

    # 获取预测概率
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # 风险级别分类
    def get_risk_level(prob):
        if prob <= 0.25:
            return "Green Zone"
        elif prob <= 0.50:
            return "Yellow Zone"
        elif prob <= 0.75:
            return "Orange Zone"
        else:
            return "Red Zone"

    # 显示前10个样本的预测概率和风险等级
    for i, prob in enumerate(probabilities[:10]):
        risk_level = get_risk_level(prob)
        # print(f"Sample {i + 1}: Probability = {prob * 100:.2f}%, Risk Level = {risk_level}")

    # 保存模型
    model_path = "./svm_model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"SVM Model saved to {model_path}")

    # 评估模型
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # print(f"SVM: Accuracy={accuracy:.4f}, F1 Score={f1:.4f}, ROC-AUC={roc_auc:.4f}")
    # print("Confusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))
    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred))



def model_save():
    # 载入并处理数据
    data = load_and_process_data()
    X = data.drop(columns=["HeartDisease"])
    y = data["HeartDisease"]

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练并保存逻辑回归模型
    train_logistic_regression_and_save(X_train, X_test, y_train, y_test)

    # 训练并保存SVM模型
    train_svm_and_save(X_train, X_test, y_train, y_test)

def evaluate_and_plot(model, model_name, X_train, X_test, y_train, y_test):
    # Fit the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Confusion Matrix and Classification Report
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Create 1x2 subplots for Confusion Matrix and ROC Curve
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot Confusion Matrix
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', ax=axes[0])
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    axes[0].set_title(f"{model_name} Confusion Matrix")

    # Plot ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[1])
    axes[1].set_title(f"{model_name} ROC Curve")

    # Show plots
    st.pyplot(fig)

    return accuracy, f1, roc_auc

def logistic_regression_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    return evaluate_and_plot(model, "Logistic Regression", X_train, X_test, y_train, y_test)

def random_forest_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    return evaluate_and_plot(model, "Random Forest", X_train, X_test, y_train, y_test)

def gradient_boosting_model(X_train, X_test, y_train, y_test):
    model = GradientBoostingClassifier()
    return evaluate_and_plot(model, "Gradient Boosting", X_train, X_test, y_train, y_test)

def svm_model(X_train, X_test, y_train, y_test):
    model = SVC(probability=True)
    return evaluate_and_plot(model, "SVM", X_train, X_test, y_train, y_test)

def knn_model(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier()
    return evaluate_and_plot(model, "KNN", X_train, X_test, y_train, y_test)

def display_model_results():
    # Load and preprocess data
    data = load_and_process_data()
    numerical_features = ['Age', 'FastingBS', 'MaxHR', 'Oldpeak']
    categorical_features = ['Sex', 'ChestPainType', 'ExerciseAngina', 'ST_Slope']

    # Define ColumnTransformer with OneHotEncoder and StandardScaler
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), numerical_features),
                      ('cat', OneHotEncoder(), categorical_features)])

    # Separate features and target variable
    X = data.drop(columns=["HeartDisease"])
    y = data["HeartDisease"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply preprocessing to both training and test data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Evaluate and display results for each model with enhanced formatting
    st.write("### Evaluating **Logistic Regression**:")
    accuracy, f1, roc_auc = logistic_regression_model(X_train, X_test, y_train, y_test)
    st.write(f"<h3 style='font-size:20px;'><b>Accuracy:</b> {accuracy:.4f}, <b>F1 Score:</b> {f1:.4f}, <b>ROC-AUC:</b> {roc_auc:.4f}</h3>", unsafe_allow_html=True)
    st.markdown("""
        - Confusion Matrix:
          - True Negatives (TN) = 64: Correctly predicted "no heart disease".
          - False Positives (FP) = 8: Incorrectly predicted "heart disease".
          - False Negatives (FN) = 16: Incorrectly predicted "no heart disease".
          - True Positives (TP) = 96: Correctly predicted "heart disease".
        - Classification Report:
          - Precision for positive class: 0.92
          - Recall for positive class: 0.86
          - F1 Score for positive class: 0.8889
          - Accuracy: 0.8696
          - ROC-AUC: 0.9214
        """)
    st.write(
        "<h4 style='font-size:18px;'><b>Indicates that the model performs better under all possible classification thresholds and has a strong classification ability.</b></h4>",
        unsafe_allow_html=True)

    st.write("### Evaluating **Random Forest**:")
    accuracy, f1, roc_auc = random_forest_model(X_train, X_test, y_train, y_test)
    st.write(
        f"<h3 style='font-size:20px;'><b>Accuracy:</b> {accuracy:.4f}, <b>F1 Score:</b> {f1:.4f}, <b>ROC-AUC:</b> {roc_auc:.4f}</h3>",
        unsafe_allow_html=True)
    st.markdown("""
            - Confusion Matrix:
              - True Negatives (TN) = 60
              - False Positives (FP) = 12
              - False Negatives (FN) = 17
              - True Positives (TP) = 95
            - Classification Report:
              - Precision for positive class: 0.89
              - Recall for positive class: 0.83
              - F1 Score for positive class: 0.8676
              - Accuracy: 0.8424
              - ROC-AUC: 0.9138
        """)
    st.write(
        "<h4 style='font-size:18px;'><b>Indicates that the model's classification ability is strong, distinguishing positive and negative classes effectively.</b></h4>",
        unsafe_allow_html=True)

    st.write("### Evaluating **Gradient Boosting**:")
    accuracy, f1, roc_auc = gradient_boosting_model(X_train, X_test, y_train, y_test)
    st.write(
        f"<h3 style='font-size:20px;'><b>Accuracy:</b> {accuracy:.4f}, <b>F1 Score:</b> {f1:.4f}, <b>ROC-AUC:</b> {roc_auc:.4f}</h3>",
        unsafe_allow_html=True)
    st.markdown("""
            - Confusion Matrix:
              - True Negatives (TN) = 62
              - False Positives (FP) = 10
              - False Negatives (FN) = 20
              - True Positives (TP) = 92
            - Classification Report:
              - Precision for positive class: 0.90
              - Recall for positive class: 0.82
              - F1 Score for positive class: 0.8598
              - Accuracy: 0.8370
              - ROC-AUC: 0.9232
        """)
    st.write(
        "<h4 style='font-size:18px;'><b>Indicates that the model performs well at distinguishing positive and negative samples, with a good balance of precision and recall.</b></h4>",
        unsafe_allow_html=True)

    st.write("### Evaluating **SVM**:")
    accuracy, f1, roc_auc = svm_model(X_train, X_test, y_train, y_test)
    st.write(
        f"<h3 style='font-size:20px;'><b>Accuracy:</b> {accuracy:.4f}, <b>F1 Score:</b> {f1:.4f}, <b>ROC-AUC:</b> {roc_auc:.4f}</h3>",
        unsafe_allow_html=True)
    st.markdown("""
            - Confusion Matrix:
              - True Negatives (TN) = 62
              - False Positives (FP) = 10
              - False Negatives (FN) = 13
              - True Positives (TP) = 99
            - Classification Report:
              - Precision for positive class: 0.91
              - Recall for positive class: 0.88
              - F1 Score for positive class: 0.8959
              - Accuracy: 0.8750
              - ROC-AUC: 0.9281
        """)
    st.write(
        "<h4 style='font-size:18px;'><b>Indicates that the model performs excellently with a high ROC-AUC score, distinguishing positive and negative classes very effectively.</b></h4>",
        unsafe_allow_html=True)

    st.write("### Evaluating **KNN**:")
    accuracy, f1, roc_auc = knn_model(X_train, X_test, y_train, y_test)
    st.write(
        f"<h3 style='font-size:20px;'><b>Accuracy:</b> {accuracy:.4f}, <b>F1 Score:</b> {f1:.4f}, <b>ROC-AUC:</b> {roc_auc:.4f}</h3>",
        unsafe_allow_html=True)
    st.markdown("""
            - Confusion Matrix:
              - True Negatives (TN) = 63
              - False Positives (FP) = 9
              - False Negatives (FN) = 15
              - True Positives (TP) = 97
            - Classification Report:
              - Precision for positive class: 0.92
              - Recall for positive class: 0.87
              - F1 Score for positive class: 0.8899
              - Accuracy: 0.8696
              - ROC-AUC: 0.9132
        """)
    st.write(
        "<h4 style='font-size:18px;'><b>Indicates that the model is strong at predicting positive cases with a relatively high accuracy and recall.</b></h4>",
        unsafe_allow_html=True)

    st.markdown("""
    ### **Summary**:
SVM performs best in terms of accuracy, F1 scores and ROC-AUC, especially in the prediction of positive classes (with heart disease.) The high F1 score means that it does a good job of balancing precision and recall.

Logistic Regression and KNN are close in performance, with relatively high precision and F1 scores, especially for the positive class.

Random Forest and Gradient Boosting perform better in predicting the negative class (no heart disease), but are slightly worse in recall for the positive class.

    """
                )


if __name__ == "__main__":
    display_model_results()
