import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import LabelEncoder
import matplotlib

# Function to load and process the dataset
def load_and_process_data():
    data = pd.read_csv('./heart.csv')

    # Data Preliminary Check
    st.write("### Data Preliminary Check")

    # Display the first few rows of the data
    st.write("#### First Few Rows of the Data")
    st.write(data.head())

    # Display the shape of the data
    st.write("#### Shape of the Data")
    st.write(data.shape)

    # Replace records with a cholesterol value of 0 with NaN
    data['Cholesterol'] = data['Cholesterol'].replace(0, np.nan)

    # Plot missing values heatmap
    plot_missing_values(data)

    # Replace missing cholesterol values with mean cholesterol based on gender
    mean_cholesterol_male = data[data['Sex'] == 'M']['Cholesterol'].mean()
    mean_cholesterol_female = data[data['Sex'] == 'F']['Cholesterol'].mean()
    data.loc[(data['Sex'] == 'M') & (data['Cholesterol'].isna()), 'Cholesterol'] = mean_cholesterol_male
    data.loc[(data['Sex'] == 'F') & (data['Cholesterol'].isna()), 'Cholesterol'] = mean_cholesterol_female

    # Filter out extreme values for RestingBP, MaxHR, and FastingBS
    data = data[data['RestingBP'] != 0]
    data = data[(data['MaxHR'] >= 60) & (data['MaxHR'] <= 202)]
    data = data[(data['FastingBS'] == 0) | (data['FastingBS'] == 1)]
    data = data[data['Sex'].isin(["M", "F"])]

    return data

# Function to plot missing values heatmap
def plot_missing_values(data):
    st.write("### Missing Values Heatmap")
    st.write(
        "This heatmap shows the missing values in the dataset. Each cell represents a missing value (NaN) in the data.")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(data.isnull(), cmap='magma', cbar=False, ax=ax)
    st.pyplot(fig)
    st.markdown("""
    - Box plot showing cholesterol with missing values in the dataset, subsequently filled in using the average value.""")

def plot_descriptive_statistics(data):
    st.write("### Descriptive Statistics")

    # 整体数据的描述性统计
    st.write("#### Overall Descriptive Statistics")
    st.write(data.describe().T)

    # 按心脏病分类的描述性统计
    yes = data[data['HeartDisease'] == 1].describe().T
    no = data[data['HeartDisease'] == 0].describe().T


    # 可视化心脏病和无心脏病的描述性统计
    st.write("#### Visualization of Descriptive Statistics by Heart Disease Classification")
    colors = ['#F93822', '#FDD20E']
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # 心脏病患者
    plt.subplot(1, 2, 1)
    sns.heatmap(yes[['mean']], annot=True, cmap=colors, linewidths=0.4, linecolor='black', cbar=False, fmt='.2f')
    plt.title('Heart Disease')

    # 非心脏病患者
    plt.subplot(1, 2, 2)
    sns.heatmap(no[['mean']], annot=True, cmap=colors, linewidths=0.4, linecolor='black', cbar=False, fmt='.2f')
    plt.title('No Heart Disease')

    fig.tight_layout(pad=2)
    st.pyplot(fig)
    st.markdown("""
            - **Mean values** of all the features for cases of heart diseases and non-heart diseases.""")
# Function to plot the distribution of categorical features

def Classification_of_features(data):
    st.write("### Classification of features")
    col = list(data.columns)
    categorical_features = []
    numerical_features = []
    for i in col:
        if i != 'HeartDisease':  # 排除目标变量 HeartDisease
            if len(data[i].unique()) > 4:
                numerical_features.append(i)
            else:
                categorical_features.append(i)

    st.write('**Categorical Features:**', *categorical_features)
    st.write('**Numerical Features:**', *numerical_features)
    st.markdown("""
    - Here, categorical features are defined if the the attribute has less than 4 unique elements else it is a numerical feature.
    - Typical approach for this division of features can also be based on the datatypes of the elements of the respective attribute.""")

def plot_categorical_features(data):
    st.write("### Distribution of Categorical Features")
    st.write("These bar charts show the distribution of categorical features in the dataset.")
    categorical_features = ['Sex', 'ChestPainType', 'FastingBS','RestingECG', 'ExerciseAngina', 'ST_Slope']
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
    for i, feature in enumerate(categorical_features):
        sns.countplot(x=feature, data=data, palette='magma', edgecolor='black', ax=axes[i // 2, i % 2])
        axes[i // 2, i % 2].set_title(f'{feature} Distribution')
    plt.tight_layout()
    st.pyplot(fig)

# Function to plot the distribution of numerical features
def plot_numerical_features(data):
    st.write("### Distribution of Numerical Features")
    st.write(
        "These histograms show the distribution of numerical features in the dataset.")
    numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
    for i, feature in enumerate(numerical_features):
        sns.histplot(data[feature], kde=True, color='skyblue', ax=axes[i // 2, i % 2])
        axes[i // 2, i % 2].set_title(f'{feature} Distribution')
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("""
    -  Check its distribution pattern (e.g., normal, skewed, etc.)
    -  Numerical characteristics are close to normal distribution. ldpeak's data distribution is rightly skewed.""")

def plot_categorical_feature_distributions(data, categorical_features, colors):

    st.write("### Categorical Features vs Heart Disease")
    st.write("Mapping Categorical Characteristics in Relation to Heart Disease.")
    # 分类特征与心脏病分布的关系图
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
    for i, feature in enumerate(categorical_features):
        plt.subplot(3, 2, i + 1)
        current_ax = sns.countplot(x=feature, data=data, hue="HeartDisease", palette=colors, edgecolor='black')
        for rect in current_ax.patches:
            current_ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 2, rect.get_height(),
                            horizontalalignment='center', fontsize=11)
        title = feature + ' vs Heart Disease'
        plt.legend(['No Heart Disease', 'Heart Disease'])
        plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("""
    - **Male** population has more heart disease patients than no heart disease patients. In the case of **Female** population, heart disease patients are less than no heart disease patients. 
- **ASY** type of chest pain boldly points towards major chances of heart disease.
- **Fasting Blood Sugar** is tricky! Patients diagnosed with Fasting Blood Sugar and no Fasting Blood Sugar have significant heart disease patients. 
- **RestingECG** does not present with a clear cut category that highlights heart disease patients. All the 3 values consist of high number of heart disease patients.
- **Exercise Induced Engina** definitely bumps the probability of being diagnosed with heart diseases.
- With the **ST_Slope** values, **flat** slope displays a very high probability of being diagnosed with heart disease. **Down** also shows the same output but in very few data points. """)
    # 计算每个分类特征的百分比
    st.write("### Categorical Features vs Positive Heart Disease Cases")
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(20, 20))

    # 性别
    plt.subplot(3, 2, 1)
    sex = data[data['HeartDisease'] == 1]['Sex'].value_counts(normalize=True) * 100
    labels = ['Male', 'Female']
    plt.pie(sex, labels=labels, autopct='%1.1f%%', startangle=90, explode=(0.1, 0), colors=colors,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'antialiased': True})
    plt.title('Sex')

    # 胸痛类型
    plt.subplot(3, 2, 2)
    cp = data[data['HeartDisease'] == 1]['ChestPainType'].value_counts(normalize=True) * 100
    labels = ['ASY', 'NAP', 'ATA', 'TA']
    plt.pie(cp, labels=labels, autopct='%1.1f%%', startangle=90, explode=(0, 0.1, 0.1, 0.1),
            wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'antialiased': True})
    plt.title('ChestPainType')

    # 空腹血糖
    plt.subplot(3, 2, 3)
    fbs = data[data['HeartDisease'] == 1]['FastingBS'].value_counts(normalize=True) * 100
    labels = ['FBS < 120 mg/dl', 'FBS > 120 mg/dl']
    plt.pie(fbs, labels=labels, autopct='%1.1f%%', startangle=90, explode=(0.1, 0), colors=colors,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'antialiased': True})
    plt.title('FastingBS')

    # 静息心电图
    plt.subplot(3, 2, 4)
    restecg = data[data['HeartDisease'] == 1]['RestingECG'].value_counts(normalize=True) * 100
    labels = ['Normal', 'ST', 'LVH']
    plt.pie(restecg, labels=labels, autopct='%1.1f%%', startangle=90, explode=(0, 0.1, 0.1),
            wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'antialiased': True})
    plt.title('RestingECG')

    # 运动诱发心绞痛
    plt.subplot(3, 2, 5)
    exang = data[data['HeartDisease'] == 1]['ExerciseAngina'].value_counts(normalize=True) * 100
    labels = ['Angina', 'No Angina']
    plt.pie(exang, labels=labels, autopct='%1.1f%%', startangle=90, explode=(0.1, 0), colors=colors,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'antialiased': True})
    plt.title('ExerciseAngina')

    # ST段斜率
    plt.subplot(3, 2, 6)
    slope = data[data['HeartDisease'] == 1]['ST_Slope'].value_counts(normalize=True) * 100
    labels = ['Flat', 'Up', 'Down']
    plt.pie(slope, labels=labels, autopct='%1.1f%%', startangle=90, explode=(0, 0.1, 0.1),
            wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'antialiased': True})
    plt.title('ST_Slope')

    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("""- Out of all the heart disease patients, a staggering 90% patients are **male**.
- When it comes to the type of chest pain, **ASY** type holds the majority with 77% that lead to heart diseases.
- **Fasting Blood Sugar** level < 120 mg/dl displays high chances of heart diseases.
- For **RestingECG**, **Normal** level accounts for 56% chances of heart diseases than **LVH** and **ST** levels.
- Detection of **Exercise Induced Angina** also points towards heart diseases.
- When it comes to **ST_Slope** readings, **Flat** level holds a massive chunk with 75% that may assist in detecting underlying heart problems. """)

def plot_grouped_numerical_features(data, numerical_features, colors):
    st.write("### Numerical Features vs Target Variable **(HeartDisease)**")
    # 绘制数值特征与心脏病分布的关系图
    fig, ax = plt.subplots(nrows=len(numerical_features), ncols=1, figsize=(15, 30))
    for i, feature in enumerate(numerical_features):
        plt.subplot(len(numerical_features), 1, i + 1)
        sns.countplot(x=feature, data=data, hue="HeartDisease", palette=colors, edgecolor='black')
        title = feature + ' vs Heart Disease'
        plt.legend(['No Heart Disease', 'Heart Disease'])
        plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)
    st.write("To make the data more digestible and enable easier comparisons, scale the continuous variables.")
    # 根据区间划分数值型特征为分类变量
    data['Age_Group'] = pd.cut(data['Age'], bins=[20, 29, 39, 49, 59, 69, 79, 120], labels=["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"])
    data['RestingBP_Group'] = pd.cut(data['RestingBP'], bins=[0, 90, 119, 139, 200], labels=["Low (< 90)", "Normal (90-119)", "High (120-139)", "Hypertension (>=140)"])
    data['Cholesterol_Group'] = pd.cut(data['Cholesterol'], bins=[0, 200, 239, 400], labels=["Low (< 200)", "Normal (200-239)", "High (>= 240)"])
    data['MaxHR_Group'] = pd.cut(data['MaxHR'], bins=[0, 60, 100, 200], labels=["Low (< 60)", "Normal (60-100)", "High (> 100)"])
    data['Oldpeak_Group'] = pd.cut(data['Oldpeak'], bins=[-np.inf, 0, 1, 2, 3, np.inf], labels=["Negative (< 0)", "0 to 1", "1 to 2", "2 to 3", "Greater than 3"])
    st.markdown("""
    - **Age**: Age groups above 50 have a higher prevalence of heart disease.
- **Blood Pressure**: High blood pressure (especially in the 120-139 range) is a significant risk factor for heart disease.
- **Cholesterol**: High cholesterol levels (above 240) have a strong association with heart disease.
- **Heart Rate**: A higher heart rate (> 100) correlates with a higher likelihood of heart disease.
- **Oldpeak**: Negative and moderate ST segment changes are associated with a higher probability of heart disease diagnosis.
    """)
    # 分组后的条形图
    group_numerical_features = ['Age_Group', 'RestingBP_Group', 'Cholesterol_Group', 'MaxHR_Group', 'Oldpeak_Group']
    fig, ax = plt.subplots(nrows=len(group_numerical_features), ncols=1, figsize=(10, 25))
    for i, feature in enumerate(group_numerical_features):
        plt.subplot(len(group_numerical_features), 1, i + 1)
        sns.countplot(x=feature, data=data, hue="HeartDisease", palette=colors, edgecolor='black')
        plt.legend(['No Heart Disease', 'Heart Disease'])
        plt.title(feature + ' vs Heart Disease')
    plt.tight_layout()
    st.pyplot(fig)
# Function to plot correlation heatmap

def plot_numerical_features_vs_sex(data, numerical_features, colors):
    st.write("### Numerical features vs Categorical features w.r.t Target variable(HeartDisease) :")
    st.write("#### Sex vs Numerical Features ")

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        sns.stripplot(x='Sex', y=numerical_features[i], data=data, hue='HeartDisease', palette=colors)
        plt.legend(['No Heart Disease', 'Heart Disease'])
        title = numerical_features[i] + ' vs Sex'
        plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    for i in [-1, -2]:
        plt.subplot(1, 2, -i)
        sns.stripplot(x='Sex', y=numerical_features[i], data=data, hue='HeartDisease', palette=colors)
        plt.legend(['No Heart Disease', 'Heart Disease'])
        title = numerical_features[i] + ' vs Sex'
        plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("""
    - **Male** population displays heart diseases at near about all the values of the numerical features. Above the age of 50, positive old peak values and maximum heart rate below 140, heart diseases in male population become dense.
- **Female** population data points are very less as compared to **male** population data points. Hence, we cannot point to specific ranges or values that display cases of heart diseases. """)
def plot_chest_pain_type_vs_numerical_features(data, numerical_features, colors):

    st.write("#### ChestPainType vs Numerical Features")
    st.markdown("""
    - ASY type of chest pain dominates other types of chest pain in all the numerical features by a lot.
    """)

    # 前三个数值特征
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        sns.stripplot(x='ChestPainType', y=numerical_features[i], data=data, hue='HeartDisease', palette=colors)
        plt.legend(['No Heart Disease', 'Heart Disease'])
        title = numerical_features[i] + ' vs ChestPainType'
        plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)

    # 后两个数值特征
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    for i in [-1, -2]:
        plt.subplot(1, 2, -i)
        sns.stripplot(x='ChestPainType', y=numerical_features[i], data=data, hue='HeartDisease', palette=colors)
        plt.legend(['No Heart Disease', 'Heart Disease'])
        title = numerical_features[i] + ' vs ChestPainType'
        plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)
def plot_fasting_bs_vs_numerical_features(data, numerical_features, colors):
    st.write("#### FastingBS vs Numerical Features")
    st.markdown("""
    - Above the age 50, heart diseases are found throughout the data irrespective of the patient being diagnosed with Fasting Blood Sugar or not.
    - Fasting Blood Sugar with Resting BP over 100 has displayed more cases of heart diseases than patients with no fasting blood sugar.
    - Cholesterol with Fasting Blood Sugar does not seem to have an effect in understanding reason behind heart diseases.
    - Patients that have not been found positive with Fasting Blood Sugar but have maximum heart rate below 130 are more prone to heart diseases.
    """)

    # 前三个数值特征
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        sns.stripplot(x='FastingBS', y=numerical_features[i], data=data, hue='HeartDisease', palette=colors)
        plt.legend(['No Heart Disease', 'Heart Disease'])
        title = numerical_features[i] + ' vs FastingBS'
        plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)

    # 后两个数值特征
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    for i in [-1, -2]:
        plt.subplot(1, 2, -i)
        sns.stripplot(x='FastingBS', y=numerical_features[i], data=data, hue='HeartDisease', palette=colors)
        plt.legend(['No Heart Disease', 'Heart Disease'])
        title = numerical_features[i] + ' vs FastingBS'
        plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)
def plot_resting_ecg_vs_numerical_features(data, numerical_features, colors):
    st.write("#### RestingECG vs Numerical Features")
    st.markdown("""
    - Heart diseases with RestingECG values of Normal, ST and LVH are detected starting from 30, 40 & 40 respectively. Patients above the age of 50 are more prone than any other ages irrespective of RestingECG values.
    - Heart diseases are found consistently throughout any values of RestingBP and RestingECG.
    - Cholesterol values between 200 - 300 coupled with ST value of RestingECG display a patch of patients suffering from heart diseases.
    - For maximum Heart Rate values, heart diseases are detected in dense below 140 points and Normal RestingECG. ST & LVH throughout the maximum heart rate values display heart disease cases.
    """)

    # 前三个数值特征
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        sns.stripplot(x='RestingECG', y=numerical_features[i], data=data, hue='HeartDisease', palette=colors)
        plt.legend(['No Heart Disease', 'Heart Disease'])
        title = numerical_features[i] + ' vs RestingECG'
        plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)

    # 后两个数值特征
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    for i in [-1, -2]:
        plt.subplot(1, 2, -i)
        sns.stripplot(x='RestingECG', y=numerical_features[i], data=data, hue='HeartDisease', palette=colors)
        plt.legend(['No Heart Disease', 'Heart Disease'])
        title = numerical_features[i] + ' vs RestingECG'
        plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)
def plot_exercise_angina_vs_numerical_features(data, numerical_features, colors):
    st.write("#### ExerciseAngina vs Numerical Features")
    st.markdown("""
    - A crystal clear observation can be made about the relationship between heart disease case and Exercise induced Angina. A positive correlation between the 2 features can be concluded throughout all the numerical features.
    """)

    # 前三个数值特征
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        sns.stripplot(x='ExerciseAngina', y=numerical_features[i], data=data, hue='HeartDisease', palette=colors)
        plt.legend(['No Heart Disease', 'Heart Disease'])
        title = numerical_features[i] + ' vs ExerciseAngina'
        plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)

    # 后两个数值特征
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    for i in [-1, -2]:
        plt.subplot(1, 2, -i)
        sns.stripplot(x='ExerciseAngina', y=numerical_features[i], data=data, hue='HeartDisease', palette=colors)
        plt.legend(['No Heart Disease', 'Heart Disease'])
        title = numerical_features[i] + ' vs ExerciseAngina'
        plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)
def plot_st_slope_vs_numerical_features(data, numerical_features, colors):

    st.write("#### ST_Slope vs Numerical Features")
    st.markdown("""
    - Another crystal clear positive observation can be made about the positive correlation between ST_Slope value and Heart Disease cases.
    - Flat, Down and Up in that order display high, middle and low probability of being diagnosed with heart diseases respectively.
    """)

    # 前三个数值特征
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        sns.stripplot(x='ST_Slope', y=numerical_features[i], data=data, hue='HeartDisease', palette=colors)
        plt.legend(['No Heart Disease', 'Heart Disease'])
        title = numerical_features[i] + ' vs ST_Slope'
        plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)

    # 后两个数值特征
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    for i in [-1, -2]:
        plt.subplot(1, 2, -i)
        sns.stripplot(x='ST_Slope', y=numerical_features[i], data=data, hue='HeartDisease', palette=colors)
        plt.legend(['No Heart Disease', 'Heart Disease'])
        title = numerical_features[i] + ' vs ST_Slope'
        plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)
def plot_numerical_features_vs_numerical_features(data, numerical_features, colors):
    st.write("#### Numerical Features vs Numerical Features w.r.t Target Variable (HeartDisease)")
    st.markdown("""
    - For age 50+, RestingBP between 100 - 175, Cholesterol level of 200 - 300, Max Heart Rate below 160 and positive oldpeak values display high cases of heart disease.
    - For RestingBP values 100 - 175, highlights too many heart disease patients for all the features.
    - Cholesterol values 200 - 300 dominate the heart disease cases.
    - Similarly, Max Heart Rate values below 140 have a high probability of being diagnosed with heart diseases.
    """)

    a = 0
    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(15, 25))
    for i in range(len(numerical_features)):
        for j in range(len(numerical_features)):
            if i != j and j > i:
                a += 1
                plt.subplot(5, 2, a)
                sns.scatterplot(x=numerical_features[i], y=numerical_features[j], data=data, hue='HeartDisease',
                                palette=colors, edgecolor='black')
                plt.legend(['No Heart Disease', 'Heart Disease'])
                title = numerical_features[i] + ' vs ' + numerical_features[j]
                plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    ### Summary of EDA
####  Values of features for positive cases of heart disease:

- **Categorical Features (Order) :**
    - Sex : Male > Female
    - ChestPainType : ASY > NAP > ATA > TA
    - FastingBS : ( FBS < 120 mg/dl ) > ( FBS > 120 mg/dl)
    - RestingECG : Normal > ST > LVH
    - ExerciseAngina : Angina > No Angina
    - ST_Slope : Flat > Up > Down
 
- **Numerical Features (Range) :**
    - Age : 50+
    - RestingBP : 95 - 170 
    - Cholesterol : 160 - 340
    - MaxHR : 70 - 180
    - Oldpeak : 0 - 4
    """)
def plot_correlation_matrix(data, numerical_features, colors):
    """
    绘制相关系数矩阵的热力图，并进行标准化和归一化处理。
    """
    st.write("#### Correlation Matrix on Heart Disease")

    # 创建副本以避免修改原始数据
    df1 = data.copy(deep=True)

    # 对分类特征进行编码
    le = LabelEncoder()
    categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    for feature in categorical_features:
        df1[feature] = le.fit_transform(df1[feature])

    # 初始化标准化和归一化对象
    ss = StandardScaler()  # 标准化

    # 对数值特征进行标准化处理
    for feature in numerical_features:
        df1[feature] = ss.fit_transform(df1[[feature]])

    # 计算相关系数矩阵，只对数值型列进行计算
    corr_matrix = df1.corr()

    # 绘制热力图
    fig = plt.figure(figsize=(25, 15))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5, ax=fig.add_subplot(111))
    plt.title('Correlation Matrix')
    st.pyplot(fig)

    # 绘制热力图
    corr = df1.corrwith(df1['HeartDisease']).sort_values(ascending=False).to_frame()
    corr.columns = ['Correlations']
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, cmap=colors, linewidths=0.4, linecolor='black', annot_kws={"size": 10},
                ax=ax)
    plt.title('Correlation w.r.t HeartDisease')
    st.pyplot(fig)
def plot_feature_selection_categorical(data, categorical_features, colors):
    st.write("#### Feature Selection for Categorical Features (Chi Squared Test)")
    st.markdown("""
    - **Exercise-Induced Angina (ExerciseAngina)**: Has the highest Chi Squared score (114.40), indicating a strong association with heart disease, making it an important feature for predicting heart disease.
    - **Chest Pain Type (ChestPainType)** and **Fasting Blood Sugar (FastingBS)**: These features have Chi Squared scores of 57.15 and 21.83 respectively, showing a relatively strong association with heart disease.
    - **Sex** and **Resting ECG (RestingECG)**: These features have lower Chi Squared scores of 4.57 and 0.12, indicating a weaker association with heart disease. Particularly, Resting ECG has the lowest score, suggesting it might not be a significant predictor of heart disease.
    """)
    # 对分类特征进行编码
    le = LabelEncoder()
    for feature in categorical_features[:-1]:  # Exclude the target variable
        data[feature] = le.fit_transform(data[feature])

    # 准备特征和目标变量
    features = data.loc[:, categorical_features[:-1]]
    target = data.loc[:, categorical_features[-1]]

    # 特征选择
    best_features = SelectKBest(score_func=chi2, k='all')
    fit = best_features.fit(features, target)

    feature_scores = pd.DataFrame(data=fit.scores_, index=list(features.columns), columns=['Chi Squared Score'])

    # 创建一个绘图对象，确保fig是figure对象
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(feature_scores.sort_values(ascending=False, by='Chi Squared Score'), annot=True, cmap=colors,
                linewidths=0.4,
                linecolor='black', fmt='.2f', ax=ax)  # 确保传递ax参数
    plt.title('Selection of Categorical Features')
    plt.tight_layout()

    # 传递fig到Streamlit
    st.pyplot(fig)  # 使用fig作为参数，传递给st.pyplot()
def plot_feature_selection_numerical(data, numerical_features, colors):

    st.write("#### Feature Selection for Numerical Features (ANOVA Test)")
    st.markdown("""
    - **Old Peak (Oldpeak)** and **Maximum Heart Rate (MaxHR)**: These features have the highest ANOVA scores of 178.09 and 175.75, indicating a strong association with heart disease, making them important features for predicting heart disease.
    - **Age** and **Cholesterol**: These features have ANOVA scores of 79.06 and 7.16, showing a relatively strong association with heart disease.
    - **Resting Blood Pressure (RestingBP)**: Has the lowest score (12.92), indicating the weakest association with heart disease, suggesting it may have minimal impact on predicting heart disease.""")

    # 准备特征和目标变量
    features = data.loc[:, numerical_features]  # 删除多余的 categorical_features 参数
    target = data.loc[:, 'HeartDisease']  # 假设 'HeartDisease' 是目标变量

    # 特征选择
    best_features = SelectKBest(score_func=f_classif, k='all')
    fit = best_features.fit(features, target)

    feature_scores = pd.DataFrame(data=fit.scores_, index=list(features.columns), columns=['ANOVA Score'])

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(feature_scores.sort_values(ascending=False, by='ANOVA Score'), annot=True, cmap=colors, linewidths=0.4,
                linecolor='black', fmt='.2f', ax=ax)  # 确保传递ax参数
    plt.title('Selection of Numerical Features')
    plt.tight_layout()

    # 传递fig到Streamlit
    st.pyplot(fig)  # 使用fig作为参数，传递给st.pyplot()

    st.markdown(""" 
    ### Summary
    - **Exercise-Induced Angina**, **Chest Pain Type**, **Fasting Blood Sugar**, **Old Peak**, and **Maximum Heart Rate** are important features for predicting heart disease.
    - Sex and Resting ECG have weaker associations with heart disease, with Resting ECG being the least significant, potentially being excluded from the modeling process.
    - Resting Blood Pressure has the weakest association, and may be considered for exclusion in modeling.
    - Finally, We will leave out Resting ECG, RestingBP and Cholesterol from the modeling part and take the remaining features.""")

    st.markdown("""
    ### Model Training and Evaluation After Feature Selection

    After testing and evaluating the models, and excluding features with weaker associations, we observed an improvement in the overall accuracy and other parameters of the model. This indicates that focusing on the most relevant features has positively impacted the model's predictive capabilities.
    
    By excluding features with less impact on heart disease prediction, the model becomes more efficient and effective in identifying patients with heart disease.
    """)

# Main function to run all analysis and visualizations
def run_data_analysis():
    matplotlib.rcParams['figure.max_open_warning'] = 50
    # Visualizations
    colors = ['#F93822', '#FDD20E']
    data = load_and_process_data()
    plot_descriptive_statistics(data)
    Classification_of_features(data)
    categorical_features = ['Sex', 'ChestPainType', 'FastingBS','RestingECG', 'ExerciseAngina', 'ST_Slope']
    numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    plot_categorical_features(data)
    plot_numerical_features(data)
    plot_categorical_feature_distributions(data, categorical_features, colors)
    plot_grouped_numerical_features(data, numerical_features, colors)
    plot_numerical_features_vs_sex(data, numerical_features, colors)
    plot_chest_pain_type_vs_numerical_features(data, numerical_features, colors)
    plot_fasting_bs_vs_numerical_features(data, numerical_features, colors)
    plot_resting_ecg_vs_numerical_features(data, numerical_features, colors)
    plot_exercise_angina_vs_numerical_features(data, numerical_features, colors)
    plot_st_slope_vs_numerical_features(data, numerical_features, colors)
    plot_numerical_features_vs_numerical_features(data, numerical_features, colors)
    plot_correlation_matrix(data, numerical_features, colors)
    plot_feature_selection_categorical(data, categorical_features, colors)
    plot_feature_selection_numerical(data, numerical_features, colors)

