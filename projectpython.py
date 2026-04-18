# ============================================================
# CREDIT CARD FRAUD DETECTION PROJECT (FINAL VERSION)
# ============================================================

print("Project Started...")

# ============================================================
# 1. NUMPY (ARRAY OPERATIONS)
# ============================================================
import numpy as np

arr = np.array([10, 20, 30, 40])
print("\nNumPy Array:", arr)
print("Mean:", np.mean(arr))
print("Standard Deviation:", np.std(arr))
print("Sum:", np.sum(arr))


# ============================================================
# 2. PANDAS (DATA HANDLING)
# ============================================================
import pandas as pd

df = pd.read_csv("creditcard.csv")

print("\nFirst 5 rows:\n", df.head())
print("\nDataset Info:")
df.info()

print("\nSeries Example:\n", df['Amount'].head())


# ============================================================
# DATA CLEANING & PREPARATION
# ============================================================

print("\nMissing Values:\n", df.isnull().sum())

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Feature Engineering
df['Hour'] = df['Time'] // 3600


# ============================================================
# 3. DATA VISUALIZATION
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
sns.set_context("talk")

# Fraud vs Normal
plt.figure()
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Normal Transactions")
plt.show()

# Amount Distribution
plt.figure()
plt.hist(df['Amount'], bins=50)
plt.title("Transaction Amount Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\nSummary Statistics:\n", df.describe())
print("\nCorrelation:\n", df.corr())
print("\nCovariance:\n", df.cov())

# Class Imbalance
print("\nClass Distribution (%):\n", df['Class'].value_counts(normalize=True)*100)

# Boxplot (Outliers)
plt.figure()
sns.boxplot(x='Class', y='Amount', data=df)
plt.title("Outlier Detection")
plt.show()

# Z-score Outliers
from scipy import stats

z_scores = np.abs(stats.zscore(df['Amount']))
outliers = df[z_scores > 3]
print("\nNumber of Outliers:", len(outliers))

# Pairplot
sns.pairplot(df[['Amount','Time','Class']].sample(500))
plt.show()


# ============================================================
# 5. STATISTICAL ANALYSIS
# ============================================================

# Shapiro-Wilk Test
sample_data = df['Amount'].sample(500)
stat, p = stats.shapiro(sample_data)
print("\nShapiro-Wilk p-value:", p)

# T-Test
t_stat, p_val = stats.ttest_ind(
    df[df['Class']==0]['Amount'].sample(500),
    df[df['Class']==1]['Amount'].sample(100)
)
print("T-Test p-value:", p_val)

# Chi-Square Test
from scipy.stats import chi2_contingency

contingency = pd.crosstab(df['Class'], df['Hour'])
chi2, p, dof, exp = chi2_contingency(contingency)
print("Chi-Square p-value:", p)

# Z-Test
sample = df['Amount'].sample(500)
sample_mean = np.mean(sample)
population_mean = df['Amount'].mean()
std = np.std(sample)

z_score = (sample_mean - population_mean) / (std / np.sqrt(len(sample)))
print("Z-Test Value:", z_score)

# Confidence Interval
confidence = 0.95
n = len(sample)
mean = np.mean(sample)
se = stats.sem(sample)
h = se * stats.t.ppf((1 + confidence) / 2, n-1)

print("95% Confidence Interval:", (mean - h, mean + h))


# ============================================================
# VIF (MULTICOLLINEARITY)
# ============================================================

from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = df.drop(['Class'], axis=1).iloc[:, :10]
X_vif = X_vif.replace([np.inf, -np.inf], np.nan).dropna()

vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(len(X_vif.columns))
]

print("\nVIF:\n", vif_data)


# ============================================================
# 6. PROBABILITY DISTRIBUTIONS
# ============================================================

# Normal Distribution
data = df['Amount']
mean = np.mean(data)
std = np.std(data)

x = np.linspace(mean-3*std, mean+3*std, 100)
plt.plot(x, stats.norm.pdf(x, mean, std))
plt.title("Normal Distribution")
plt.show()

# Binomial
binomial = np.random.binomial(10, 0.5, 1000)
sns.histplot(binomial)
plt.title("Binomial Distribution")
plt.show()

# Poisson
poisson = np.random.poisson(2, 1000)
sns.histplot(poisson)
plt.title("Poisson Distribution")
plt.show()

# Uniform
uniform = np.random.uniform(0, 1, 1000)
sns.histplot(uniform)
plt.title("Uniform Distribution")
plt.show()


# ============================================================
# 7. MACHINE LEARNING (CRISP-DM)
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature Scaling applied using StandardScaler")


# ============================================================
# MODELS
# ============================================================

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Logistic Regression
log_model = LogisticRegression(max_iter=2000, class_weight='balanced')
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)

print("\nLogistic Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt.predict(X_test)))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print("Random Forest Accuracy:", accuracy_score(y_test, rf.predict(X_test)))

# Linear Regression (CRISP-DM)
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
pred = lin_model.predict(X_test)

print("\nLinear Regression Output:", pred[:5])


# ============================================================
# CRISP-DM
# ============================================================

print("\nCRISP-DM Framework:")
print("1. Business Understanding → Fraud detection")
print("2. Data Understanding → Dataset exploration")
print("3. Data Preparation → Cleaning + feature engineering")
print("4. Modeling → ML algorithms")
print("5. Evaluation → Accuracy, confusion matrix, F1-score")
print("6. Deployment → Conceptual")


print("\nProject Executed Successfully!")
