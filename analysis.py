import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# dataset
df = pd.read_csv("data.csv")

# ---------------- TASK 1 ----------------
print("First 10 rows:")
print(df.head(10))

print("\nData types:")
print(df.dtypes)

print("\nMissing values:\n")
print(df.isnull().sum())

# ---------------- TASK 2 ----------------

# Remove duplicate rows
df = df.drop_duplicates()

# Handle missing values
df = df.fillna(method='ffill')

# Standardize column names
df.columns = df.columns.str.lower().str.replace(" ", "_")

# Display cleaned data
print("\nCleaned Data Preview:")
print(df.head())

# Check again for missing values
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# ---------------- TASK 3 ----------------
# Summary statistics
print("\nMean:\n", df.mean(numeric_only=True))
print("\nMedian:\n", df.median(numeric_only=True))
print("\nMode:\n", df.mode(numeric_only=True).iloc[0])
print("\nDescribe:\n", df.describe())

# Histogram
df.hist(figsize=(10,8))
plt.suptitle("Histograms of Numerical Columns")
plt.show()

# Box plot
num_cols = df.select_dtypes(include=['int64','float64']).columns

for col in num_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Box Plot of {col}")
    plt.show()

# Churn analysis
if 'churn' in df.columns:
    sns.countplot(x='churn', data=df)
    plt.title("Churn vs Non-Churn")
    plt.show()

    print("\nChurn Percentage:\n", df['churn'].value_counts(normalize=True)*100)

# ---------------- TASK 4 ----------------
# Create tenure groups
def tenure_group(x):
    if x <= 12:
        return "0-12 months"
    elif x <= 36:
        return "13-36 months"
    else:
        return "37+ months"

df['tenure_group'] = df['tenure'].apply(tenure_group)

# Pie Chart
tenure_counts = df['tenure_group'].value_counts()

plt.figure()
plt.pie(tenure_counts, labels=tenure_counts.index, autopct='%1.1f%%')
plt.title("Customer Distribution by Tenure")
plt.show()

# Bar Chart (Average Monthly Charges)
avg_charges = df.groupby('tenure_group')['monthlycharges'].mean()

plt.figure()
bars = plt.bar(avg_charges.index, avg_charges.values)

# Add annotations
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval,2), ha='center')

plt.title("Average Monthly Charges by Tenure Group")
plt.xlabel("Tenure Group")
plt.ylabel("Average Monthly Charges")
plt.show()

# Interactive Plot (Plotly)
fig = px.bar(
    avg_charges,
    x=avg_charges.index,
    y=avg_charges.values,
    title="Interactive Average Monthly Charges by Tenure Group"
)

fig.show()

# ---------------- TASK 5 ----------------
# Churn by gender
sns.countplot(x='gender', hue='churn', data=df)
plt.title("Churn by Gender")
plt.show()

# Churn by contract
sns.countplot(x='contract', hue='churn', data=df)
plt.title("Churn by Contract")
plt.show()