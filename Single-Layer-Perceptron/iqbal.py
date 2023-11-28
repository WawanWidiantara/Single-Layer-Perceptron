# %%
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

# %%
dt = pd.read_csv(
    "/Volumes/Programming/coding/Artificial-Neural-Network/Untitled/Single-Layer-Perceptron/data/test_data_cl.csv",
    header=None,
    names=list(range(12)),
)

gdt = pd.read_csv(
    "/Volumes/Programming/coding/Artificial-Neural-Network/Untitled/Single-Layer-Perceptron/data/test_data_GroundTruth_cl.csv"
)
dt = dt.head(331)
dt

# %%
col_name = dt.iloc[0, :11].values
col_name = np.insert(col_name, 3, "FirstName")

dt.columns = col_name
dt = dt.drop(0).reset_index(drop=True)
dt

# %%
dt["Name"] = dt["FirstName"] + " " + dt["Name"]
dt = dt.drop("FirstName", axis=1)
dt

# %%
print(dt["Sex"].unique().tolist())
true_sex_val = ["male", "female"]
count = 0
for val in dt["Sex"].values:
    if val not in true_sex_val:
        dt.loc[count, "Sex"] = np.random.choice(true_sex_val)
    count += 1
print(dt["Sex"].unique().tolist())

# %%
count = 0
for val in dt["Fare"].values:
    try:
        float(val)
    except ValueError:
        dt.loc[count, "Fare"] = 0
    count += 1

# %%
count = 0
for val in dt["Parch"].values:
    if len(val) > 1:
        dt.loc[count, "Parch"] = 0
    count += 1

# %%
dt.info()

# %%
dt = dt.astype(
    {"PassengerId": "int64", "Pclass": "int64", "SibSp": "int64", "Parch": "int64"}
)
dt = dt.astype(
    {
        "Name": "string",
        "Sex": "string",
        "Ticket": "string",
        "Cabin": "string",
        "Embarked": "string",
    }
)
dt = dt.astype({"Age": "float64", "Fare": "float64"})
dt.info()

# %%
del dt["Name"]
del dt["Ticket"]
del dt["Cabin"]

# %%
missing = pd.DataFrame(
    {"total": dt.isnull().sum(), "percent": dt.isnull().sum() / dt.shape[0] * 100}
)

missing

# %%
dt["Embarked"] = dt["Embarked"].fillna(
    value=np.random.choice(dt["Embarked"].unique().tolist())
)
dt["Fare"] = dt["Fare"].fillna(value=0)
dt.isna().sum()

# %%
lbenc = LabelEncoder()

for col in dt.columns.values:
    if dt[col].dtype == "object":
        dt[col] = lbenc.fit_transform(dt[col])

dt.head()

# # %%
merged_test_data = pd.merge(dt, gdt, on="PassengerId", how="left")

# %%
merged_test_data

# %%
test = merged_test_data.copy()
X_test = test.iloc[:, 1:8]
y_test = merged_test_data.iloc[:, 8]
