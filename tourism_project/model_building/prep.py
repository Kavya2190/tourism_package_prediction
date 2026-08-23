import pandas as pd
from sklearn.model_selection import train_test_split

#Read the data
df = pd.read_csv("tourism_project/data/tourism.csv")

# Data Cleaning : Removal of unnecessary columns
df.drop(columns=["CustomerID"], inplace=True)

# Data Cleaning : Updating the Gender and Marital Status column values as identified earlier
df["Gender"] = df["Gender"].replace("Fe Male", "Female")
df["MaritalStatus"] = df["MaritalStatus"].replace("Unmarried", "Single")

 
# The training pipeline one-hot-encodes the categorical data, and the Streamlit app also sends data to match the same.
# Defining X and Y
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# stratify=y keeps the (imbalanced) failure ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save the splitted X and Y train and test datasets to be used in the next steps.
Xtrain.to_csv("tourism_project/deployment/Xtrain.csv", index=False)
Xtest.to_csv("tourism_project/deployment/Xtest.csv", index=False)
ytrain.to_csv("tourism_project/deployment/ytrain.csv", index=False)
ytest.to_csv("tourism_project/deployment/ytest.csv", index=False)

# Prints the status of the split and values of corresponding categorical values.
print("Data prepared: train/test splits written.")
print("Categorical types and their values are :" )
print("TypeofContact field will have values that could be chosen from:", sorted(X["TypeofContact"].unique()))
print("Occupation field will have values that could be chosen from:", sorted(X["Occupation"].unique()))
print("Gender field will have values that could be chosen from:", sorted(X["Gender"].unique()))
print("ProductPitched field will have values that could be chosen from:", sorted(X["ProductPitched"].unique()))
print("MaritalStatus field will have values that could be chosen from:", sorted(X["MaritalStatus"].unique()))
print("Designation field will have values that could be chosen from:", sorted(X["Designation"].unique()))
