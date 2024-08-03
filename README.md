# MLDataPrepocesspy
This project demonstrates how to process data in a CSV file
DOCUMENTATION
numpy (imported as np): A library for numerical operations.
matplotlib.pyplot (imported as plt): A library for creating visualizations.
pandas (imported as pd): A library for data manipulation and analysis.
dataset = pd.read_csv('Data.csv'): Reads the CSV file 'Data.csv' into a DataFrame.
X = dataset.iloc[:, :-1].values: Selects all columns except the last one as features and converts them to a NumPy array.
y = dataset.iloc[:, -1].values: Selects the last column as the target variable and converts it to a NumPy array.
print(X): Prints the feature array.
print(y): Prints the target array
from sklearn.impute import SimpleImputer: Imports the SimpleImputer class from sklearn.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean'): Creates an imputer object to replace missing values with the mean.
imputer.fit(X[:, 1:3]): Fits the imputer on the columns 1 and 2 (second and third columns).
X[:, 1:3] = imputer.transform(X[:, 1:3]): Transforms the data, replacing missing values with the mean of the column.
print(X): Prints the modified feature array.
from sklearn.compose import ColumnTransformer: Imports the ColumnTransformer class from sklearn.
from sklearn.preprocessing import OneHotEncoder: Imports the OneHotEncoder class from sklearn.
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough'): Creates a column transformer that applies one-hot encoding to the first column and leaves other columns unchanged.
X = np.array(ct.fit_transform(X)): Applies the transformation and converts the result to a NumPy array.
print(X): Prints the transformed feature array.
from sklearn.preprocessing import LabelEncoder: Imports the LabelEncoder class from sklearn.
le = LabelEncoder(): Creates a label encoder object.
y = le.fit_transform(y): Applies label encoding to the target variable.
print(y): Prints the encoded target array.
from sklearn.model_selection import train_test_split: Imports the train_test_split function from sklearn.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1): Splits the dataset into training and test sets with 20% of the data for testing and a random state for reproducibility.
print(X_train): Prints the training features.
print(X_test): Prints the test features.
print(y_train): Prints the training labels.
print(y_test): Prints the test labels.
from sklearn.preprocessing import StandardScaler: Imports the StandardScaler class from sklearn.
sc = StandardScaler(): Creates a standard scaler object.
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:]): Applies feature scaling to the training set (excluding the first three columns) and fits the scaler.
X_test[:, 3:] = sc.transform(X_test[:, 3:]): Applies the same transformation to the test set (excluding the first three columns).
print(X_train): Prints the scaled training features.
print(X_test): Prints the scaled test features.
