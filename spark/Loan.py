import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import graphviz as graphviz
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy import randint

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

data = pd.read_csv('data.csv')
# Perform the mapping operations
data['default'] = data['default'].map({'no': 0, 'yes': 1, 'unknown': 0})
data['y'] = data['y'].map({'no': 0, 'yes': 1})
# Display the first few rows of the DataFrame to verify the mappings
data.head()

# Split the data into features (X) and target (y)
X = data.drop('y', axis=1)
y = data['y']
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Identify categorical columns
categorical_features = X.select_dtypes(include=['object']).columns
# Create a column transformer with one-hot encoder for categorical features
preprocessor = ColumnTransformer(
transformers=[
('cat', OneHotEncoder(), categorical_features)
],remainder='passthrough' # Keep the remaining columns as they are
)
# Create a pipeline with preprocessing and model training
model_pipeline = Pipeline(steps=[
('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])

34
# Define the hyperparameter distribution
param_dist = {
'classifier__n_estimators': randint(50, 500),
'classifier__max_depth': randint(1, 20)
}
# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(model_pipeline, param_distributions=param_dist, n_iter=5, cv=5, random_state=42, n_jobs=-1)
# Fit the random search object to the data
rand_search.fit(X_train, y_train)
# Create a variable for the best model
best_rf = rand_search.best_estimator_
# Print the best hyperparameters
print('Best hyperparameters:', rand_search.best_params_)
# Predict and evaluate using the best model
y_pred = best_rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Save the model to a file
joblib.dump(rand_search, 'Random_Forest.sav')



#Matplotlib dalam streamlit (st.pyplot-histogram)

# rand= np.random.normal(1,2, size=20)
# fig, ax = plt.subplots()
# ax.hist(rand, bins= 15)
# st.pyplot(fig)


#diagram garis(st.line_chart())
#df = pd.DataFrame(np.random.randn(10,2), columns = ['x', 'y'])
# st.line_chart(df)-> diagram garis
# st.bar_chart(df)->  bar charts
# st.area_chart(df)

# graph chart 
# st.graphviz_chart(''' digraph {
# Tomex -> Arnold -> Joy -> Jan ->Ken
# }
# ''')

# df2 = pd.DataFrame(np.random.randn(500, 2)/[50,50]+ [47.76, -122.4],
#                    columns=['lat','lon'])

# st.map(df2)






