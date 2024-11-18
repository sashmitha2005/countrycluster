from flask import Flask, request, render_template
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("data.csv")

# Select numeric columns for clustering (excluding 'Country')
numeric_columns = ['AveragScore', 'SafetySecurity', 'PersonelFreedom', 'Governance',
                   'SocialCapital', 'InvestmentEnvironment', 'EnterpriseConditions',
                   'MarketAccessInfrastructure', 'EconomicQuality', 'LivingConditions',
                   'Health', 'Education', 'NaturalEnvironment']

# Define a pipeline with scaling and clustering
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(random_state=42))
])

# Define a parameter grid for GridSearchCV
param_grid = {
    'kmeans__n_clusters': [2, 3, 4, 5, 6]
}

# Use GridSearchCV to find the best number of clusters
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(data[numeric_columns])

# Get the best estimator from GridSearchCV
best_model = grid_search.best_estimator_

# Define a dictionary to map cluster number to category
cluster_categories = {
    0: 'Underdeveloped',
    1: 'Developing',
    2: 'Developed'
}

@app.route('/')
def index():
    # Render the template with numeric columns and a country field
    return render_template('index.html', columns=numeric_columns)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the country and numeric data from the form
    country = request.form['Country']  # Capturing the country name
    input_data = [float(request.form[col]) for col in numeric_columns]  # Capturing the numeric input data
    
    # Predict the cluster for the input data using the best model
    cluster = best_model.predict([input_data])[0]
    
    # Get the corresponding category for the predicted cluster
    category = cluster_categories.get(cluster, 'Unknown')
    
    # Render the result on a new HTML page
    return render_template('result.html', country=country, cluster=cluster, category=category)

if __name__ == '__main__':
    app.run(debug=True)
