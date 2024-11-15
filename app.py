from flask import Flask, request, render_template
import pandas as pd
from sklearn.cluster import KMeans

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("data.csv")

# Select numeric columns for clustering (excluding 'Country')
numeric_columns = ['AveragScore', 'SafetySecurity', 'PersonelFreedom', 'Governance',
                   'SocialCapital', 'InvestmentEnvironment', 'EnterpriseConditions',
                   'MarketAccessInfrastructure', 'EconomicQuality', 'LivingConditions',
                   'Health', 'Education', 'NaturalEnvironment']

# Train a KMeans model on the numeric columns
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data[numeric_columns])

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
    
    # Predict the cluster for the input data
    cluster = kmeans.predict([input_data])[0]
    
    # Get the corresponding category for the predicted cluster
    category = cluster_categories[cluster]
    
    # Render the result on a new HTML page
    return render_template('result.html', country=country, cluster=cluster, category=category)

if __name__ == '__main__':
    app.run(debug=True)
