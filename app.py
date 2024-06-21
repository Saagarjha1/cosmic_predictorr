import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the dataset from the CSV file
file_path = '/workspaces/codespaces-flask/Real_time_dataset.csv'  # Update with your dataset file path
df = pd.read_csv(file_path)

# Drop rows with missing values in the target variable 'bt'
df.dropna(subset=['bt'], inplace=True)

# Separate features and target variable
target_column_name = 'bt'
X = df[['bx_gse', 'by_gse', 'bz_gse', 'theta_gse', 'phi_gse',
        'bx_gsm', 'by_gsm', 'bz_gsm', 'theta_gsm', 'phi_gsm',
        'density', 'speed', 'temperature']]
y = df[target_column_name]

# Split the dataset into training and testing sets
X_train, X_test_full, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select only 8 features for testing
X_test = X_test_full[['bx_gsm', 'by_gsm', 'bz_gsm', 'theta_gsm', 'phi_gsm',
                      'density', 'speed', 'temperature']]

# Impute missing values in features
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

# Initialize regressors
hist_gb_regressor = HistGradientBoostingRegressor(max_iter=1000, max_depth=3, learning_rate=0.1)
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the regressors
hist_gb_regressor.fit(X_train_imputed, y_train)
rf_regressor.fit(X_train_imputed, y_train)

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the form data
        form_data = request.form.to_dict()
        bx_gsm = float(form_data['bx_gsm'])
        by_gsm = float(form_data['by_gsm'])
        bz_gsm = float(form_data['bz_gsm'])
        theta_gsm = float(form_data['theta_gsm'])
        phi_gsm = float(form_data['phi_gsm'])
        density = float(form_data['density'])
        speed = float(form_data['speed'])
        temperature = float(form_data['temperature'])
        
        # Prepare input for prediction
        input_data = pd.DataFrame({
            'bx_gse': 0, 'by_gse': 0, 'bz_gse': 0, 'theta_gse': 0, 'phi_gse': 0,  # Provide dummy values
            'bx_gsm': [bx_gsm], 'by_gsm': [by_gsm], 'bz_gsm': [bz_gsm], 'theta_gsm': [theta_gsm], 'phi_gsm': [phi_gsm],
            'density': [density], 'speed': [speed], 'temperature': [temperature]
        })
        
        # Impute missing values in features
        input_data_imputed = imputer.transform(input_data)

        # Make predictions
        gb_prediction = hist_gb_regressor.predict(input_data_imputed)[0]
        rf_prediction = rf_regressor.predict(input_data_imputed)[0]

        return render_template('index.html', gb_prediction=gb_prediction, rf_prediction=rf_prediction)

if __name__ == '__main__':
    app.run(debug=True)
