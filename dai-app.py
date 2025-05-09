from flask import Flask, request, render_template
import pandas as pd
import os
import joblib
import logging

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def create_app():
    app = Flask(__name__)

    # Set up logging
    logging.basicConfig(level=logging.DEBUG)

    # Define file paths
    model_path = 'voting_clf.pkl'
    scaler_path = 'scaler.pkl'
    data_path = r'D:\diabetes\diabetes.csv'

    # Train model if not already trained
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        try:
            data = pd.read_csv(data_path)
            target = 'Outcome'
            X = data.drop(target, axis=1)
            y = data[target]

            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            log = LogisticRegression(max_iter=1000, random_state=42)
            svc = SVC(probability=True, random_state=42)

            voting_clf = VotingClassifier(estimators=[
                ('rf', rf),
                ('log', log),
                ('svc', svc)
            ], voting='soft')

            voting_clf.fit(X_train, y_train)

            joblib.dump(voting_clf, model_path)
            joblib.dump(scaler, scaler_path)

            app.logger.info("Model and scaler trained and saved successfully.")

        except Exception as e:
            app.logger.error(f"Error during training: {e}")
            return None
    else:
        app.logger.info("Model and scaler already exist. Skipping training.")

    @app.route('/')
    def home():
        return render_template('diabetes-index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            input_features = [float(x) for x in request.form.values()]
            expected_length = 8  # Adjust based on dataset feature count

            if len(input_features) != expected_length:
                raise ValueError(f"Expected {expected_length} features, got {len(input_features)}")

            scaler = joblib.load(scaler_path)
            model = joblib.load(model_path)

            input_scaled = scaler.transform([input_features])
            prediction = model.predict(input_scaled)
            result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'

            return render_template('diabetes-index.html', prediction_text=f'The model predicts: {result}')
        except Exception as e:
            app.logger.error(f"Prediction error: {e}")
            return render_template('diabetes-index.html', error_message=f"Error: {e}")

    return app

if __name__ == '__main__':
    app = create_app()
    if app:
        app.run(debug=True)
    else:
        print("Failed to create the Flask app.")
