Heart Disease Prediction Web Application

AI-powered web application to predict the risk of heart disease using Machine Learning.
The project uses Random Forest for prediction, Jupyter Notebook for model training, and
Streamlit for building an interactive web dashboard.

--------------------------------------------------

Project Overview

Heart disease is one of the leading causes of death worldwide.
This project predicts the likelihood of heart disease based on patient health parameters.
It is designed as a professional internship-level project and portfolio application.

--------------------------------------------------

Technologies Used

Python
Pandas
NumPy
Scikit-learn
Jupyter Notebook
Streamlit
Plotly
Git & GitHub

--------------------------------------------------

Dataset Information

The dataset contains medical attributes such as:
- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- ECG results
- Maximum heart rate
- Exercise-induced angina

Target column:
- num (indicates presence of heart disease)

--------------------------------------------------

Model Training (Jupyter Notebook)

Model training is done in Jupyter Notebook (heartdisease.ipynb).

Steps included:
- Data cleaning
- Handling missing values
- Encoding categorical features
- Feature selection
- Random Forest model training
- Model evaluation
- Saving the trained model as best_rf_model.pkl

Jupyter Notebook is used because it is ideal for experimentation, analysis,
and visualization during model development.

--------------------------------------------------

Web Application (Streamlit)

The trained model is deployed using Streamlit (heart.py).

Features of the web app:
- Dark themed professional UI
- Patient data input form
- Heart disease prediction
- Animated risk gauge
- Interactive charts and graphs
- Feature importance visualization

--------------------------------------------------

Project Structure

heart-disease-streamlit/
|
|-- heart.py
|-- heartdisease.ipynb
|-- heart.csv
|-- best_rf_model.pkl
|-- requirements.txt
|-- README.md

--------------------------------------------------

How to Run the Project Locally

Step 1: Clone the repository
git clone https://github.com/yourusername/heart-disease-streamlit.git
cd heart-disease-streamlit

Step 2: Install dependencies
pip install -r requirements.txt

Step 3: Run the Streamlit app
streamlit run heart.py

The application will open in the browser at:
http://localhost:8501

--------------------------------------------------

Why requirements.txt is Important

The requirements.txt file contains all required Python libraries.
It helps others run the project easily and is required for deployment.
It is generated using:

pip freeze > requirements.txt

--------------------------------------------------

Deployment

The project is hosted on GitHub and can be deployed on Streamlit Cloud
to get a live web application link.

--------------------------------------------------

Author

Rishika Peddireddy
Graduated
Machine Learning & Data Analytics

--------------------------------------------------

License

This project is developed for internship in DynamicNetworks