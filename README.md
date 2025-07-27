<!DOCTYPE html>
<html>
<head>
    <title>Insurance Coverage & Premium Prediction Project</title>
</head>
<body>

<h1>Insurance Coverage & Premium Prediction System</h1>

<p>
This project predicts an employee's suggested health insurance coverage and monthly premium 
based on their age, income, occupation, lifestyle, and health metrics using Machine Learning.
</p>

<h2>Features</h2>
<ul>
    <li>Predicts suggested health insurance coverage amount (₹).</li>
    <li>Estimates monthly premium based on user profile.</li>
    <li>User-friendly Streamlit web app interface.</li>
    <li>Handles multiple inputs: age, gender, income, dependents, BMI, smoker status, and occupation type.</li>
    <li>Displays warnings for high-risk profiles (smokers or high BMI).</li>
</ul>

<h2>Technologies Used</h2>
<ul>
    <li>Python</li>
    <li>Streamlit</li>
    <li>scikit-learn</li>
    <li>XGBoost</li>
    <li>NumPy and Pandas</li>
    <li>Joblib (for saving and loading models)</li>
</ul>

<h2>Project Workflow</h2>
<ol>
    <li>Data Cleaning and Preprocessing (removing nulls, encoding categories).</li>
    <li>Feature Engineering (derived features like age group, BMI category, risk flags).</li>
    <li>Scaling of numerical data using StandardScaler.</li>
    <li>Model Training using Random Forest and XGBoost for Coverage and Premium.</li>
    <li>Deploying predictions via a Streamlit Web Application.</li>
</ol>

<h2>How to Run</h2>
<ol>
    <li>Clone or download the project repository.</li>
    <li>Install dependencies using:
        <pre>pip install -r requirements.txt</pre>
    </li>
    <li>Make sure the trained models and scaler are saved inside the <b>models</b> folder:
        <ul>
            <li>final_scaler.pkl</li>
            <li>best_coverage_model.pkl</li>
            <li>best_premium_model.pkl</li>
        </ul>
    </li>
    <li>Run the Streamlit app:
        <pre>streamlit run app.py</pre>
    </li>
    <li>Open the app in your browser at:
        <pre>http://localhost:8501</pre>
    </li>
</ol>

<h2>Input Parameters</h2>
<ul>
    <li>Age (Years)</li>
    <li>Gender (Male/Female)</li>
    <li>Annual Income (₹)</li>
    <li>Number of Dependents</li>
    <li>Smoker Status (Yes/No)</li>
    <li>Occupation Type (IT, Finance, Healthcare, Manufacturing, Education, Retail, Government, Other)</li>
    <li>BMI (Body Mass Index)</li>
</ul>

<h2>Outputs</h2>
<ul>
    <li>Predicted Health Insurance Coverage (₹)</li>
    <li>Estimated Monthly Premium (₹)</li>
    <li>Warnings for high-risk profiles (if applicable).</li>
</ul>

<h2>Author</h2>
<p>Developed by Dhruv Kajla and Mukund Malik.</p>

</body>
</html>
