import os
import numpy as np
import pandas as pd
import joblib  # To load the saved models
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, request, redirect, url_for, flash,session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.preprocessing import StandardScaler
import io
import base64
from flask_session import Session
import matplotlib.pyplot as plt
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import io
import tkinter as tk
import base64
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
import tempfile
import matplotlib
matplotlib.use('Agg')





# Initialize Flask App
app = Flask(__name__, template_folder='templates')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx'}
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:/Users/nikhi/Desktop/final completed project/instance/site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
Session(app)

# Initialize Database
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
def get_user_by_id(user_id):
    if user_id:
        # Instead of User.query.get(), use db.session.get() for SQLAlchemy 2.0
        return db.session.get(User, user_id)  # Get user from database
    return None
def generate_plot_url(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches="tight")
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()
def plot_bankruptcy_pie_chart(df):
    bankruptcy_counts = df['Bankruptcy Prediction GA'].value_counts()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bankruptcy_counts.plot(kind='pie', labels=bankruptcy_counts.index, autopct='%1.1f%%', colors=['green', 'red'], ax=ax)
    plt.title("Bankruptcy Prediction Distribution")
    plt.ylabel('')  # Hide y-label
    
    plot_pie_url = generate_plot_url(fig)
    plt.close(fig)
    return plot_pie_url
def plot_pairplot(df):
    selected_features = df.iloc[:, :5]  # Select first 5 financial features for visualization

    fig = sns.pairplot(selected_features, diag_kind='kde', corner=True)
    fig.fig.suptitle("Pairplot of Financial Features", y=1.02)
    
    plot_pairplot_url = generate_plot_url(fig.fig)
    plt.close(fig.fig)
    return plot_pairplot_url

def plot_bankruptcy_prediction(df):
    # Count occurrences of each bankruptcy status
    bankruptcy_counts = df['Bankruptcy Prediction GA'].value_counts()

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    bankruptcy_counts.plot(kind='bar', color=['green', 'red'], ax=ax)
    plt.title("Bankruptcy Prediction Results")
    plt.xlabel("Prediction")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    
    # Convert plot to URL
    plot_bankruptcy_url = generate_plot_url(fig)
    plt.close(fig)
    return plot_bankruptcy_url

# 1. Box Plot
def plot_boxplot(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    boxprops = dict(linestyle='-', linewidth=1.5, color='black')
    sns.boxplot(data=df.iloc[:, -3:], orient="h", boxprops=boxprops)
    plt.title("Box Plot of Bankruptcy Predictions")
    plt.xlabel("Prediction Categories")
    plt.ylabel("Values")
    plt.grid(True)
    
    plot_box_url = generate_plot_url(fig)
    plt.close(fig)
    
    return plot_box_url

# 2. Histogram
def plot_histogram(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    df.iloc[:, -3].hist(bins=20, ax=ax, color='skyblue', edgecolor='black')
    plt.title("Histogram of Last Column")
    
    plot_hist_url = generate_plot_url(fig)
    plt.close(fig)
    return plot_hist_url

def generate_plot_url(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches="tight")
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_feature_importance(df):
    try:
        X = df.iloc[:, :-2]  # Select all columns except last two
        y = df.iloc[:, -1]   # Select last column as target

        # Train model
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        rf_model.fit(X, y)

        # Create feature importance plot
        fig, ax = plt.subplots(figsize=(8, 6))
        feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        feature_importances.plot(kind='bar', ax=ax, color='teal')
        plt.title("Feature Importance (Multi-Class)")

        plot_feature_importance_url = generate_plot_url(fig)
        plt.close(fig)
        return plot_feature_importance_url

    except Exception as e:
        print(f"Error in plot_feature_importance: {e}")  # Debugging
        return None

# Decision Tree Visualization for Multi-Class
from sklearn.tree import DecisionTreeClassifier

def plot_decision_tree(df):
    try:
        X = df.iloc[:, :-2]  # Select all columns except last two
        y = df.iloc[:, -1]   # Select last column as target

        # Train a single decision tree
        dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt_model.fit(X, y)

        # Create decision tree plot
        fig, ax = plt.subplots(figsize=(12, 8))
        class_labels = list(map(str, y.unique()))  # Ensure correct class names
        plot_tree(dt_model, feature_names=X.columns, class_names=class_labels, filled=True, ax=ax)
        plt.title("Decision Tree (Multi-Class)")

        plot_tree_url = generate_plot_url(fig)
        plt.close(fig)
        return plot_tree_url
    
    except Exception as e:
        print(f"Error in plot_decision_tree: {e}")  # Debugging
        return None


# User Model
class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# File Upload Helper
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_plot_url(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches="tight")
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()
def determine_bankruptcy_status(df, file_name):
    ga_high_risk_count = (df['Bankruptcy Prediction GA'] == "High Risk").sum()
    pso_high_risk_count = (df['Bankruptcy Prediction PSO'] == "High Risk").sum()

    total_records = len(df)
    
    # Calculate risk percentages
    ga_risk_percentage = (ga_high_risk_count / total_records) * 100
    pso_risk_percentage = (pso_high_risk_count / total_records) * 100

    # Average risk percentage from both models
    avg_risk_percentage = (ga_risk_percentage + pso_risk_percentage) / 2

    if avg_risk_percentage > 70:
        return f"{file_name} is **BANKRUPT** (Both models strongly predict bankruptcy)"
    elif avg_risk_percentage > 50:
        return f"{file_name} is at HIGH RISK (Both models indicate significant bankruptcy risk)"
    elif avg_risk_percentage > 30:
        return f"{file_name} is at MODERATE RISK (At least one model suggests a noticeable risk)"
    elif avg_risk_percentage > 10:
        return f"{file_name} is at LOW RISK (Minimal bankruptcy indications)"
    else:
        return f"{file_name} is NOT BANKRUPT (Both models indicate very low risk)"
    
# Load Models and Scalers
def load_models():
    # Load the GA-optimized model and scaler
    model_ga = joblib.load("model/model_ga.sav", mmap_mode=None)
    model_pso = joblib.load("model/model_pso.sav")
    
    # Load the respective scalers for each model
    scaler_ga = joblib.load("model/scaler_ga.sav")  # Assuming scaler for 64 features
    scaler_pso = joblib.load("model/scaler_pso.sav")  # Assuming scaler for 24 features
    num_features1 = model_ga.coef_.shape[1]  # Number of features in the model
    num_features2 = model_pso.coef_.shape[1]  # Number of features in the model
    print(f"Number of features used by model_ga: {num_features1}")
    print(f"Number of features used by model_pso: {num_features2}")
    
    return model_ga, model_pso, scaler_ga, scaler_pso

def calculate_bankruptcy(row, model, scaler):
    # Use 24 features for this model (model_pso)
    X_values = np.array([row[f'X{i}'] for i in range(1, 25)]).reshape(1, -1)  # First 24 features
    X_scaled = scaler.transform(X_values)  # Apply scaling
    Z = model.predict_proba(X_scaled)[0][1]
    
    print(f"X_values: {X_values}")  # Debugging line
    print(f"Predicted Probability: {Z}")  # Debugging line
    
    if Z > 0.8:
        return "High Risk"
    elif Z > 0.5:
        return "Medium Risk"
    else:
        return "Low Risk"

def calculate_bankruptcy_standardscaler(row, model, scaler):
    # Use 64 features for this model (model_ga)
    X_values = np.array([row[f'X{i}'] for i in range(1, 65)]).reshape(1, -1)  # First 64 features
    X_scaled = scaler.transform(X_values)  # Apply scaling
    Z = model.predict_proba(X_scaled)[0][1]
    
    print(f"X_values: {X_values}")  # Debugging line
    print(f"Predicted Probability: {Z}")  # Debugging line
    
    if Z > 0.8:
        return "High Risk"
    elif Z > 0.5:
        return "Medium Risk"
    else:
        return "Low Risk"








@app.route('/')
def home():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        print("Request form:", request.form)
        print("Request args:", request.args)
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('predict'))
        flash('Invalid email or password.', 'danger')
        
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        full_name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm-password')

        if not all([full_name, email, password, confirm_password]):
            flash('All fields are required!', 'danger')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already exists.', 'warning')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(full_name=full_name, email=email, password=hashed_password)
        try:
           db.session.add(new_user)
           db.session.commit()
           print("User successfully added!")
        except Exception as e:
           db.session.rollback()
           print("Error adding user:", e)

       
        flash('Account created successfully!', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/preview', methods=['GET', 'POST'])
@login_required
def preview():
    if 'df' not in session:
        flash("Please upload a file first to view the graphs.", "warning")
        return redirect(url_for('predict'))
    file_name = session.get('filename', 'Uploaded File')
    # Assuming session['df'] is a JSON string, we use pd.read_json()
    try:
        df = pd.read_json(session['df'])
    except ValueError as e:
        flash(f"Error loading data: {str(e)}", "danger")
        return redirect(url_for('predict'))

    final_bankruptcy_status = determine_bankruptcy_status(df, file_name)
    plot_box_url = plot_boxplot(df)
    plot_hist_url = plot_histogram(df)
    plot_feature_importance_url = plot_feature_importance(df)
    plot_tree_url = plot_decision_tree(df)
    plot_bankruptcy_url = plot_bankruptcy_prediction(df)
    plot_pie_url = plot_bankruptcy_pie_chart(df)
    plot_pairplot_url = plot_pairplot(df)

    return render_template('preview.html',
                           tables=df.to_html(classes='table table-bordered').strip(),
                           final_bankruptcy_status=final_bankruptcy_status,
                           plot_box_url=plot_box_url,
                           plot_hist_url=plot_hist_url,
                           plot_feature_importance_url=plot_feature_importance_url,
                           plot_tree_url=plot_tree_url,
                           plot_pie_url=plot_pie_url,
                           plot_bankruptcy_url=plot_bankruptcy_url,
                           plot_pairplot_url=plot_pairplot_url)
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        new_password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if not user:
            flash('Email not found.', 'danger')
            return redirect(url_for('forgot_password'))

        hashed_password = generate_password_hash(new_password, method='pbkdf2:sha256')
        user.password = hashed_password
        db.session.commit()
        
        flash('Password reset successfully! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('forgot_password.html')

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    model_ga, model_pso, scaler_ga, scaler_pso = load_models()  # Load models and scalers

    if request.method == 'POST':
        session.pop('df', None)  # Clear previous session data
        session.pop('file_name', None)

        file = request.files.get('file')

        # Check if a file is uploaded and it's of the allowed type
        if not file or not allowed_file(file.filename):
            flash("Invalid file type! Upload CSV or Excel.", "danger")
            return redirect(url_for('predict'))  

        # Secure the filename and save the file to the specified folder
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        file_name_without_extension = os.path.splitext(filename)[0]

        # Read the file into a DataFrame (CSV or Excel)
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
        except Exception as e:
            flash(f"Error reading the file: {str(e)}", "danger")
            return redirect(url_for('predict'))  

        # Replace '?' with NaN and handle non-numeric values
        df.replace('?', np.nan, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')

        # Fill NaN values with column means
        df = df.fillna(df.mean()).reset_index(drop=True)

        # Apply prediction using both models
        try:
            df['Bankruptcy Prediction PSO'] = df.apply(lambda row: calculate_bankruptcy_standardscaler(row, model_pso, scaler_pso), axis=1)
            df['Bankruptcy Prediction GA'] = df.apply(lambda row: calculate_bankruptcy(row, model_ga, scaler_ga), axis=1)
        except Exception as e:
            flash(f"Error during prediction: {str(e)}", "danger")
            return redirect(url_for('predict'))  

        # --- Ensure Data is in Float Format and Handle Missing Data ---
        numeric_predictions = df[['Bankruptcy Prediction PSO', 'Bankruptcy Prediction GA']].apply(pd.to_numeric, errors='coerce')

        # Fill NaN values with 0 or another appropriate placeholder
        numeric_predictions.fillna(0, inplace=True)

        # Check if all values in prediction columns are NaN
        if numeric_predictions.isnull().all().any():
            flash('No valid data for plotting!', 'warning')
            return redirect(url_for('predict'))

        final_bankruptcy_status = determine_bankruptcy_status(df, file_name_without_extension)
        session['df'] = df.to_json()
        session['file_name'] = file_name_without_extension

        return render_template('predict.html', final_bankruptcy_status=final_bankruptcy_status, file_uploaded=True)

    # If GET request, render the upload form
    return render_template('predict.html', file_uploaded=False)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
    root = tk.Tk()
    root.mainloop()