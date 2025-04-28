from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return redirect(url_for('register'))
        
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully.', 'info')
    return redirect(url_for('home'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        username = request.form['username']
        new_password = generate_password_hash(request.form['password'])
        user = User.query.filter_by(username=username).first()
        
        if user:
            user.password = new_password
            db.session.commit()
            flash('Password reset successful! Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash('User not found.', 'danger')
    return render_template('forgot_password.html')

@app.route('/predict', methods=['POST'])
def predict():
    assets = float(request.form['assets'])
    liabilities = float(request.form['liabilities'])
    monthly_income = float(request.form['monthly_income'])
    monthly_expenses = float(request.form['monthly_expenses'])
    debt_ratio = float(request.form['debt_ratio'])
    credit_score = int(request.form['credit_score'])
    loan_amount = float(request.form['loan_amount'])
    loan_duration = int(request.form['loan_duration'])
    interest_rate = float(request.form['interest_rate'])
    savings = float(request.form['savings'])
    investment = float(request.form['investment'])
    other_income = float(request.form['other_income'])
    bankruptcies = int(request.form['bankruptcies'])
    
    # Basic bankruptcy eligibility check logic
    eligibility_score = (assets - liabilities) + (monthly_income - monthly_expenses) - (debt_ratio * 100) + (credit_score / 10)
    
    if eligibility_score < 0 or bankruptcies > 0:
        result = "High risk of bankruptcy. Seek financial advice."
    else:
        result = "Low risk of bankruptcy. Maintain financial discipline."
    
    return render_template('home.html', result=result)

if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
