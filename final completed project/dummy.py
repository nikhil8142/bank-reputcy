from app2 import User, app  # Ensure correct file name

with app.app_context():
    db.create_all()
    print("Database initialized successfully!")
