# import pandas as pd
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

# # Load dataset
# df = pd.read_csv("1st_yr.csv")

# # Prepare data (adjust column names as needed)
# X = df.drop(columns=['Bankrupt', 'Id'], errors='ignore')
# y = df['Bankrupt']

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# # Save model
# joblib.dump(model, "bankruptcy_model.pkl")
# print("Model saved successfully!")




# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # Load the dataset
# file_path = "1st_yr.csv"  # Change if using a different dataset
# df = pd.read_csv(file_path, low_memory=False)

# # Convert all '?' values to NaN
# df.replace('?', np.nan, inplace=True)

# # Convert all columns to numeric (forcing conversion)
# df = df.apply(pd.to_numeric, errors='coerce')

# # Fill missing values with column means
# df.fillna(df.mean(), inplace=True)

# # Split into features (X) and target variable (y)
# X = df.drop(columns=['Bankrupt', 'Id'], errors='ignore')  # Drop non-feature columns
# y = df['Bankrupt'] if 'Bankrupt' in df.columns else None  # Ensure target column exists

# # Check if target variable exists
# if y is None or y.isnull().all():
#     raise ValueError("Target variable 'Bankrupt' not found in dataset!")

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Train the model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Save the model
# joblib.dump(model, "bankruptcy_model.pkl")
# print("Model training complete. Saved as bankruptcy_model.pkl")





# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # Load the dataset
# file_path = "1st_yr.csv"  # Change if using a different dataset
# df = pd.read_csv(file_path, low_memory=False)

# # Convert all '?' values to NaN
# df.replace('?', np.nan, inplace=True)

# # Convert all columns to numeric
# df = df.apply(pd.to_numeric, errors='coerce')

# # Fill missing values with column means
# df.fillna(df.mean(), inplace=True)

# # Split into features (X) and target variable (y)
# X = df.drop(columns=['Bankrupt', 'Id'], errors='ignore')  
# y = df['Bankrupt'] if 'Bankrupt' in df.columns else None  

# # Check if target variable exists
# if y is None or y.isnull().all():
#     raise ValueError("Target variable 'Bankrupt' not found in dataset!")

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Train the model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Save the model and feature names
# joblib.dump((model, list(X.columns)), "bankruptcy_model.pkl")
# print("Model training complete. Saved as bankruptcy_model.pkl")



# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # Load the dataset
# file_path = "1st_yr.csv"  
# df = pd.read_csv(file_path, low_memory=False)

# # Convert all '?' values to NaN
# df.replace('?', np.nan, inplace=True)

# # Convert all columns to numeric
# df = df.apply(pd.to_numeric, errors='coerce')

# # Fill missing values with column means
# df.fillna(df.mean(), inplace=True)

# # Split into features (X) and target variable (y)
# X = df.drop(columns=['Bankrupt', 'Id'], errors='ignore')  
# y = df['Bankrupt'] if 'Bankrupt' in df.columns else None  

# # Check if target variable exists
# if y is None or y.isnull().all():
#     raise ValueError("Target variable 'Bankrupt' not found in dataset!")

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Train the model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Save the model and feature names
# joblib.dump((model, X.columns.tolist()), "bankruptcy_model.pkl")  
# print("Model training complete. Saved as bankruptcy_model.pkl")





import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from deap import base, creator, tools, algorithms  # For GA
from pyswarm import pso  # For PSO

# Folder containing multiple training datasets
DATASET_FOLDER = "train/"

# Load and preprocess multiple datasets
dataframes = []
for file in os.listdir(DATASET_FOLDER):
    if file.endswith(".csv"):  # Load only CSV files
        df = pd.read_csv(os.path.join(DATASET_FOLDER, file))
        df.replace('?', np.nan, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric
        df.fillna(df.mean(), inplace=True)  # Fill missing values with column means
        dataframes.append(df)

# Merge all datasets
combined_df = pd.concat(dataframes, ignore_index=True)

combined_df['bankrupt'] = pd.to_numeric(combined_df['bankrupt'], errors='coerce')
combined_df['bankrupt'].fillna(combined_df['bankrupt'].mode()[0], inplace=True)  # Fill NaN with most common value
combined_df['bankrupt'] = combined_df['bankrupt'].astype(int)  # Convert to integer

# Extract features and target
X = combined_df[[f'X{i}' for i in range(1, 65)]].astype(float)  # Features X1 to X64
y = combined_df['bankrupt']  # Target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Data preprocessing completed successfully!")

# --- Genetic Algorithm for Feature Selection ---
def evaluate(individual):
    selected_features = [i for i in range(len(individual)) if individual[i] > 0.5]
    if not selected_features:  # Ensure at least one feature is selected
        return 1000,
    
    X_selected = X_train[:, selected_features]
    
    model = LogisticRegression(max_iter=500)
    model.fit(X_selected, y_train)
    
    accuracy = model.score(X_test[:, selected_features], y_test)
    return -accuracy,  # We minimize the negative accuracy

# Define GA
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.randint, 0, 2)  # Binary selection
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X_train.shape[1])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Run Genetic Algorithm
population = toolbox.population(n=50)
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=30, verbose=True)

# Get best individual
best_individual = tools.selBest(population, k=1)[0]
num_features = 24

# Select the top 'num_features' features
sorted_features = sorted(range(len(best_individual)), key=lambda i: best_individual[i], reverse=True)
selected_features = sorted_features[:num_features]

# Train final GA model with selected features
X_selected = X_train[:, selected_features]
final_model_ga = LogisticRegression(max_iter=500)
final_model_ga.fit(X_selected, y_train)

# --- Particle Swarm Optimization (PSO) for Hyperparameter Tuning ---
def pso_objective(params):
    C, tol = params  # Extract hyperparameters
    model = LogisticRegression(C=C, tol=tol, max_iter=500)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return -accuracy  # We minimize the negative accuracy

# Define search bounds for C (Regularization) and tol (Tolerance)
lb = [0.0001, 1e-6]  # Lower bounds
ub = [10, 1e-2]  # Upper bounds

# Run Particle Swarm Optimization
best_params, _ = pso(pso_objective, lb, ub)

# Train final PSO model with best parameters
best_C, best_tol = best_params
final_model_pso = LogisticRegression(C=best_C, tol=best_tol, max_iter=500)
final_model_pso.fit(X_train, y_train)

# --- Save Models and Scalers ---
joblib.dump(final_model_ga, "model/model_ga.sav")
joblib.dump(final_model_pso, "model/model_pso.sav")

scaler_ga = StandardScaler()
X_train_ga_scaled = scaler_ga.fit_transform(X_selected)
joblib.dump(scaler_ga, 'model/scaler_ga.sav')

scaler_pso = StandardScaler()
X_train_pso_scaled = scaler_pso.fit_transform(X_train)
joblib.dump(scaler_pso, 'model/scaler_pso.sav')

print("Models and scalers saved successfully.")

# --- Example Predictions ---
sample = X_test[0].reshape(1, -1)
sample_selected = sample[:, selected_features]  # Select features chosen by GA
print("GA Model Prediction:", final_model_ga.predict(sample_selected))
print("PSO Model Prediction:", final_model_pso.predict(sample))
