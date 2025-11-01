import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
MODEL_FILENAME = 'random_forest_model.pkl'
FEATURE_NAMES = ['temperature', 'humidity', 'soil_pct', 'light_raw', 'gas_raw']
NUM_SAMPLES = 1500 # Total data points for training

# --- DATA GENERATION FUNCTION ---

def generate_balanced_data(n_samples):
    """
    Generates a mock dataset with distinct clusters for Optimal (2), Monitor (1), and Urgent (0).
    This ensures the model learns the boundaries correctly.
    """
    data = []
    
    # --- Class 2: Optimal Growth (Target ~ 500 samples) ---
    for _ in range(n_samples // 3):
        data.append([
            np.random.uniform(22, 27),    # Temp (Ideal range)
            np.random.uniform(60, 75),    # Hum (Ideal range)
            np.random.uniform(70, 85),    # Soil (Ideal moisture)
            np.random.randint(2500, 4000),# Light (High)
            np.random.randint(400, 1000), # Gas (Low)
            2                             # Label: Optimal
        ])
        
    # --- Class 1: Monitoring Required (Target ~ 500 samples) ---
    for _ in range(n_samples // 3):
        data.append([
            np.random.uniform(18, 22),    # Temp (Slightly low)
            np.random.uniform(75, 90),    # Hum (High)
            np.random.uniform(50, 65),    # Soil (Slightly low)
            np.random.randint(1000, 2500),# Light (Moderate)
            np.random.randint(1000, 2000),# Gas (Moderate)
            1                             # Label: Monitor
        ])

    # --- Class 0: Urgent Intervention (Target ~ 500 samples) ---
    for _ in range(n_samples // 3):
        data.append([
            np.random.uniform(30, 40),    # Temp (Very high/low risk)
            np.random.uniform(30, 45),    # Hum (Very low)
            np.random.uniform(10, 40),    # Soil (Critical low)
            np.random.randint(0, 1000),   # Light (Very low)
            np.random.randint(2500, 4000),# Gas (Very high risk)
            0                             # Label: Urgent
        ])

    df = pd.DataFrame(data, columns=FEATURE_NAMES + ['target'])
    return df

# --- MAIN EXECUTION ---

if __name__ == '__main__':
    print("--- 1. Generating Balanced Training Data ---")
    
    # 1. Generate the synthetic data
    df = generate_balanced_data(NUM_SAMPLES)

    # 2. Separate features (X) and target (y)
    X = df[FEATURE_NAMES]
    y = df['target']

    # 3. Split data (optional for mock, good practice)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Data Distribution (Optimal: {len(df[df['target'] == 2])}, Monitor: {len(df[df['target'] == 1])}, Urgent: {len(df[df['target'] == 0])})")
    
    print("\n--- 2. Training Random Forest Model ---")
    
    # 4. Train the classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # 5. Evaluate the model
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Training Complete. Accuracy on Test Set: {accuracy:.2f}")

    print("\n--- 3. Saving Model Asset ---")
    
    # 6. Save the trained model to the .pkl file
    with open(MODEL_FILENAME, 'wb') as file:
        pickle.dump(rf_classifier, file)
        
    print(f"SUCCESS: New trained model saved as {MODEL_FILENAME}. Restart your FastAPI server now!")
