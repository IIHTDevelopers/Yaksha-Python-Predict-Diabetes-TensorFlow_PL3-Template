import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
def load_dataset():
    """
    TODO:
    - Load the CSV file 'pima-indians-diabetes.data.csv'.
    - Assign column names as per the Pima dataset specification.
    - Return the DataFrame.
    """
    # Example placeholder structure
    # df = pd.read_csv("pima-indians-diabetes.data.csv", names=[...])
    # return df
    return pd.DataFrame()  # placeholder to avoid crash
    pass

# 2. Preprocess dataset
def preprocess_data(df):
    """
    TODO:
    - Split the DataFrame into X (features) and y (target - 'Outcome').
    - Use StandardScaler to scale the feature values.
    - Split into train/test using train_test_split.
    - Return X_train, X_test, y_train, y_test
    """
    # return X_train, X_test, y_train, y_test
    return None, None, None, None
    pass

# 3. Build model
def build_model(input_dim):
    """
    TODO:
    - Create a tf.keras.Sequential model with:
        * Dense(64, relu)
        * Dense(32, relu)
        * Dense(1, sigmoid)
    - Use input_dim for input_shape of the first layer.
    - Return the model.
    """
    return None
    pass

# 4. Compile model
def compile_model(model):
    """
    TODO:
    - Compile the model using:
        * Optimizer: 'adam'
        * Loss: 'binary_crossentropy'
        * Metric: 'accuracy'
    """
    pass

# 5. Train model
def train_model(model, X_train, y_train, X_val, y_val, epochs=20):
    """
    TODO:
    - Fit the model on X_train and y_train.
    - Use validation_data=(X_val, y_val), batch_size=32.
    - Set verbose=0 and epochs to the input value.
    - Print "Training completed."
    - Return the history object.
    """
    return None
    pass

# 6. Evaluate model
def evaluate_model(model, X_test, y_test):
    """
    TODO:
    - Evaluate the model using model.evaluate().
    - Print the accuracy in percentage format.
    - Return the accuracy.
    """
    return 0.0
    pass

# 7. Predict on a new sample
def predict_sample(model, sample):
    """
    TODO:
    - Reshape the input sample to match model's input shape.
    - Use model.predict() to get the output probability.
    - If probability >= 0.5, return "Diabetic", else "Non-Diabetic".
    """
    return "Unknown"
    pass

# 8. Prepare a sample input using StandardScaler
def prepare_sample_input(raw_sample, scaler):
    """
    TODO:
    - Use the fitted scaler to transform the raw input list.
    - Return the transformed sample as a flat array.
    """
    return np.zeros(len(raw_sample))
    pass

# 9. Load sample input from file
def load_sample_from_file(filename="sample_input.txt"):
    """
    TODO:
    - Open the given file and read the first line.
    - Split the line by comma and convert to a list of floats.
    - Handle exceptions and print error if any.
    - Return the list.
    """
    try:
        return []  # placeholder
    except Exception as e:
        print("Error reading sample input from file:", e)
        return None
    pass

# ===== Main Driver Code =====
if __name__ == "__main__":
    # Step 1: Load dataset
    df = load_dataset()

    # Step 2: Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Step 3: Extract scaler separately for later use
    scaler = StandardScaler().fit(df.drop('Outcome', axis=1).values)

    # Step 4: Build and compile model
    model = build_model(input_dim=X_train.shape[1]) if X_train is not None else None
    if model:
        compile_model(model)

        # Step 5: Train model
        history = train_model(model, X_train, y_train, X_test, y_test, epochs=20)

        # Step 6: Evaluate model
        evaluate_model(model, X_test, y_test)

        # Step 7: Predict using sample input file
        raw_sample = load_sample_from_file("sample_input.txt")
        if raw_sample:
            processed_sample = prepare_sample_input(raw_sample, scaler)
            result = predict_sample(model, processed_sample)
            print("Prediction for sample:", result)
        else:
            print("Prediction could not be performed due to invalid input.")
    else:
        print("Model build failed. Check input data or model structure.")
