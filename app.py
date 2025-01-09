import joblib

# Feature names based on your dataset
feature_names = [
    "area", "bedrooms", "bathrooms", "stories", "mainroad",
    "guestroom", "basement", "hotwaterheating", "airconditioning",
    "parking", "prefarea", "furnishingstatus"
]

# Load the pre-trained model
model = joblib.load('House_Price_model.pkl')

# Collect input dynamically
loan_list = []
print("Please enter the following details:")
for name in feature_names:
    # If the feature is categorical, you might want specific instructions
    if name in ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]:
        value = input(f"Is there {name}? (yes=1, no=0): ")
    elif name == "furnishingstatus":
        value = input("Furnishing status (furnished=1, semi-furnished=2, unfurnished=0): ")
    else:
        value = input(f"Enter {name}: ")
    loan_list.append(float(value))  # Convert input to float for numerical features

# Reshape the input to 2D (required format for prediction)
loan_array = [loan_list]  # Wrapping it in another list to create 2D array-like structure

# Predict
pred = model.predict(loan_array)
print(f"Prediction: {pred[0]}")
