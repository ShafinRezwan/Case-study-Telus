# this file is ran after the training file
import joblib

# Load the trained model 
rf_model = joblib.load("random_forest_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Define the category mapping
categories = {
    0: "account",
    1: "technical support",
    2: "loan inquiries",
    3: "credit card",
    4: "transaction history",
    #5: "general_inquiry"
}

# Function to classify new customer messages
def classify_text(text):
    vectorized_text = vectorizer.transform([text])
    # predict category using the trained model
    label = rf_model.predict(vectorized_text)[0]
    return categories[label]

# Example usage
print(classify_text("I WANT TO CHANGE MY ACCOUNT ADDRESS "))
print(classify_text("I was wrongly charged for late fees even if I paid back everything on time"))
print(classify_text("I lost my credit card, what should I do?"))
print(classify_text("Order 47 checks for me and tell me bank's routing number"))
