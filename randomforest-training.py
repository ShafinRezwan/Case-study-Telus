import pandas as pd #handles reading/processing data
from sklearn.model_selection import train_test_split #splits data into training and testing sets
from sklearn.feature_extraction.text import TfidfVectorizer #used to vectorize customer utterances
from sklearn.ensemble import RandomForestClassifier # ML learning model used
from sklearn.metrics import accuracy_score, classification_report # evaluates model performance
import joblib #after model is trained it saves it for future use

# Load the dataset
df = pd.read_csv('finance.tsv')


#clean the data
def clean(data):
    # remove missing rows
    df.dropna(inplace = True, subset=['utterance', 'authorRole'])
    # Ensure proper sorting by conversationId and turnNumber
    df.sort_values(by=['conversationId', 'turnNumber'], inplace=True)
    return df

df = clean(df)

#print(df.info()) #check for null values

# Filter only customer utterances
df = df[df["authorRole"] == "customer"]

# Define categories based on identified topics that I found using LDA
categories = {
    "account": 0,
    "technical support": 1,
    "loan inquiries": 2,
    "credit card": 3,
    "transaction history": 4,
   # "general_inquiry": 5
}

# Assign labels based on keyword presence
def assign_label(text):
    text = str(text).lower()
    if any(word in text for word in ["account", "balance", "transfer", "change"]):
        return categories["account"]
    elif any(word in text for word in ["password", "help"]):
        return categories["technical support"]
    elif any(word in text for word in ["loan", "rate", "commercial"]):
        return categories["loan inquiries"]
    elif any(word in text for word in ["card", "credit", "lost"]):
        return categories["credit card"]
    elif any(word in text for word in ["date", "history", "transaction"]):
        return categories["transaction history"]
    #elif any(word in text for word in ["hi", "hello", "thanks"]):
        #return categories["general_inquiry"]
    return None

df["label"] = df["utterance"].apply(assign_label) # assign label to each customer message
df = df.dropna(subset=["label"])  # Remove unclassified entries

# Vectorize text using TF-IDF
# converts text into numerical vectors.
#remove all stop words
vectorize = TfidfVectorizer(stop_words='english', max_features=500) 
x = vectorize.fit_transform(df["utterance"]) # makes a matrix
y = df["label"]

# checks x==y
#print(f"Number of Samples in X: {x.shape[0]}")
#print(f"Number of Labels in Y: {len(y)}")

# Split dataset into training and testing sets
# data is split into 60% training and 40% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=60) 



# Train Random Forest model
# use training x and y
rf_model = RandomForestClassifier(n_estimators=100, random_state=60, n_jobs=-1)
rf_model.fit(x_train, y_train)


#save trained model
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(vectorize, "tfidf_vectorizer.pkl")


# Evaluate the model
y_pred = rf_model.predict(x_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=categories.keys()))

print("Model and vectorizer saved successfully!")

'''
model accuracy shows how well model performs

Classification report shows
- percision - how many predicted categories were actually correct
- recall - how many actual categories were correctly predicted
- F1 score - A balance between percision and recall
- Support - # of test samples per category
'''


#LDA MODEL. I found this to figure out most frequent words based on 5 topics
# then using my own judgment I came up with categories names for them
'''

# Apply LDA for topic modeling
lda = LatentDirichletAllocation(n_components=5, random_state=60)
lda.fit(martix)

# Get top words per topic
words = vectorize.get_feature_names_out()
topics = {}
for i, topic in enumerate(lda.components_):
    top_words = [words[idx] for idx in topic.argsort()[-10:]]  # Get top 10 words per topic
    topics[f"Topic {i+1}"] = top_words

# Convert topics to DataFrame and display
topics_df = pd.DataFrame(topics)
print("Frequent Topics:" )
print(topics_df)
'''