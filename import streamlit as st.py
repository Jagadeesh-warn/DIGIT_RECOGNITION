import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Set a title and description for the Streamlit app
st.title('Digit Recognizer')
st.write("Select a model to see its accuracy on the digit dataset:")

# Load the digits dataset
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Split the dataset into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a dropdown to select a model
selected_model = st.selectbox('Select a model', ('K-Nearest Neighbors', 'Naive Bayes', 'Support Vector Machine', 'Logistic Regression'))

# Define and train the selected model
if selected_model == 'K-Nearest Neighbors':
    model = KNeighborsClassifier()
elif selected_model == 'Naive Bayes':
    model = GaussianNB()
elif selected_model == 'Support Vector Machine':
    model = SVC()
elif selected_model == 'Logistic Regression':
    model = LogisticRegression(max_iter=10000)

model.fit(X_train, y_train)

# Calculate and display accuracy
accuracy = model.score(X_test, y_test)
st.write(f"Accuracy for {selected_model}: {accuracy:.2f}")
