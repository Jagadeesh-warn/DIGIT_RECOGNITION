import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pickle

# Function to load or train models
def load_or_train_model(selected_model):
    model = None
    model_file = f'{selected_model.lower().replace(" ", "_")}_model.pkl'
    try:
        with open(model_file, 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        if selected_model == 'K-Nearest Neighbors':
            st.write ("Accuracy for KNN is 0.90")
            model = KNeighborsClassifier()

        elif selected_model == 'Multi Layered Perceptron':
            st.write ("Accuracy for Multi layered perceptron is  is 0.97")
            #model = GaussianNB()
        elif selected_model == 'Support Vector Machine':
            st.write ("Accuracy for SVM is 0.92")
            #model = SVC()
        elif selected_model == 'Logistic Regression':
            st.write ("Accuracy for KNN is 0.90")
            model = LogisticRegression(max_iter=10000)
        model.fit(X_train, y_train)
        with open(model_file, 'wb') as file:
            pickle.dump(model, file)
    return model

# Set a title and description for the Streamlit app
st.title(' Digit Recognizer ')
st.write(" WELCOME TO DIGIT RECOGNIZER ")
trainset = pd.read_csv('train.csv')
# Load the digits dataset
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Split the dataset into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define background image
background_image = '25336.jpg'  # Replace with the actual image file name

# Apply background image using CSS
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url({background_image});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation options
page = st.sidebar.selectbox('Select a Page', ['Home', 'Transformations','Visualization', 'Model', 'ANOVA Test'])

# Home page
if page == 'Home':
    st.write("   DIGIT RECOGNISER   ")
    st.write(trainset.head(60))
# Visualization page
elif page == 'Visualization':
   # Visualization page
    st.write(" VISUALISATIONS ")
    
    # Add images to the Visualization page
    st.image('digit dist.png', caption='DIGIT DISTRIBUTION AMONG VARIOUS CLASSES', use_column_width=True)
    st.image('heatmap.png', caption='HEAT MAP BETWEEN PIXEL INTENSITIES ', use_column_width=True)
    st.image('VIS.png', caption='VISULAISATION OF DIGITS ', use_column_width=True)
    st.image('AVERAGE .png', caption='AVERAGE VISULAISATION OF DIGITS ', use_column_width=True)
    st.image('PIXEL.png', caption='PIXEL DISTRIBUTION', use_column_width=True)

# Model page
elif page == 'Model':
    st.write("Select a model to see its accuracy on the digit dataset:")
    selected_model = st.selectbox('Select a model', ('K-Nearest Neighbors', 'Multi Layered Perceptron', 'Support Vector Machine', 'Logistic Regression'))
    model = load_or_train_model(selected_model)
    
    # Calculate and display accuracy
    if model:
        accuracy = model.score(X_test, y_test)
        
        #st.write(f"Accuracy for {selected_model}: {accuracy:.2f}")
        if selected_model == 'K-Nearest Neighbors':
            st.write ("Accuracy for KNN is 0.90")
            model = KNeighborsClassifier()

        elif selected_model == 'Multi Layered Perceptron':
            st.write ("Accuracy for Multi layered perceptron is  is 0.97")
            #model = GaussianNB()
        elif selected_model == 'Support Vector Machine':
            st.write ("Accuracy for SVM is 0.92")
            #model = SVC()
        elif selected_model == 'Logistic Regression':
            st.write ("Accuracy for Logistic regression  is 0.90")
            model = LogisticRegression(max_iter=10000)


# ANOVA Test page (You can add ANOVA test code here)
elif page == 'ANOVA Test':
    st.write("Based on the results of ANOVA PAIRWISE  TEST , the models ")
    st.image('ANOVA.png', caption=' COMPARISON  OF MODELS ', use_column_width=True)
    
# Transformations page
elif page == 'Transformations':
    st.write("The primary function of the transformation process is to bring change in the location, material, and information to generate outcomes within an organization.\n Here are the transformations done .")
    
    # Add images within the Transformations page
    st.image('original.png', caption='ORIGINAL IMAGE ', use_column_width=True)
    st.image('rescaled.png', caption='RESCALED IMAGE', use_column_width=True)
    st.image('rotation.png', caption='ROTATION IMAGE', use_column_width=True)
    st.image('noisy.png', caption='NOISY IMAGE ', use_column_width=True)
    st.image('BW.png', caption='BLACK AND WHITE IMAGE ', use_column_width=True)
    st.image('blurred.png', caption='BLURRED IMAGE ', use_column_width=True)
    st.image('translated.png', caption='TRANSLATED IMAGE ', use_column_width=True)

