import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (KFold, cross_val_score, train_test_split,
                                     ShuffleSplit, LeaveOneOut)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, 
                             classification_report, roc_auc_score, roc_curve, 
                             mean_squared_error, log_loss, accuracy_score)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import joblib
import io

# Function to load and preprocess the Lung Cancer dataset
@st.cache_data
def load_lung_cancer_data(uploaded_file):
    dataframe = pd.read_csv(uploaded_file)
    
    # Impute missing values for numeric data (mean strategy)
    numeric_imputer = SimpleImputer(strategy='mean')
    dataframe.iloc[:, :-1] = numeric_imputer.fit_transform(dataframe.iloc[:, :-1])

    return dataframe

# Function to load and preprocess the Chicago Air Pollution dataset
@st.cache_data
def load_air_pollution_data(uploaded_file):
    dataframe = pd.read_csv(uploaded_file)
    
    # Drop the 'id' column as it is not needed for modeling
    dataframe = dataframe.drop(columns=['id'])

    # Impute missing values for numeric data (mean strategy)
    numeric_imputer = SimpleImputer(strategy='mean')
    dataframe.iloc[:, :-1] = numeric_imputer.fit_transform(dataframe.iloc[:, :-1])

    return dataframe

# Function for user input in Lung Cancer model
def get_lung_cancer_input():
    # Assuming the dataset has the following columns
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    smoking = st.selectbox("Smoking (1: No, 2: Yes)", [1,2])
    yellow_fingers = st.selectbox("Yellow Fingers (1: No, 2: Yes)", [1,2])
    anxiety = st.selectbox("Anxiety (1: No, 2: Yes)", [1,2])
    peer_pressure = st.selectbox("Peer Pressure (1: No, 2: Yes)", [1,2])
    chronic_disease = st.selectbox("Chronic Disease (1: No, 2: Yes)", [1,2])
    fatigue = st.selectbox("Fatigue (1: No, 2: Yes)", [1,2])
    allergy = st.selectbox("Allergy (1: No, 2: Yes)", [1,2])
    wheezing = st.selectbox("Wheezing (1: No, 2: Yes)", [1,2])
    alcohol_consumption = st.selectbox("Alcohol Consumption (1: No, 2: Yes)", [1,2])
    coughing = st.selectbox("Coughing (1: No, 2: Yes)", [1,2])
    shortness_of_breath = st.selectbox("Shortness of Breath (1: No, 2: Yes)", [1,2])
    swallowing_difficulty = st.selectbox("Swallowing Difficulty (1: No, 2: Yes)", [1,2])
    chest_pain = st.selectbox("Chest Pain (1: No, 2: Yes)", [1,2])

    features = np.array([[age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, 
                          allergy, wheezing, alcohol_consumption, coughing, shortness_of_breath, swallowing_difficulty, 
                          chest_pain]])
    return features


# Function for user input in Air Pollution model
def get_air_pollution_input():
    # Input features based on the columns in chicagoairpollution.csv
    tmpd = st.number_input("Temperature (tmpd)", min_value=-50.0, value=20.0)
    dptp = st.number_input("Dew Point (dptp)", min_value=-50.0, value=10.0)
    pm25tmean2 = st.number_input("PM2.5 Mean (pm25tmean2: Fine Particulate Matter Reading)", min_value=0.0, value=10.0)
    pm10tmean2 = st.number_input("PM10 Mean (pm10tmean2: Fine Particulate Matter Reading)", min_value=0.0, value=20.0)
    no2tmean2 = st.number_input("NO2 Mean (Nitrogen Dioxide mean)", min_value=0.0, value=15.0)

    features = np.array([[tmpd, dptp, pm25tmean2, pm10tmean2, no2tmean2]])
    return features


# Main app
def main():
    st.title("Model Selection and Evaluation")

    # Add a sidebar for model selection
    model_choice = st.sidebar.selectbox("Choose the Model", ("Lung Cancer Prediction Model", "Air Pollution Regression Model"))

    if model_choice == "Lung Cancer Prediction Model":
        # Upload file
        uploaded_file = st.file_uploader("Upload your Lung Cancer CSV file", type=["csv"])

        if uploaded_file is not None:
            # Load the dataset
            st.write("Loading the dataset...")
            dataframe = load_lung_cancer_data(uploaded_file)

            # Display the first few rows of the dataset
            st.subheader("Dataset Preview")
            st.write(dataframe.head())

            # Tabs for Lung Cancer model evaluation
            tabs = st.tabs(["K-fold Cross Validation", "Leave-One-Out Cross Validation", "Prediction"])

            with tabs[0]:
                st.subheader("K-fold Cross Validation")
                array = dataframe.values
                X = array[:, :-1]  # Features
                Y = array[:, -1]   # Target variable
                num_folds = st.slider("Select number of folds for KFold Cross Validation:", 2, 10, 5)
                kfold = KFold(n_splits=num_folds)
                model = LogisticRegression(max_iter=210)
                results = cross_val_score(model, X, Y, cv=kfold)
                st.write(f"Accuracy: {results.mean() * 100:.3f}%")
                st.write(f"Standard Deviation: {results.std() * 100:.3f}%")

                # Metrics Calculation
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
                model.fit(X_train, Y_train)
                Y_prob = model.predict_proba(X_test)[:, 1]
                
                # Classification Metrics
                st.subheader("Classification Metrics")
                st.write("Confusion Matrix:")
                predicted = model.predict(X_test)
                matrix = confusion_matrix(Y_test, predicted)

                # Create the figure and axes using subplots
                fig, ax = plt.subplots()
                disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
                disp.plot(cmap=plt.cm.Blues, ax=ax)

                # Render the confusion matrix plot
                st.pyplot(fig)

                st.write("Classification Report:")
                report = classification_report(Y_test, predicted, output_dict=True)
                st.write(report)

                st.write(f"ROC AUC Score: {roc_auc_score(Y_test, Y_prob):.3f}")

               # Plot ROC Curve
                st.write("ROC Curve")
                fpr, tpr, _ = roc_curve(Y_test, Y_prob)

                # Create the figure and axes
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc_score(Y_test, Y_prob):.2f})')
                ax.plot([0, 1], [0, 1], color='red', linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic')
                ax.legend(loc='lower right')

                # Render the ROC curve plot
                st.pyplot(fig)

                st.write(f"Logarithmic Loss: {log_loss(Y_test, Y_prob):.3f}")

                # Classification Accuracy
                accuracy = accuracy_score(Y_test, predicted)
                st.write(f"Classification Accuracy: {accuracy * 100:.3f}%")

                # Save the trained model
                model_filename = "lung_cancer_kfold_model.pkl"
                joblib.dump(model, model_filename)
                with open(model_filename, "rb") as f:
                    st.download_button("Download Trained Model", f, file_name=model_filename)

            with tabs[1]:
                st.subheader("Leave-One-Out Cross Validation (LOOCV)")
                loocv = LeaveOneOut()
                model = LogisticRegression(max_iter=500)
                results = cross_val_score(model, X, Y, cv=loocv)
                st.write(f"Accuracy: {results.mean() * 100:.3f}%")
                st.write(f"Standard Deviation: {results.std() * 100:.3f}%")

                # Metrics Calculation
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
                model.fit(X_train, Y_train)
                Y_prob = model.predict_proba(X_test)[:, 1]
                
                # Classification Metrics
                st.subheader("Classification Metrics")
                st.write("Confusion Matrix:")
                predicted = model.predict(X_test)
                matrix = confusion_matrix(Y_test, predicted)
                st.write(matrix)
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay(confusion_matrix=matrix).plot(cmap=plt.cm.Blues, ax=ax)
                st.pyplot(fig)

                st.write("Classification Report:")
                report = classification_report(Y_test, predicted, output_dict=True)
                st.write(report)

                st.write(f"ROC AUC Score: {roc_auc_score(Y_test, Y_prob):.3f}")

                # Plot ROC Curve
                fpr, tpr, _ = roc_curve(Y_test, Y_prob)
                plt.figure()
                plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc_score(Y_test, Y_prob))
                plt.plot([0, 1], [0, 1], color='red', linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc='lower right')
                st.pyplot()

                st.write(f"Logarithmic Loss: {log_loss(Y_test, Y_prob):.3f}")

                # Classification Accuracy
                accuracy = accuracy_score(Y_test, predicted)
                st.write(f"Classification Accuracy: {accuracy * 100:.3f}%")

                # Save the trained model
                model_filename = "lung_cancer_loocv_model.pkl"
                joblib.dump(model, model_filename)
                with open(model_filename, "rb") as f:
                    st.download_button("Download Trained Model", f, file_name=model_filename)

            with tabs[2]:
                st.subheader("Prediction")
                st.write("Upload your trained model for prediction:")
                uploaded_model = st.file_uploader("Upload your trained model file", type=["pkl"])
                user_data = get_lung_cancer_input()
                if uploaded_model is not None:
                    loaded_model = joblib.load(uploaded_model)
                    prediction = loaded_model.predict(user_data)
                    st.write("Prediction Result:")
                    st.write("No Lung Cancer" if prediction[0] == 0 else "Lung Cancer Present")

    elif model_choice == "Air Pollution Regression Model":
        # Upload file
        uploaded_file = st.file_uploader("Upload your Chicago Air Pollution CSV file", type=["csv"])

        if uploaded_file is not None:
            # Load the dataset
            st.write("Loading the dataset...")
            dataframe = load_air_pollution_data(uploaded_file)

            # Display the first few rows of the dataset
            st.subheader("Dataset Preview")
            st.write(dataframe.head())

            # Tabs for Air Pollution model evaluation
            tabs = st.tabs(["Train-Test Split", "Repeated Random Test-Train Splits", "Prediction"])

            with tabs[0]:
                st.subheader("Split into Train and Test Sets")
                test_size = st.slider("Test size (as a percentage)", 10, 50, 20) / 100
                X = dataframe.drop('o3tmean2', axis=1).values
                Y = dataframe['o3tmean2'].values
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
                model = LinearRegression()
                model.fit(X_train, Y_train)
                result = model.score(X_test, Y_test)
                st.write(f"R-squared: {result:.3f}")

                # Regression Metrics
                st.subheader("Regression Metrics")
                Y_pred = model.predict(X_test)
                mse = mean_squared_error(Y_test, Y_pred)
                st.write(f"Mean Squared Error (MSE): {mse:.3f}")

                mae = np.mean(np.abs(Y_test - Y_pred))
                st.write(f" Mean Absolute Error (MAE): {mae:.3f}")

                r2 = model.score(X_test, Y_test)
                st.write(f"R-squared: {r2:.3f}")

                # Save the trained model
                model_filename = "air_pollution_splitmodel.pkl"
                joblib.dump(model, model_filename)
                with open(model_filename, "rb") as f:
                    st.download_button("Download Trained Model", f, file_name=model_filename)

            with tabs[1]:
                st.subheader("Repeated Random Test-Train Splits")
                n_splits = st.slider("Select number of splits:", 2, 20, 10)
                test_size = st.slider("Select test size proportion:", 0.1, 0.5, 0.33)
                shuffle_split = ShuffleSplit(n_splits=n_splits, test_size=test_size)

                # Fit the model and evaluate
                model = LinearRegression()
                results = cross_val_score(model, X, Y, cv=shuffle_split)
                st.write(f"Mean R-squared: {results.mean():.3f}")
                st.write(f"Standard Deviation: {results.std():.3f}")

                # Fit the model on the complete dataset for regression metrics
                model.fit(X, Y)
                Y_pred = model.predict(X_test)

                # Regression Metrics
                st.subheader("Regression Metrics")
                mse = mean_squared_error(Y_test, Y_pred)
                st.write(f"Mean Squared Error (MSE): {mse:.3f}")

                mae = np.mean(np.abs(Y_test - Y_pred))
                st.write(f" Mean Absolute Error (MAE): {mae:.3f}")

                r2 = model.score(X_test, Y_test)
                st.write(f"R-squared: {r2:.3f}")

                # Save the trained model
                model_filename = "air_pollution_repeatedmodel.pkl"
                joblib.dump(model, model_filename)
                with open(model_filename, "rb") as f:
                    st.download_button("Download Trained Model", f, file_name=model_filename)

            with tabs[2]:
                st.subheader("Prediction")
                st.write("Upload your trained model for prediction:")
                uploaded_model = st.file_uploader("Upload your trained model file", type=["pkl"])
                user_data = get_air_pollution_input()
                if uploaded_model is not None:
                    loaded_model = joblib.load(uploaded_model)
                    prediction = loaded_model.predict(user_data)
                    st.write(f"Predicted Ozone Level (O3tmean2): {prediction[0]:.3f}")

if __name__ == "__main__":
    main()
