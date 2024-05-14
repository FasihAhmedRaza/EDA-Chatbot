import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler



state = {'show_result': False, 'knn_inputs': {}, 'user_inputs': {}}
def main():
    st.title("Hello, World! EDA Streamlit App")
    
    st.header("Upload your CSV data file")
    data_file = st.file_uploader("Upload CSV", type=["csv"])

    data = None  # Initialize data as None

    if data_file is not None:
        data = pd.read_csv(data_file)  # Read the entire dataset
        max_num_rows = len(data)  # Determine the maximum number of rows
        
        st.write("Data overview:")
        st.write(data.head())  # Display the entire dataset
        
        # Slider for selecting the number of rows
        num_rows = st.sidebar.slider("Select number of rows to use", min_value=100, max_value=max_num_rows, value=1000)
        
        st.write("Data overview (first {} rows):".format(num_rows))
        st.write(data.head(num_rows))  # Display the selected number of rows

        st.sidebar.header("Visualizations")
        plot_options = ["Bar plot", "Scatter plot", "Histogram", "Box plot", "Heatmap"]
        selected_plot = st.sidebar.selectbox("Choose a plot type", plot_options)
        selected_columns = []
        if selected_plot in ["Bar plot", "Scatter plot"]:
            selected_columns.append(st.sidebar.selectbox("Select x-axis", data.columns, key="plot_x_axis"))
            selected_columns.append(st.sidebar.selectbox("Select y-axis", data.columns, key="plot_y_axis"))
        if selected_plot == "Histogram":
            selected_columns.append(st.sidebar.selectbox("Select a column", data.columns, key="histogram_column"))

        if selected_plot == "Box plot":
            selected_columns.append(st.sidebar.selectbox("Select a column", data.columns, key="boxplot_column"))

        if selected_plot == "Heatmap":
            st.write("Heatmap:")
            fig, ax = plt.subplots()
            sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
            st.pyplot(fig)

        if st.sidebar.button("Plot"):
            st.write(f"{selected_plot}:")

            plot_requirements = {
                "Bar plot": "Two numeric columns are required. One for the x-axis and one for the y-axis.",
                "Scatter plot": "Two numeric columns are required. One for the x-axis and one for the y-axis.",
                "Histogram": "One numeric column is required.",
                "Box plot": "One numeric column is required."
            }
            st.write(f"Data requirements for {selected_plot}:")
            st.write(plot_requirements[selected_plot])

            if data[selected_columns[0]].dtype == 'object' or data[selected_columns[1]].dtype == 'object':
                st.warning("Please change the data type of selected columns to numeric for plotting.")
            else:
                fig, ax = plt.subplots()
                try:
                    if selected_plot == "Bar plot":
                        sns.barplot(x=data[selected_columns[0]], y=data[selected_columns[1]], ax=ax)
                    elif selected_plot == "Scatter plot":
                        sns.scatterplot(x=data[selected_columns[0]], y=data[selected_columns[1]], ax=ax)
                    elif selected_plot == "Histogram":
                        sns.histplot(data[selected_columns[0]], bins=20, ax=ax)
                    elif selected_plot == "Box plot":
                        sns.boxplot(data[selected_columns[0]], ax=ax)
                    st.pyplot(fig)
                except ValueError as e:
                    st.error(f"An error occurred: {str(e)}. Please check your data and selected columns.")

        display_null_values_and_datatype(data)

        fill_null_values_section(data)
        drop_null_values_section(data)
        rename_column_section(data)
        change_data_type_section(data)
        model_section(data)
        normalization_section(data)


def display_null_values_and_datatype(data):
    null_counts = data.isnull().sum()
    datatypes = data.dtypes
    null_table = pd.DataFrame({'Null Values': null_counts, 'Data Type': datatypes})
    st.sidebar.write("Number of null values and datatype in each column:")
    st.sidebar.table(null_table)


def fill_null_values_section(data):
    st.sidebar.header("Fill Null Values")
    fill_methods = ["Mean", "Median", "Mode", "Most Frequent", "Custom Value"]
    selected_fill_method = st.sidebar.selectbox("Choose a fill method", fill_methods)
    if selected_fill_method == "Custom Value":
        custom_value = st.sidebar.text_input("Enter custom value")
    else:
        custom_value = None

    if st.sidebar.button("Fill Null Values"):
        data = fill_null_values(data, selected_fill_method, custom_value)
        st.sidebar.success("Null values filled successfully.")
        display_null_values_and_datatype(data)


def drop_null_values_section(data):
    st.sidebar.header("Drop Null Values")
    if st.sidebar.button("Drop Null Values"):
        data = drop_null_values(data)
        st.sidebar.success("Null values dropped successfully.")
        display_null_values_and_datatype(data)


def rename_column_section(data):
    data2 = None
    st.sidebar.header("Change Column Name")
    old_column_name = st.sidebar.selectbox("Select a column", data.columns, key="rename_column_select")
    new_column_name = st.sidebar.text_input("Enter new column name", key="rename_new_column_name")
    if st.sidebar.button("Change Column Name"):
        data = rename_column(data, old_column_name, new_column_name)
        st.sidebar.success(f"Column '{old_column_name}' renamed to '{new_column_name}' successfully.")
        st.write("Data overview (after renaming column):")
        st.write(data.head())


def change_data_type_section(data):
    st.sidebar.header("Change Data Type")
    column_to_change = st.sidebar.selectbox("Select a column", data.columns, key="change_datatype_select")
    new_data_type = st.sidebar.selectbox("Select new data type", ["int64", "float64", "object", "datetime64", "bool", "string"], key="change_datatype_new")
    if st.sidebar.button("Change Data Type"):
        data = change_data_type(data, column_to_change, new_data_type)
        st.sidebar.success(f"Data type of column '{column_to_change}' changed to '{new_data_type}' successfully.")
        display_null_values_and_datatype(data)


def model_section(data):
    model_type = st.sidebar.selectbox("Select Model", ["KNN Classification", "Naive Bayes", "Linear Regression", "Logistic Regression", "Decision Tree", "Random Forest"])

    if model_type == "KNN Classification":
        apply_knn_classification(data, state)
    elif model_type == "Naive Bayes":
        apply_naive_bayes(data)
    elif model_type == "Linear Regression":
        apply_linear_regression(data)
    elif model_type == "Logistic Regression":
        apply_logistic_regression(data)
    elif model_type == "Decision Tree":
        apply_decision_tree(data)
    elif model_type == "Random Forest":
        apply_random_forest(data)


def normalization_section(data):
    st.sidebar.header("Normalization")
    normalization_options = ["Standard Scaler", "Min-Max Scaler", "Robust Scaler"]
    selected_normalization = st.sidebar.selectbox("Select normalization technique", normalization_options)

    if selected_normalization:
        selected_columns = st.sidebar.multiselect("Select columns to normalize", data.columns)

        if st.sidebar.button("Normalize Data"):
            if not selected_columns:
                st.sidebar.error("Please select at least one column to normalize.")
            else:
                if selected_normalization == "Standard Scaler":
                    data = standard_scaler_normalization(data, selected_columns)
                    st.sidebar.success("Data normalized using Standard Scaler.")
                elif selected_normalization == "Min-Max Scaler":
                    data = min_max_scaler_normalization(data, selected_columns)
                    st.sidebar.success("Data normalized using Min-Max Scaler.")
                elif selected_normalization == "Robust Scaler":
                    data = robust_scaler_normalization(data, selected_columns)
                    st.sidebar.success("Data normalized using Robust Scaler.")

                st.write("Data overview (after normalization):")
                st.write(data.head())


def standard_scaler_normalization(data, column):
    scaler = StandardScaler()
    data[column] = scaler.fit_transform(data[[column]])
    return data


def min_max_scaler_normalization(data, column):
    scaler = MinMaxScaler()
    data[column] = scaler.fit_transform(data[[column]])
    return data


def robust_scaler_normalization(data, column):
    scaler = RobustScaler()
    data[column] = scaler.fit_transform(data[[column]])
    return data


def fill_null_values(data, method, custom_value=None):
    for column in data.columns:
        if data[column].dtype == 'object':
            if method == "Custom Value" and custom_value is not None:
                data[column].fillna(custom_value, inplace=True)
            elif method == "Most Frequent":
                most_frequent_value = data[column].mode().iloc[0]
                data[column].fillna(most_frequent_value, inplace=True)
        else:
            if method == "Mean":
                data[column].fillna(data[column].mean(), inplace=True)
            elif method == "Median":
                data[column].fillna(data[column].median(), inplace=True)
            elif method == "Mode":
                data[column].fillna(data[column].mode().iloc[0], inplace=True)
            elif method == "Custom Value" and custom_value is not None:
                data[column].fillna(custom_value, inplace=True)
    return data


def drop_null_values(data):
    return data.dropna()


def rename_column(data, old_column_name, new_column_name):
    data.rename(columns={old_column_name: new_column_name}, inplace=True)
    return data


def change_data_type(data, column, new_type):
    if new_type == "datetime64":
        new_type = "datetime64[ns]"  # Correct format for datetime
    elif new_type == "string":
        new_type = "object"  # pandas dtype for string is 'object'

    data[column] = data[column].astype(new_type)
    return data

def apply_knn_classification(data, state):
    if 'show_result' not in state:
        state['show_result'] = False
    if 'knn_inputs' not in state:
        state['knn_inputs'] = {}

    st.header("KNN Classification")

    target_column = st.selectbox("Select the target column", data.columns, key="knn_target_column")
    feature_columns = st.multiselect("Select the feature columns", data.columns, key="knn_feature_columns")
    test_size = st.slider("Select test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.slider("Select random state", min_value=0, max_value=100, value=42, step=1)
    enable_user_inputs = st.radio("Enable user inputs", ["Yes", "No"])

    if enable_user_inputs == "Yes":
        k_value = st.slider("Select the value of k", min_value=1, max_value=10, value=5, step=1)

        # Collect user inputs for each feature column
        user_inputs = {}
        for column in feature_columns:
            user_input = st.number_input(f"Enter value for '{column}'", key=f"knn_input_{column}")
            user_inputs[column] = user_input

        if st.button("Apply KNN Classifier"):
            if len(feature_columns) < 1:
                st.error("Please select at least one feature column.")
                return

            # Store user inputs in state
            state['user_inputs'] = user_inputs

            # Create a DataFrame with user inputs
            user_df = pd.DataFrame([user_inputs])

            # Load the dataset again to ensure consistency in column ordering
            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            # Train KNN model
            knn_model = KNeighborsClassifier(n_neighbors=k_value)
            knn_model.fit(X_train, y_train)

            # Predict the target value for user inputs
            predicted_value = knn_model.predict(user_df)

            # Display the predicted value along with user inputs
            st.write("Predicted Target Value and User Inputs:")
            user_inputs_df = pd.DataFrame([user_inputs])
            predicted_value_df = pd.DataFrame({target_column: predicted_value})
            combined_df = pd.concat([user_inputs_df, predicted_value_df], axis=1)
            st.table(combined_df)

            # Predict the target values for test data
            y_pred = knn_model.predict(X_test)

            # Calculate accuracy on test data if available
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy:.2f}")

            # Display classification report for test data
            st.write("Classification Report:")
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            df_classification_rep = pd.DataFrame(classification_rep).transpose()
            st.table(df_classification_rep)

            # Update state to show the result
            state['show_result'] = True

    elif enable_user_inputs == "No":
        if st.button("Apply KNN Classifier"):
            if len(feature_columns) < 1:
                st.error("Please select at least one feature column.")
                return

            # Load the dataset again to ensure consistency in column ordering
            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            # Train KNN model
            knn_model = KNeighborsClassifier()
            knn_model.fit(X_train, y_train)

            # Predict the target values for test data
            y_pred = knn_model.predict(X_test)

            # Calculate accuracy on test data if available
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy:.2f}")

            # Display classification report for test data
            st.write("Classification Report:")
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            df_classification_rep = pd.DataFrame(classification_rep).transpose()
            st.table(df_classification_rep)

            # Update state to show the result
            state['show_result'] = True





def apply_naive_bayes(data):
    st.header("Naive Bayes")
    
    target_column = st.selectbox("Select the target column", data.columns, key="nb_target_column")
    feature_columns = st.multiselect("Select the feature columns", data.columns, key="nb_feature_columns")
    test_size = st.slider("Select test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.slider("Select random state", min_value=0, max_value=100, value=42, step=1)
    enable_user_inputs = st.radio("Enable user inputs", ["Yes", "No"])

    if enable_user_inputs == "Yes":
        # Collect user inputs for each feature column
        user_inputs = {}
        for column in feature_columns:
            user_input = st.number_input(f"Enter value for '{column}'", key=f"nb_input_{column}")
            user_inputs[column] = user_input

        if st.button("Apply Naive Bayes"):
            if len(feature_columns) < 1:
                st.error("Please select at least one feature column.")
                return

            # Create a DataFrame with user inputs
            user_df = pd.DataFrame([user_inputs])

            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            nb_model = GaussianNB()
            nb_model.fit(X_train, y_train)

            y_pred = nb_model.predict(user_df)

            # Display the predicted value along with user inputs
            st.write("Predicted Target Value and User Inputs:")
            user_inputs_df = pd.DataFrame([user_inputs])
            predicted_value_df = pd.DataFrame({target_column: y_pred})
            combined_df = pd.concat([user_inputs_df, predicted_value_df], axis=1)
            st.table(combined_df)

            # Predict the target values for test data
            y_pred_test = nb_model.predict(X_test)

            # Calculate accuracy on test data if available
            accuracy = accuracy_score(y_test, y_pred_test)
            st.write(f"Accuracy on Test Data: {accuracy:.2f}")

            # Display classification report for test data
            st.write("Classification Report on Test Data:")
            classification_rep = classification_report(y_test, y_pred_test, output_dict=True)
            df_classification_rep = pd.DataFrame(classification_rep).transpose()
            st.table(df_classification_rep)

    elif enable_user_inputs == "No":
        if st.button("Apply Naive Bayes"):
            if len(feature_columns) < 1:
                st.error("Please select at least one feature column.")
                return

            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            nb_model = GaussianNB()
            nb_model.fit(X_train, y_train)

            y_pred = nb_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy:.2f}")

            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            df_classification_rep = pd.DataFrame(classification_rep).transpose()
            st.write("Classification Report:")
            st.table(df_classification_rep)


def apply_linear_regression(data):
    st.header("Linear Regression")

    target_column = st.selectbox("Select the target column", data.columns, key="linear_target_column")
    feature_columns = st.multiselect("Select the feature columns", data.columns, key="linear_feature_columns")
    test_size = st.slider("Select test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.slider("Select random state", min_value=0, max_value=100, value=42, step=1)
    enable_user_inputs = st.radio("Enable user inputs", ["Yes", "No"])

    if enable_user_inputs == "Yes":
        # Collect user inputs for each feature column
        user_inputs = {}
        for column in feature_columns:
            user_input = st.number_input(f"Enter value for '{column}'", key=f"linear_input_{column}")
            user_inputs[column] = user_input

        if st.button("Apply Linear Regression"):
            if len(feature_columns) < 1:
                st.error("Please select at least one feature column.")
                return

            # Create a DataFrame with user inputs
            user_df = pd.DataFrame([user_inputs])

            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)

            y_pred = linear_model.predict(user_df)

            st.write("Predicted Target Value based on User Inputs:")
            st.write(y_pred)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"Mean Squared Error on Test Data: {mse:.2f}")
            st.write(f"R-squared on Test Data: {r2:.2f}")

    elif enable_user_inputs == "No":
        if st.button("Apply Linear Regression"):
            if len(feature_columns) < 1:
                st.error("Please select at least one feature column.")
                return

            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)

            y_pred = linear_model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"Mean Squared Error on Test Data: {mse:.2f}")
            st.write(f"R-squared on Test Data: {r2:.2f}")



def apply_logistic_regression(data):
    st.header("Logistic Regression")

    target_column = st.selectbox("Select the target column", data.columns, key="logistic_target_column")
    feature_columns = st.multiselect("Select the feature columns", data.columns, key="logistic_feature_columns")
    test_size = st.slider("Select test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.slider("Select random state", min_value=0, max_value=100, value=42, step=1)
    enable_user_inputs = st.radio("Enable user inputs", ["Yes", "No"])

    if enable_user_inputs == "Yes":
        # Collect user inputs for each feature column
        user_inputs = {}
        for column in feature_columns:
            user_input = st.number_input(f"Enter value for '{column}'", key=f"logistic_input_{column}")
            user_inputs[column] = user_input

        if st.button("Apply Logistic Regression"):
            if len(feature_columns) < 1:
                st.error("Please select at least one feature column.")
                return

            # Create a DataFrame with user inputs
            user_df = pd.DataFrame([user_inputs])

            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            logistic_model = LogisticRegression()
            logistic_model.fit(X_train, y_train)

            y_pred = logistic_model.predict(user_df)

            st.write("Predicted Target Value based on User Inputs:")
            st.write(y_pred)

            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy on Test Data: {accuracy:.2f}")

            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            df_classification_rep = pd.DataFrame(classification_rep).transpose()
            st.write("Classification Report on Test Data:")
            st.table(df_classification_rep)

    elif enable_user_inputs == "No":
        if st.button("Apply Logistic Regression"):
            if len(feature_columns) < 1:
                st.error("Please select at least one feature column.")
                return

            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            logistic_model = LogisticRegression()
            logistic_model.fit(X_train, y_train)

            y_pred = logistic_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy on Test Data: {accuracy:.2f}")

            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            df_classification_rep = pd.DataFrame(classification_rep).transpose()
            st.write("Classification Report on Test Data:")
            st.table(df_classification_rep)


def apply_decision_tree(data):
    st.header("Decision Tree")

    target_column = st.selectbox("Select the target column", data.columns, key="dt_target_column")
    feature_columns = st.multiselect("Select the feature columns", data.columns, key="dt_feature_columns")
    test_size = st.slider("Select test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.slider("Select random state", min_value=0, max_value=100, value=42, step=1)
    enable_user_inputs = st.radio("Enable user inputs", ["Yes", "No"])

    if enable_user_inputs == "Yes":
        # Collect user inputs for each feature column
        user_inputs = {}
        for column in feature_columns:
            user_input = st.number_input(f"Enter value for '{column}'", key=f"dt_input_{column}")
            user_inputs[column] = user_input

        if st.button("Apply Decision Tree"):
            if len(feature_columns) < 1:
                st.error("Please select at least one feature column.")
                return

            # Create a DataFrame with user inputs
            user_df = pd.DataFrame([user_inputs])

            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            dt_model = DecisionTreeClassifier()
            dt_model.fit(X_train, y_train)

            y_pred = dt_model.predict(user_df)

            st.write("Predicted Target Value based on User Inputs:")
            st.write(y_pred)

            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy on Test Data: {accuracy:.2f}")

            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            df_classification_rep = pd.DataFrame(classification_rep).transpose()
            st.write("Classification Report on Test Data:")
            st.table(df_classification_rep)

    elif enable_user_inputs == "No":
        if st.button("Apply Decision Tree"):
            if len(feature_columns) < 1:
                st.error("Please select at least one feature column.")
                return

            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            dt_model = DecisionTreeClassifier()
            dt_model.fit(X_train, y_train)

            y_pred = dt_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy on Test Data: {accuracy:.2f}")

            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            df_classification_rep = pd.DataFrame(classification_rep).transpose()
            st.write("Classification Report on Test Data:")
            st.table(df_classification_rep)



def apply_random_forest(data):
    st.sidebar.header("Random Forest")

    target_column = st.sidebar.selectbox("Select the target column", data.columns, key="rf_target_column")
    feature_columns = st.sidebar.multiselect("Select the feature columns", data.columns, key="rf_feature_columns")
    test_size = st.sidebar.slider("Select test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.sidebar.slider("Select random state", min_value=0, max_value=100, value=42, step=1)
    n_estimators = st.sidebar.slider("Select the number of estimators", min_value=1, max_value=100, value=10, step=1)

    if st.sidebar.button("Apply Random Forest"):
        if len(feature_columns) < 1:
            st.sidebar.error("Please select at least one feature column.")
            return

        X = data[feature_columns]
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        rf_model = RandomForestClassifier(n_estimators=n_estimators)
        rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")

        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        df_classification_rep = pd.DataFrame(classification_rep).transpose()

        # Convert the "support" column to integers
        df_classification_rep["support"] = df_classification_rep["support"].astype(int)

        st.write("Classification Report:")
        st.table(df_classification_rep)

if __name__ == "__main__":
    main()
