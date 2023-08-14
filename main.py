import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

def read_data_from_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            values = line.strip().strip(',').split(',')
            data.append(values)
    return data

def check_data(data):
    for idx, row in enumerate(data, 1):
        if len(row) < 1:
            raise ValueError(f"Row {idx}: Each row should have at least 1 element.")
        for col_idx, value in enumerate(row):
            if col_idx != len(row) - 1:  # Exclude the last column (labels)
                try:
                    float(value)
                except ValueError:
                    raise ValueError(f"Row {idx}, Column {col_idx}: Value should be an integer or float.")

def plot_results(y_test, predictions, X_test, model, title_suffix=""):
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix of the classifier ' + title_suffix)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve ' + title_suffix)
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve ' + title_suffix)
    plt.legend(['AUC=%.3f' % roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])])
    plt.show()

    # Feature Importance Plot
    importance = model.feature_importances_
    plt.bar(range(len(importance)), importance)
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance Plot ' + title_suffix)
    plt.show()

if __name__ == "__main__":
    file_path = "C:/Users/long/Downloads/Historial dataset.txt"
    try:
        data = read_data_from_file(file_path)
        check_data(data)
        print("Data is valid.")

        column_names = ['value1', 'value2', 'value3', 'value4', 'value5', 'value6',
                        'value7', 'value8', 'value9', 'value10', 'value11', 'value12', 'label']
        df = pd.DataFrame(data, columns=column_names)

        # Convert all columns to numeric (handle non-numeric values)
        for column in df.columns[:-1]:  # Exclude the 'label' column
            df[column] = pd.to_numeric(df[column], errors='coerce')

        # Convert labels to numeric
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['label'])

        df.dropna(inplace=True)

        print("Number of samples in the dataset:", len(df))

        if len(df) < 14:
            print("Error: Insufficient samples in the dataset.")
            exit(1)

        y = df['label']
        X = df.drop('label', axis=1)

        if len(X) < 2 or len(y) < 2:
            print("Error: Insufficient samples for training or testing.")
            exit(1)

        # Stratify to maintain ratio of labels in both training and testing sets
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Save the LabelEncoder object
        with open("label_encoder.pkl", "wb") as le_file:
            pickle.dump(le, le_file)

        # Prepare testing dataset with labels
        test_file_path = "C:/Users/long/Downloads/Historial testdataset.txt"
        test_data = read_data_from_file(test_file_path)
        check_data(test_data)

        df_test = pd.DataFrame(test_data, columns=column_names)
        for column in df_test.columns[:-1]:  # Exclude the 'label' column
            df_test[column] = pd.to_numeric(df_test[column], errors='coerce')

        # Load the LabelEncoder object for transforming the testing labels
        with open("label_encoder.pkl", "rb") as le_file:
            le = pickle.load(le_file)

        df_test['label'] = le.transform(df_test['label'])
        df_test.dropna(inplace=True)

        X_test = df_test.drop('label', axis=1)
        y_test = df_test['label']

        # Make predictions on test data
        predictions = model.predict(X_test)

        print("Model Accuracy on new data:", accuracy_score(y_test, predictions))

        plot_results(y_test, predictions, X_test, model, title_suffix="on test dataset")

    except Exception as e:
        print("Error:", e)
