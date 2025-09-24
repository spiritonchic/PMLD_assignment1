import pandas as pd
from sklearn.model_selection import train_test_split
import os


def process_pima_dataset(input_path='data/raw/diabetes.csv',
                         train_path='data/processed/train.csv',
                         test_path='data/processed/test.csv',
                         test_size=0.2,
                         random_state=42):

    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    df = pd.read_csv(input_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"Filled missing values in column '{col}' with median {median_value}")

    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Processed data saved: {train_path} ({train_df.shape[0]} rows), {test_path} ({test_df.shape[0]} rows)")


if __name__ == "__main__":
    process_pima_dataset()