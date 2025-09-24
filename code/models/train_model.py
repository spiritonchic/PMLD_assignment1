import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os


def train_and_save_model(train_path='data/processed/train.csv',
                         test_path='data/processed/test.csv',
                         model_path='models/rf_model.pkl',
                         random_state=42):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=['Outcome'])
    y_train = train_df['Outcome']

    X_test = test_df.drop(columns=['Outcome'])
    y_test = test_df['Outcome']

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, preds))

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train_and_save_model()