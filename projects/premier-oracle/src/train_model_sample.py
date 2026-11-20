from sklearn.base import clone
from lightgbm import LGBMClassifier

def train_production_models(df, preprocessor_template, target_col='Target'):
    """
    Implements a 'Master Preprocessor' strategy and chronological validation.
    Ensures zero data leakage by never shuffling matches out of time order.
    """
    # 1. CHRONOLOGICAL SPLIT (No random shuffling in time-series/sports)
    split_idx = int(len(df) * 0.85)
    train_data = df.iloc[:split_idx]
    test_data = df.iloc[split_idx:]

    X_train, y_train = train_data.drop(columns=[target_col]), train_data[target_col]
    X_test, y_test = test_data.drop(columns=[target_col]), test_data[target_col]

    # 2. MASTER PREPROCESSOR (Fit strictly on training history to avoid leakage)
    master_preprocessor = clone(preprocessor_template)
    X_train_transformed = master_preprocessor.fit_transform(X_train)
    X_test_transformed = master_preprocessor.transform(X_test)

    # 3. ISOLATED TRAINING
    model = LGBMClassifier(n_estimators=600, learning_rate=0.03)
    model.fit(X_train_transformed, y_train)

    return model, master_preprocessor, (X_test_transformed, y_test)