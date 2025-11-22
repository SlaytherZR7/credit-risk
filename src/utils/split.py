from pathlib import Path
from sklearn.model_selection import train_test_split


def split_and_save(
    df,
    target_col,
    output_dir="data/processed",
    test_size=0.2,
    random_state=42
):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    X_train.to_parquet(output_path / "X_train.parquet")
    X_test.to_parquet(output_path / "X_test.parquet")
    y_train.to_frame(name=target_col).to_parquet(output_path / "y_train.parquet")
    y_test.to_frame(name=target_col).to_parquet(output_path / "y_test.parquet")

    print(f"✅ Guardado en {output_path} (train: {X_train.shape}, test: {X_test.shape})")

    return X_train, X_test, y_train, y_test
