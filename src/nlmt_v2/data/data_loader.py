import pandas as pd
import numpy as np

def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """Loads and preprocesses the data from the given CSV path."""
    df = pd.read_csv(csv_path)
    languages = set(col.split(" ", 1)[1] for col in df.columns[1:])
    merged_data = pd.DataFrame(index=df.index)

    for lang in languages:
        merged_data[lang] = 0
        for prefix in ["Studying", "Fluent", "Native"]:
            col = f"{prefix} {lang}"
            if col in df.columns:
                merged_data[lang] |= df[col].apply(lambda x: 1 if x > 0 else 0)

    return merged_data.astype(np.float32)
