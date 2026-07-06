import pandas as pd
import io


def load_data(file, filename=None):
    if filename is None:
        filename = getattr(file, "name", getattr(file, "filename", ""))
    file_name = filename.lower()

    if file_name.endswith(".csv"):
        # Read raw bytes once so we can retry with different encodings
        raw_bytes = file.read() if hasattr(file, 'read') else file
        # Try multiple encodings gracefully
        for encoding in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
            try:
                df = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
                return df
            except (UnicodeDecodeError, Exception):
                continue
        raise ValueError("Could not decode the CSV file. Try saving it as UTF-8.")
    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        df = pd.read_excel(file)
        return df
    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel.")


def get_basic_info(df):
    rows, cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    memory_usage_kb = round(df.memory_usage(deep=True).sum() / 1024, 2)

    return {
        "rows": rows,
        "columns": cols,
        "duplicate_rows": duplicate_rows,
        "memory_usage_kb": memory_usage_kb
    }


def get_column_names(df):
    return list(df.columns)


def get_data_types(df):
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })
    return dtype_df


def get_null_info(df):
    total_rows = len(df)

    if total_rows == 0:
        null_percentage = [0.0] * len(df.columns)
    else:
        null_percentage = ((df.isnull().sum() / total_rows) * 100).round(2).values

    null_df = pd.DataFrame({
        "Column": df.columns,
        "Null Count": df.isnull().sum().values.astype(int),
        "Null Percentage": [float(x) for x in null_percentage]
    }).sort_values(by="Null Count", ascending=False)

    return null_df


def detect_column_types(df):
    numerical_cols = []
    categorical_cols = []
    text_cols = []
    datetime_cols = []
    boolean_cols = []
    id_like_cols = []

    total_rows = len(df)

    for col in df.columns:
        series = df[col]
        non_null_series = series.dropna()

        # Skip fully empty columns for type heuristics
        if non_null_series.empty:
            categorical_cols.append(col)
            continue

        # Boolean detection
        if pd.api.types.is_bool_dtype(series):
            boolean_cols.append(col)
            continue

        unique_non_null = non_null_series.nunique(dropna=True)
        unique_ratio = unique_non_null / max(total_rows, 1)

        # Numeric detection
        if pd.api.types.is_numeric_dtype(series):
            # Treat true binary numeric columns as boolean-like
            if unique_non_null <= 2:
                boolean_cols.append(col)
            # ID-like only if highly unique and not very small-cardinality numeric
            elif unique_ratio > 0.95:
                id_like_cols.append(col)
            else:
                numerical_cols.append(col)
            continue

        # Datetime detection
        try:
            parsed = pd.to_datetime(non_null_series, errors="coerce")
            if len(non_null_series) > 0 and (parsed.notna().sum() / len(non_null_series)) >= 0.7:
                datetime_cols.append(col)
                continue
        except Exception:
            pass

        # Object / string columns
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            avg_len = non_null_series.astype(str).map(len).mean()

            if unique_ratio > 0.95:
                id_like_cols.append(col)
            elif avg_len > 40:
                text_cols.append(col)
            else:
                categorical_cols.append(col)
            continue

        # Fallback
        categorical_cols.append(col)

    return {
        "numerical": numerical_cols,
        "categorical": categorical_cols,
        "text": text_cols,
        "datetime": datetime_cols,
        "boolean": boolean_cols,
        "id_like": id_like_cols
    }


def get_dataset_profile(df, column_types):
    rows, cols = df.shape

    total_cells = rows * cols
    total_missing = int(df.isnull().sum().sum())
    missing_percentage = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0

    numerical_count = len(column_types["numerical"])
    categorical_count = len(column_types["categorical"])
    text_count = len(column_types["text"])
    datetime_count = len(column_types["datetime"])
    boolean_count = len(column_types["boolean"])
    id_like_count = len(column_types["id_like"])

    if rows < 1000:
        size_label = "Small"
    elif rows < 50000:
        size_label = "Medium"
    else:
        size_label = "Large"

    high_cardinality_cols = []
    for col in column_types["categorical"]:
        if df[col].dropna().nunique() > 20:
            high_cardinality_cols.append(col)

    return {
        "rows": rows,
        "columns": cols,
        "size_label": size_label,
        "total_missing": total_missing,
        "missing_percentage": missing_percentage,
        "numerical_count": numerical_count,
        "categorical_count": categorical_count,
        "text_count": text_count,
        "datetime_count": datetime_count,
        "boolean_count": boolean_count,
        "id_like_count": id_like_count,
        "high_cardinality_cols": high_cardinality_cols
    }


def get_target_analysis(df, target_col):
    if target_col not in df.columns:
        return {
            "target_found": False,
            "unique_values": None,
            "class_balance": None,
            "imbalance_ratio": None,
            "is_imbalanced": False
        }

    target_series = df[target_col].dropna()
    unique_values = target_series.nunique(dropna=True)

    if unique_values <= 20 and len(target_series) > 0:
        value_counts = target_series.value_counts(normalize=True) * 100
        class_balance = value_counts.round(2).to_dict()

        max_pct = value_counts.max()
        min_pct = value_counts.min()
        imbalance_ratio = round(max_pct / min_pct, 2) if min_pct > 0 else None

        is_imbalanced = min_pct < 20

        return {
            "target_found": True,
            "unique_values": unique_values,
            "class_balance": class_balance,
            "imbalance_ratio": imbalance_ratio,
            "is_imbalanced": is_imbalanced
        }

    return {
        "target_found": True,
        "unique_values": unique_values,
        "class_balance": None,
        "imbalance_ratio": None,
        "is_imbalanced": False
    }