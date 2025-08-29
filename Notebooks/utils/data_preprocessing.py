import pandas as pd
import numpy as np
from functools import reduce

def align_views_on_index(views, index_col='Local_Authority_Code'):
    """
    Cleans and aligns a dictionary of pandas DataFrames on a common index for multi-view learning.

    This function performs the following for each view:
        1. Renames standard geographic identifier columns:
            - 'Area' → 'Local_Authority_Code'
            - 'Area Level' → 'Local_Authority_Name'
        2. Drops non-numeric metadata columns such as 'Local_Authority_Name'
        3. Replaces string placeholders for missing values (e.g. 'na', 'NA') with np.nan
        4. Coerces all remaining columns to numeric type (float), invalid entries become np.nan
        5. Sets the cleaned 'Local_Authority_Code' column as the index
        6. Drops duplicated or missing index entries
        7. Computes the intersection of indices across all views to ensure row alignment
        8. Returns views subset to the common index with sorted row order

    Parameters
    ----------
    views : dict of {str: pd.DataFrame}
        Dictionary where each key is the view name and the value is a DataFrame representing one view.

    index_col : str, default='Local_Authority_Code'
        The column to use as the index for alignment. This column is renamed from 'Area' if present.

    Returns
    -------
    aligned_views : dict of {str: pd.DataFrame}
        Dictionary of cleaned and aligned DataFrames. All returned DataFrames:
            - Have identical, sorted indices
            - Contain only numeric features
            - Are safe to pass into modeling pipelines (e.g., UMAP, KMeans)

    Raises
    ------
    AssertionError
        If any pair of aligned views do not have exactly the same index after processing.

    Notes
    -----
    - This function is intended for preprocessing tabular views of different aspects of the same entities
      (e.g., local authorities).
    - It ensures compatibility with multi-view learning strategies such as early, intermediate, or late integration.

    Examples
    --------
    >>> aligned = align_views_on_index({
    ...     'health': health_df,
    ...     'economic': economic_df,
    ...     'skills': skills_df
    ... })

    >>> aligned['health'].head()
    """
    cleaned_views = {}

    for name, df in views.items():
        df = df.copy()

        # Step 1: Rename standard columns
        df = df.rename(columns={
            'Area Code': 'Local_Authority_Code',
            'Area ': 'Local_Authority_Name',
            'Area': 'Local_Authority_Name'
        })

        # Step 2: Drop non-numeric metadata columns
        df = df.drop(columns=['Local_Authority_Name', 'Area Level'], errors='ignore')

        # Step 3: Clean and set index
        if index_col not in df.columns:
            raise KeyError(f"Expected index column '{index_col}' not found in view '{name}'.")

        # Drop duplicate columns if they exist
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

        # Ensure the index column is a Series
        index_series = df[index_col]
        if isinstance(index_series, pd.DataFrame):
            index_series = index_series.iloc[:, 0]  # take first occurrence if duplicated

        df[index_col] = index_series.astype(str).str.strip()
        df = df.dropna(subset=[index_col])
        df = df.set_index(index_col)
        df = df[~df.index.duplicated(keep='first')]

        # Step 4: Clean values
        df = df.replace(r'(?i)^na$', np.nan, regex=True)  # string 'na', 'NA', etc. to np.nan
        df = df.apply(pd.to_numeric, errors='coerce')    # convert all to numeric, coerce bad values

        cleaned_views[name] = df

    # Step 5: Align all views to the common index
    common_index = reduce(lambda x, y: x.intersection(y), [df.index for df in cleaned_views.values()])
    common_index = sorted(common_index)

    aligned_views = {
        name: df.loc[common_index].sort_index()
        for name, df in cleaned_views.items()
    }

    # Step 6: Sanity check
    indices = list(aligned_views.values())
    for i in range(len(indices) - 1):
        assert (indices[i].index == indices[i + 1].index).all(), \
            f"Index mismatch between views: {list(aligned_views.keys())[i]} and {list(aligned_views.keys())[i+1]}"

    print("All views aligned to", len(common_index), "entities")
    for name, df in aligned_views.items():
        print(f"  - {name}: shape = {df.shape}, dtype(s) = {df.dtypes.unique()}")

    return aligned_views

from typing import Dict, Iterable, Tuple, Optional

# Sensible defaults; override via function args if your project defines these elsewhere.
DEFAULT_META_COLS = {"Local_Authority_Name", "Area Level"}  # non-numeric, to drop
DEFAULT_WINSOR_QUANTILES: Tuple[float, float] = (0.01, 0.99)
DEFAULT_ONS_EXCLUSIONS: Dict[str, Iterable[str]] = {}  # e.g., {"health": ["E09000001", ...]}

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bring view DataFrames to a standard shape:
      - Rename 'Area Code' -> 'Local_Authority_Code'
      - Rename 'Area' or 'Area ' -> 'Local_Authority_Name'
      - Drop duplicate columns
    """
    df = df.rename(columns={
        "Area Code": "Local_Authority_Code",
        "Area ": "Local_Authority_Name",
        "Area": "Local_Authority_Name",
    })
    # remove duplicated column names if any
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df

def preprocess_view(
    df: pd.DataFrame,
    meta_cols: Optional[Iterable[str]] = None,
    winsor_quantiles: Tuple[float, float] = DEFAULT_WINSOR_QUANTILES,
    drop_rows_with_na: bool = True,    # NEW
) -> pd.DataFrame:
    """
    Clean a single view:
      1) Standardize columns and identify feature columns (exclude meta)
      2) Replace 'na'/'NA'/etc with NaN and coerce features to numeric
      3) Winsorize numeric features at given quantiles
      4) Drop rows with any missing values in features

    Parameters
    ----------
    df : pd.DataFrame
    meta_cols : iterable of str, optional
        Columns to exclude from numeric feature processing. Defaults to DEFAULT_META_COLS plus index col.
    winsor_quantiles : (low, high)
        Quantiles for clipping, default (0.01, 0.99).
    """
    df = _standardize_columns(df.copy())

    # Build meta set and feature list
    meta_cols = set(meta_cols) if meta_cols is not None else set(DEFAULT_META_COLS)
    # Ensure the identifier stays out of features
    meta_cols = meta_cols.union({"Local_Authority_Code"})

    feature_cols = [c for c in df.columns if c not in meta_cols and c != "Local_Authority_Code"]

    # Clean missing strings and convert to numeric
    df[feature_cols] = df[feature_cols].replace(r"(?i)^na$", np.nan, regex=True)
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Winsorize
    low_q, high_q = winsor_quantiles
    for col in feature_cols:
        if col in df and df[col].notna().any():
            lower = df[col].quantile(low_q)
            upper = df[col].quantile(high_q)
            df[col] = df[col].clip(lower, upper)

    if drop_rows_with_na:
        df = df.dropna(subset=feature_cols)
    return df


def exclude_las(
    df: pd.DataFrame,
    view_name: str,
    ons_exclusions: Optional[Dict[str, Iterable[str]]] = None,
) -> pd.DataFrame:
    """
    Exclude local authorities by code for a given view.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Local_Authority_Code' or 'Area Code' (will be standardized).
    view_name : str
        View key used to look up exclusion codes.
    ons_exclusions : dict[str, Iterable[str]], optional
        Mapping of lowercase view_name -> iterable of codes to exclude.
    """
    df = _standardize_columns(df.copy())
    ons_exclusions = ons_exclusions or DEFAULT_ONS_EXCLUSIONS
    codes = set(ons_exclusions.get(view_name.lower(), []))

    if "Local_Authority_Code" not in df.columns:
        raise KeyError("Expected 'Local_Authority_Code' column after standardization.")

    if not codes:
        return df.reset_index(drop=True)

    return df[~df["Local_Authority_Code"].astype(str).isin(codes)].reset_index(drop=True)

def preprocess_and_align_views(
    view_dict: Dict[str, pd.DataFrame],
    meta_cols: Optional[Iterable[str]] = None,
    winsor_quantiles: Tuple[float, float] = DEFAULT_WINSOR_QUANTILES,
    ons_exclusions: Optional[Dict[str, Iterable[str]]] = None,
    index_col: str = "Local_Authority_Code",
) -> Tuple[pd.DataFrame, Iterable[str]]:
    """
    Clean each view (winsorize, numeric-only), apply exclusions, set a common index,
    and return a single concatenated matrix over the intersection of LAs.

    Returns
    -------
    concatenated : pd.DataFrame
        All views horizontally concatenated on the common index, sorted by index.
    common_index : list[str]
        Sorted list of Local Authority codes present in all included views.

    Notes
    -----
    - Mirrors the notebook logic but aligned to the module’s naming standards.
    - For strictly aligned per-view outputs instead of a single concatenated frame,
      you can call `align_views_on_index` first, then concatenate.
    """
    ons_exclusions = ons_exclusions or DEFAULT_ONS_EXCLUSIONS
    cleaned_views = {}
    index_sets = []

    for view_name, df in view_dict.items():
        df = preprocess_view(df, meta_cols=meta_cols, winsor_quantiles=winsor_quantiles)
        df = exclude_las(df, view_name, ons_exclusions=ons_exclusions)

        df = _standardize_columns(df)
        if index_col not in df.columns:
            raise KeyError(f"Expected index column '{index_col}' in view '{view_name}'.")

        df[index_col] = df[index_col].astype(str).str.strip()
        df = df.set_index(index_col)
        df = df[~df.index.duplicated(keep="first")]

        # Identify numeric feature columns (exclude any residual meta)
        meta = set(meta_cols) if meta_cols is not None else set(DEFAULT_META_COLS)
        # Anything not numeric will have been coerced to float earlier
        feature_cols = [c for c in df.columns if c not in meta]
        cleaned_views[view_name] = df[feature_cols].sort_index()
        index_sets.append(set(cleaned_views[view_name].index))

    # Intersect area codes across all views
    if not index_sets:
        return pd.DataFrame(), []

    common_index = sorted(set.intersection(*index_sets))

    # Align and concatenate
    aligned_views = [v.loc[common_index] for v in cleaned_views.values()]
    concatenated = pd.concat(aligned_views, axis=1)
    return concatenated, common_index

def get_combined_views_union(
    view_dict: Dict[str, pd.DataFrame],
    meta_cols: Optional[Iterable[str]] = None,
    winsor_quantiles: Tuple[float, float] = DEFAULT_WINSOR_QUANTILES,
    ons_exclusions: Optional[Dict[str, Iterable[str]]] = None,
    index_col: str = "Local_Authority_Code",
) -> Tuple[pd.DataFrame, Iterable[str]]:
    """
    Clean each view and form the UNION of all Local Authorities across views,
    horizontally concatenating features with missing rows permitted (NaNs).

    Returns
    -------
    X_combined : pd.DataFrame
        Concatenated feature matrix over the union of indices.
    all_indices : list[str]
        Sorted list of all Local Authority codes appearing in any view (after exclusions).
    """
    ons_exclusions = ons_exclusions or DEFAULT_ONS_EXCLUSIONS
    cleaned_views = []
    all_indices = set()

    for name, df in view_dict.items():
        df_clean = preprocess_view(df, meta_cols=meta_cols, winsor_quantiles=winsor_quantiles)
        df_clean = exclude_las(df_clean, name, ons_exclusions=ons_exclusions)

        df_clean = _standardize_columns(df_clean)
        if index_col not in df_clean.columns:
            raise KeyError(f"Expected index column '{index_col}' in view '{name}'.")

        df_clean[index_col] = df_clean[index_col].astype(str).str.strip()
        df_clean = df_clean.set_index(index_col)
        df_clean = df_clean[~df_clean.index.duplicated(keep="first")]

        # Keep only non-meta numeric features
        meta = set(meta_cols) if meta_cols is not None else set(DEFAULT_META_COLS)
        features = [c for c in df_clean.columns if c not in meta]
        df_view = df_clean[features].sort_index()

        cleaned_views.append(df_view)
        all_indices.update(df_view.index)

    all_indices = sorted(all_indices)
    aligned_views = [v.reindex(all_indices) for v in cleaned_views]
    X_combined = pd.concat(aligned_views, axis=1)
    return X_combined, all_indices
