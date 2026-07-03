from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Any


# ── Categorical Tests ─────────────────────────────────────────────────────────

def chi_squared_test(df: pd.DataFrame, feature: str, target: str) -> dict[str, Any]:
    """Chi-squared test of independence between a categorical feature and binary target."""
    clean   = df[[feature, target]].dropna()
    table   = pd.crosstab(clean[feature], clean[target])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return {
        "feature":    feature,
        "chi2":       round(float(chi2), 4),
        "p_value":    float(p),
        "dof":        int(dof),
        "significant": p < 0.05,
    }


# ── Numerical Tests ───────────────────────────────────────────────────────────

def t_test_independent(df: pd.DataFrame, feature: str, target: str) -> dict[str, Any]:
    """Welch's independent t-test for a numerical feature between two target groups."""
    clean  = df[[feature, target]].dropna()
    groups = [g[feature].values for _, g in clean.groupby(target)]
    if len(groups) != 2:
        raise ValueError(f"Target '{target}' must have exactly 2 unique values.")
    t_stat, p = stats.ttest_ind(*groups, equal_var=False)
    return {
        "feature":     feature,
        "t_statistic": round(float(t_stat), 4),
        "p_value":     float(p),
        "mean_group0": round(float(groups[0].mean()), 4),
        "mean_group1": round(float(groups[1].mean()), 4),
        "significant":  p < 0.05,
    }


def anova_test(df: pd.DataFrame, feature: str, target: str) -> dict[str, Any]:
    """One-way ANOVA: numerical feature across all target groups (≥ 2)."""
    clean  = df[[feature, target]].dropna()
    groups = [g[feature].values for _, g in clean.groupby(target)]
    f_stat, p = stats.f_oneway(*groups)
    return {
        "feature":     feature,
        "f_statistic": round(float(f_stat), 4),
        "p_value":     float(p),
        "n_groups":    len(groups),
        "significant":  p < 0.05,
    }


# ── Regression Summary ────────────────────────────────────────────────────────

def logistic_regression_summary(
    df: pd.DataFrame, features: list[str], target: str
) -> Any:
    """
    Fit statsmodels Logit and return its summary.
    Returns the fitted Results object (call .summary() on it for display).
    """
    import statsmodels.api as sm

    clean = df[features + [target]].dropna()
    X = sm.add_constant(clean[features].astype(float))
    y = clean[target].astype(int)
    model = sm.Logit(y, X)
    return model.fit(disp=False)


# ── Multicollinearity ─────────────────────────────────────────────────────────

def compute_vif(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Variance Inflation Factor for each feature. VIF > 10 indicates high multicollinearity."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    clean = df[features].dropna().astype(float)
    # Drop zero-variance columns to avoid singular matrix
    clean = clean.loc[:, clean.std() > 0]

    vif_data = pd.DataFrame({
        "feature": clean.columns,
        "VIF": [
            variance_inflation_factor(clean.values, i)
            for i in range(clean.shape[1])
        ],
    })
    return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)


# ── Correlation ───────────────────────────────────────────────────────────────

def correlation_matrix(
    df: pd.DataFrame, features: list[str], method: str = "pearson"
) -> pd.DataFrame:
    """
    Correlation matrix for the given features.
    method: 'pearson' | 'spearman' | 'kendall'
    """
    clean = df[features].dropna().astype(float)
    return clean.corr(method=method).round(4)


# ── Convenience: run all tests for a list of features ─────────────────────────

def summarise_features(
    df: pd.DataFrame,
    numerical: list[str],
    categorical: list[str],
    target: str,
) -> pd.DataFrame:
    """
    Run the appropriate test for each feature and return a tidy summary DataFrame.
    Useful for quick notebook overviews.
    """
    rows = []
    for feat in numerical:
        try:
            r = t_test_independent(df, feat, target)
            rows.append({"feature": feat, "type": "numerical",
                         "statistic": r["t_statistic"], "p_value": r["p_value"],
                         "significant": r["significant"]})
        except Exception as e:
            rows.append({"feature": feat, "type": "numerical", "error": str(e)})

    for feat in categorical:
        try:
            r = chi_squared_test(df, feat, target)
            rows.append({"feature": feat, "type": "categorical",
                         "statistic": r["chi2"], "p_value": r["p_value"],
                         "significant": r["significant"]})
        except Exception as e:
            rows.append({"feature": feat, "type": "categorical", "error": str(e)})

    return pd.DataFrame(rows).sort_values("p_value")
