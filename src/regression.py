import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def winsorize_series(series: pd.Series, limits=(0.01, 0.99)) -> pd.Series:
    lower = np.percentile(series.dropna(), limits[0] * 100)
    upper = np.percentile(series.dropna(), limits[1] * 100)
    return np.clip(series, lower, upper)


def winsorize_df(df: pd.DataFrame, columns: list, limits=(0.01, 0.99)) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        if col in result.columns:
            result[col] = winsorize_series(result[col], limits)
    return result


def standardize(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    scaler = StandardScaler()
    result = df.copy()
    valid_cols = [c for c in columns if c in result.columns]
    result[valid_cols] = scaler.fit_transform(result[valid_cols])
    return result


def prepare_regression_data(df: pd.DataFrame, y_var: str, x_var: str, controls: list,
                            log_y: bool = False, winsorize: bool = True,
                            winsorize_limits: tuple = (0.01, 0.99),
                            standardize_x: bool = True) -> pd.DataFrame:
    """准备回归数据：缺失值处理、缩尾、标准化。"""
    features = [x_var] + [c for c in controls if c in df.columns]
    reg_df = df[[y_var] + features].dropna().copy()

    if winsorize:
        reg_df = winsorize_df(reg_df, [y_var] + features, winsorize_limits)

    if log_y:
        reg_df[y_var] = np.log1p(reg_df[y_var])

    if standardize_x:
        reg_df = standardize(reg_df, features)

    return reg_df


def run_ols(df: pd.DataFrame, y_var: str, x_var: str, controls: list,
            log_y: bool = False, winsorize: bool = True,
            cov_type: str = "HC3") -> dict:
    """OLS 回归。"""
    reg_df = prepare_regression_data(df, y_var, x_var, controls, log_y=log_y, winsorize=winsorize)
    features = [x_var] + [c for c in controls if c in reg_df.columns]
    X = sm.add_constant(reg_df[features])
    y = reg_df[y_var]

    model = sm.OLS(y, X).fit(cov_type=cov_type)
    return _extract_results(model, x_var, features)


def run_nb(df: pd.DataFrame, y_var: str, x_var: str, controls: list,
           winsorize: bool = True) -> dict:
    """负二项回归。"""
    reg_df = prepare_regression_data(df, y_var, x_var, controls, log_y=False, winsorize=winsorize)
    features = [x_var] + [c for c in controls if c in reg_df.columns]
    X = sm.add_constant(reg_df[features])
    y = reg_df[y_var]

    try:
        model = NegativeBinomial(y, X).fit(disp=0, maxiter=200)
        return _extract_results(model, x_var, features)
    except Exception as e:
        logger.warning(f"NB 回归未收敛: {e}")
        return {"converged": False, "error": str(e)}


def run_glm(df: pd.DataFrame, y_var: str, x_var: str, controls: list,
            family=None, log_y: bool = False, winsorize: bool = True,
            cov_type: str = "HC3") -> dict:
    """GLM 回归。"""
    if family is None:
        family = sm.families.Gaussian()
    reg_df = prepare_regression_data(df, y_var, x_var, controls, log_y=log_y, winsorize=winsorize)
    features = [x_var] + [c for c in controls if c in reg_df.columns]
    X = sm.add_constant(reg_df[features])
    y = reg_df[y_var]

    model = sm.GLM(y, X, family=family).fit(cov_type=cov_type)
    return _extract_results(model, x_var, features)


def run_cox_ph(df: pd.DataFrame, duration_var: str, event_var: str,
               x_var: str, controls: list) -> dict:
    """Cox 比例风险模型。"""
    from lifelines import CoxPHFitter

    features = [x_var] + [c for c in controls if c in df.columns]
    reg_df = df[[duration_var, event_var] + features].dropna().copy()

    cph = CoxPHFitter()
    cph.fit(reg_df, duration_col=duration_var, event_col=event_var)

    result = {
        "converged": True,
        "model": cph,
        "x_var": x_var,
        "coef": cph.params_.get(x_var, None),
        "pvalue": cph.p_values().get(x_var, None),
        "ci_lower": cph.confidence_intervals_.loc[x_var, "lower 95%"] if x_var in cph.confidence_intervals_.index else None,
        "ci_upper": cph.confidence_intervals_.loc[x_var, "upper 95%"] if x_var in cph.confidence_intervals_.index else None,
    }
    return result


def _extract_results(model, x_var: str, features: list) -> dict:
    """从 statsmodels 结果中提取关键统计量。"""
    return {
        "converged": True,
        "model": model,
        "x_var": x_var,
        "coef": model.params.get(x_var, None),
        "std_err": model.bse.get(x_var, None),
        "pvalue": model.pvalues.get(x_var, None),
        "ci_lower": model.conf_int().loc[x_var, 0] if x_var in model.conf_int().index else None,
        "ci_upper": model.conf_int().loc[x_var, 1] if x_var in model.conf_int().index else None,
        "r_squared": getattr(model, "rsquared", None) or getattr(model, "pseudo_rsquared", None),
        "n_obs": int(model.nobs),
    }