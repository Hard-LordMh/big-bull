"""
stock_model.py
--------------
Core Machine Learning module for NSE stock direction prediction.
Includes:
- Chronological train/test split (80/20)
- Horizon-specific Random Forest Classifier (1-Day and 5-Day)
- Expanding Walk-Forward Time-Series Validation (with embargo gap equal to forecast horizon)
- Comprehensive evaluation metrics & baseline comparison
- Feature importance extraction
- Latest prediction generation with calibrated tree probabilities
"""

from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
import joblib


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
    """
    Computes all standard classification evaluation metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    auc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = None

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    # Majority class baseline accuracy
    majority_class = int(pd.Series(y_true).mode()[0])
    baseline_acc = float((y_true == majority_class).mean())

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": auc if auc is not None else "N/A",
        "baseline_accuracy": float(baseline_acc),
        "outperforms_baseline": bool(acc > baseline_acc),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        }
    }


def walk_forward_validation(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'Target_5D_UP',
    n_splits: int = 5,
    horizon: int = 5
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Performs Walk-Forward Validation with an embargo gap equal to forecast horizon
    to prevent target overlap leakage across folds.
    """
    n_samples = len(df)
    if n_samples < 60:
        n_splits = 2

    min_train_size = int(n_samples * 0.4)
    step_size = (n_samples - min_train_size - horizon) // n_splits
    if step_size < 5:
        step_size = 5

    wf_results = []
    
    for fold in range(n_splits):
        train_end = min_train_size + fold * step_size
        val_start = train_end + horizon  # embargo gap
        val_end = val_start + step_size

        if val_end > n_samples:
            val_end = n_samples

        if val_start >= n_samples or (val_end - val_start) < 3:
            break

        train_fold = df.iloc[:train_end]
        val_fold = df.iloc[val_start:val_end]

        X_tr = train_fold[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        y_tr = train_fold[target_col]
        X_val = val_fold[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        y_val = val_fold[target_col]

        if len(np.unique(y_tr)) < 2:
            continue

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            random_state=42 + fold,
            class_weight='balanced'
        )
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = calculate_metrics(y_val.values, y_pred, y_prob)
        wf_results.append({
            "fold": fold + 1,
            "train_start_date": str(pd.to_datetime(train_fold['Date'].min()).strftime('%Y-%m-%d')),
            "train_end_date": str(pd.to_datetime(train_fold['Date'].max()).strftime('%Y-%m-%d')),
            "val_start_date": str(pd.to_datetime(val_fold['Date'].min()).strftime('%Y-%m-%d')),
            "val_end_date": str(pd.to_datetime(val_fold['Date'].max()).strftime('%Y-%m-%d')),
            "train_size": len(train_fold),
            "val_size": len(val_fold),
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "baseline_accuracy": metrics["baseline_accuracy"]
        })

    wf_df = pd.DataFrame(wf_results)
    avg_metrics = {
        "mean_accuracy": float(wf_df["accuracy"].mean()) if not wf_df.empty else 0.0,
        "mean_balanced_accuracy": float(wf_df["balanced_accuracy"].mean()) if not wf_df.empty else 0.0,
        "mean_f1_score": float(wf_df["f1_score"].mean()) if not wf_df.empty else 0.0,
        "mean_baseline_accuracy": float(wf_df["baseline_accuracy"].mean()) if not wf_df.empty else 0.0
    }

    return wf_df, avg_metrics


def train_and_evaluate_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = None,
    horizon: int = 5,
    test_ratio: float = 0.2
) -> Dict[str, Any]:
    """
    Main model workflow:
    1. Chronologically splits dataset into Train (80%) and Holdout Test (20%).
    2. Performs Walk-Forward Validation on the training split.
    3. Fits final Random Forest model on the training set with OOB scoring enabled.
    4. Evaluates model performance on chronological Holdout Test set.
    5. Extracts feature importances.
    """
    if isinstance(target_col, int):
        horizon = target_col
        target_col = None

    if target_col is None:
        target_col = f"Target_{horizon}D_UP" if f"Target_{horizon}D_UP" in train_df.columns else "Target_5D_UP"

    n_samples = len(train_df)
    test_size = max(int(n_samples * test_ratio), 5)
    train_size = n_samples - test_size

    train_split = train_df.iloc[:train_size].copy()
    test_split = train_df.iloc[train_size:].copy()

    X_train = train_split[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y_train = train_split[target_col]
    X_test = test_split[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y_test = test_split[target_col]

    wf_df, wf_summary = walk_forward_validation(train_split, feature_cols, target_col, horizon=horizon)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced',
        oob_score=True
    )
    rf_model.fit(X_train, y_train)

    y_test_pred = rf_model.predict(X_test)
    y_test_prob = rf_model.predict_proba(X_test)[:, 1] if len(rf_model.classes_) > 1 else np.zeros(len(X_test))

    test_metrics = calculate_metrics(y_test.values, y_test_pred, y_test_prob)
    if hasattr(rf_model, "oob_score_"):
        test_metrics["oob_accuracy"] = float(rf_model.oob_score_)

    backtest_predictions = test_split[['Date', 'Close', target_col]].copy()
    backtest_predictions['Predicted_Target'] = y_test_pred
    backtest_predictions['Prob_UP'] = y_test_prob
    backtest_predictions['Prob_DOWN'] = 1.0 - y_test_prob
    backtest_predictions['Actual_Direction'] = np.where(backtest_predictions[target_col] == 1, 'UP', 'DOWN')
    backtest_predictions['Predicted_Direction'] = np.where(backtest_predictions['Predicted_Target'] == 1, 'UP', 'DOWN')
    backtest_predictions['Is_Correct'] = (backtest_predictions['Actual_Direction'] == backtest_predictions['Predicted_Direction'])

    importances = rf_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    return {
        "model": rf_model,
        "horizon": horizon,
        "target_col": target_col,
        "train_split_info": {
            "train_start": str(pd.to_datetime(train_split['Date'].min()).strftime('%Y-%m-%d')),
            "train_end": str(pd.to_datetime(train_split['Date'].max()).strftime('%Y-%m-%d')),
            "train_count": len(train_split),
            "test_start": str(pd.to_datetime(test_split['Date'].min()).strftime('%Y-%m-%d')),
            "test_end": str(pd.to_datetime(test_split['Date'].max()).strftime('%Y-%m-%d')),
            "test_count": len(test_split)
        },
        "test_metrics": test_metrics,
        "walk_forward_metrics": wf_summary,
        "walk_forward_df": wf_df,
        "backtest_predictions": backtest_predictions,
        "feature_importance": feat_imp_df,
        "feature_columns": feature_cols
    }


# =============================================================================
# Random Forest Detection Techniques
# =============================================================================

def detect_tree_consensus(model: RandomForestClassifier, X_latest: pd.DataFrame) -> Dict[str, Any]:
    """
    Extracts prediction votes across all individual decision tree estimators in the Random Forest.
    Calculates tree consensus percentage, variance, and confidence strength classification.
    """
    if not hasattr(model, "estimators_") or len(model.estimators_) == 0:
        return {
            "trees_up": 0,
            "trees_down": 0,
            "total_trees": 0,
            "consensus_pct": 50.0,
            "strength": "UNKNOWN",
            "strength_label": "Unknown",
            "tree_variance": 0.0,
            "majority_direction": "UP",
            "tree_votes": []
        }

    votes = []
    for tree in model.estimators_:
        pred = int(tree.predict(X_latest.values)[0])
        votes.append(pred)

    total_trees = len(votes)
    trees_up = int(sum(votes))
    trees_down = total_trees - trees_up
    
    majority_votes = max(trees_up, trees_down)
    consensus_pct = round((majority_votes / total_trees) * 100.0, 1)

    if consensus_pct >= 75.0:
        strength = "HIGH_CONVICTION"
        strength_label = f"Strong Forest Consensus ({consensus_pct}%)"
    elif consensus_pct >= 60.0:
        strength = "MODERATE_CONVICTION"
        strength_label = f"Moderate Forest Consensus ({consensus_pct}%)"
    else:
        strength = "DIVIDED_FOREST"
        strength_label = f"Divided Forest / Choppy Setup ({consensus_pct}%)"

    tree_variance = float(np.var(votes))

    return {
        "trees_up": trees_up,
        "trees_down": trees_down,
        "total_trees": total_trees,
        "consensus_pct": consensus_pct,
        "strength": strength,
        "strength_label": strength_label,
        "tree_variance": round(tree_variance, 4),
        "majority_direction": "UP" if trees_up >= trees_down else "DOWN",
        "tree_votes": votes
    }


def detect_breakout_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies Isolation Forest (unsupervised tree ensemble) to detect institutional volume
    surges and volatility breakout anomalies across historical sessions and the latest bar.
    """
    df_clean = df.copy()
    if len(df_clean) < 15:
        df_clean['Is_Tree_Anomaly'] = False
        df_clean['Anomaly_Score'] = 0.0
        return df_clean, {
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "anomaly_type": "NORMAL_SESSION",
            "badge": "✅ Standard Market Flow",
            "description": "Insufficient history for tree anomaly detection.",
            "volume_surge_multiple": 1.0,
            "intraday_spread_pct": 0.0,
            "historical_anomalies_count": 0,
            "total_bars_analyzed": len(df_clean)
        }

    # Derive tree anomaly features
    vol_series = df_clean['Volume'] if 'Volume' in df_clean.columns else pd.Series(1.0, index=df_clean.index)
    vol_sma20 = vol_series.rolling(20, min_periods=5).mean().replace(0, 1.0)
    vol_surge = (vol_series / vol_sma20).fillna(1.0)
    
    high_col = df_clean['High'] if 'High' in df_clean.columns else df_clean['Close']
    low_col = df_clean['Low'] if 'Low' in df_clean.columns else df_clean['Close']
    close_col = df_clean['Close']
    
    intraday_spread = ((high_col - low_col) / close_col.replace(0, 1.0)).fillna(0.0)
    abs_return_1d = (close_col.pct_change().abs()).fillna(0.0)

    features = pd.DataFrame({
        'vol_surge': vol_surge,
        'intraday_spread': intraday_spread,
        'abs_return': abs_return_1d
    }).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    preds = iso_forest.fit_predict(features)
    scores = iso_forest.decision_function(features)

    df_clean['Is_Tree_Anomaly'] = (preds == -1)
    df_clean['Anomaly_Score'] = scores

    latest_is_anomaly = bool(df_clean['Is_Tree_Anomaly'].iloc[-1])
    latest_score = float(scores[-1])
    latest_surge = float(vol_surge.iloc[-1])
    latest_spread = float(intraday_spread.iloc[-1])

    if latest_is_anomaly:
        if latest_surge >= 1.75:
            anomaly_type = "INSTITUTIONAL_VOLUME_EXPANSION"
            badge = "⚡ Institutional Volume Breakout"
            desc = f"Unusual institutional volume surge ({latest_surge:.1f}x 20-day average) detected by tree isolation."
        else:
            anomaly_type = "VOLATILITY_EXPANSION_BREAKOUT"
            badge = "⚠️ Volatility Expansion Breakout"
            desc = f"Intraday price spread ({latest_spread*100:.1f}%) significantly deviates from normal range."
    else:
        anomaly_type = "NORMAL_SESSION"
        badge = "✅ Standard Market Flow"
        desc = "Volume and intraday volatility are within standard statistical equilibrium."

    anom_dates = [str(pd.to_datetime(d).strftime('%Y-%m-%d')) for d in df_clean.loc[df_clean['Is_Tree_Anomaly'], 'Date']] if 'Date' in df_clean.columns else []

    summary = {
        "is_anomaly": latest_is_anomaly,
        "anomaly_score": round(latest_score, 4),
        "anomaly_type": anomaly_type,
        "badge": badge,
        "description": desc,
        "volume_surge_multiple": round(latest_surge, 2),
        "intraday_spread_pct": round(latest_spread * 100.0, 2),
        "historical_anomalies_count": int(df_clean['Is_Tree_Anomaly'].sum()),
        "historical_anomaly_dates": anom_dates,
        "total_bars_analyzed": len(df_clean)
    }

    return df_clean, summary


def detect_market_regime(df: pd.DataFrame, tree_consensus: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classifies the current market environment into one of 4 systematic regimes:
    1. Bullish Momentum: Strong tree agreement UP + price above moving averages.
    2. Bearish Trend: Strong tree agreement DOWN + price below moving averages.
    3. High Volatility Expansion: Large intraday spreads / ATR expansion.
    4. Consolidation Range: Low tree consensus / choppy price oscillating around SMA.
    """
    latest_close = float(df['Close'].iloc[-1])
    sma20 = float(df['Close'].rolling(20, min_periods=5).mean().iloc[-1])
    
    rsi = float(df['RSI_14'].iloc[-1]) if 'RSI_14' in df.columns else 50.0
    atr = float(df['ATR_14'].iloc[-1]) if 'ATR_14' in df.columns else (latest_close * 0.02)
    atr_norm = atr / (latest_close if latest_close > 0 else 1.0)

    consensus_pct = tree_consensus.get("consensus_pct", 50.0)
    majority_dir = tree_consensus.get("majority_direction", "UP")

    if consensus_pct >= 62.0 and majority_dir == "UP" and latest_close >= sma20 and rsi >= 48:
        regime_key = "BULLISH_MOMENTUM"
        regime_name = "Bullish Momentum Regime"
        color = "#10b981"
        badge = "🟢 Bullish Momentum"
        desc = "High tree consensus with price sustaining above SMA20. Favorable for momentum continuation."
        implication = "Trend continuation favored; dips supported by short-term moving average."
    elif consensus_pct >= 62.0 and majority_dir == "DOWN" and latest_close <= sma20 and rsi <= 52:
        regime_key = "BEARISH_TREND"
        regime_name = "Bearish Trend Regime"
        color = "#ef4444"
        badge = "🔴 Bearish Trend"
        desc = "Trees heavily favor downside with price trading below key moving averages."
        implication = "Selling pressure dominant; exercise caution on aggressive long positions."
    elif atr_norm > 0.032:
        regime_key = "HIGH_VOLATILITY_EXPANSION"
        regime_name = "High-Volatility Expansion Regime"
        color = "#f59e0b"
        badge = "⚡ High-Volatility Expansion"
        desc = "Elevated ATR spread indicating wide swings and potential trend acceleration or exhaustion."
        implication = "Requires wider stop-loss buffers and disciplined position sizing."
    else:
        regime_key = "CONSOLIDATION_RANGE"
        regime_name = "Range-Bound Consolidation Regime"
        color = "#6366f1"
        badge = "🟣 Range-Bound Consolidation"
        desc = "Balanced tree votes with price oscillating near moving averages in a sideways channel."
        implication = "Mean-reversion setups favored over trend breakout strategies."

    return {
        "regime_key": regime_key,
        "regime_name": regime_name,
        "badge": badge,
        "color": color,
        "description": desc,
        "implication": implication,
        "close_vs_sma20_pct": round(((latest_close / sma20) - 1.0) * 100.0, 2) if sma20 > 0 else 0.0,
        "rsi_14": round(rsi, 1)
    }


def explain_tree_prediction(
    latest_row: pd.DataFrame,
    feat_imp_df: pd.DataFrame,
    top_n: int = 4
) -> List[Dict[str, Any]]:
    """
    Extracts the top influential indicators for the latest session based on model feature importances
    and current normalized values.
    """
    top_features = []
    top_imp = feat_imp_df.head(top_n) if feat_imp_df is not None and not feat_imp_df.empty else pd.DataFrame()
    
    for _, row in top_imp.iterrows():
        f_name = row['Feature']
        f_val = float(latest_row[f_name].values[0]) if f_name in latest_row.columns else 0.0
        
        status = "Neutral"
        if "RSI" in f_name:
            if f_val < 35:
                status = "Oversold Bullish Setup"
            elif f_val > 65:
                status = "Overbought Caution"
            else:
                status = f"Neutral Momentum ({f_val:.1f})"
        elif "MACD" in f_name:
            status = "Bullish Momentum" if f_val > 0 else "Bearish Divergence"
        elif "Ratio" in f_name or "SMA" in f_name:
            status = "Above Average" if f_val > 1.0 else "Below Average"
        elif "ATR" in f_name or "Vol" in f_name:
            status = "Elevated Volatility" if f_val > 0.02 else "Low Volatility"
        else:
            status = f"Value: {f_val:.2f}"

        top_features.append({
            "feature": f_name,
            "importance_pct": round(float(row['Importance']) * 100.0, 1),
            "latest_value": round(f_val, 3),
            "indication": status
        })

    return top_features


def predict_latest_session(
    model: RandomForestClassifier,
    latest_df: pd.DataFrame,
    feature_cols: List[str],
    ticker: str = "NSE_STOCK",
    horizon: int = 5,
    full_df: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Generates prediction for the latest valid observation in the dataset.
    Includes full Random Forest detection diagnostics:
    - Tree consensus breakdown (UP vs DOWN votes across 100 trees)
    - Unsupervised Isolation Forest anomaly / breakout detection
    - Systematic market regime classification
    - Top driving feature contributions
    """
    if latest_df.empty:
        raise ValueError("No valid observation available for latest prediction.")

    latest_row = latest_df.iloc[[-1]].copy()
    signal_date = str(pd.to_datetime(latest_row['Date'].values[0]).strftime('%Y-%m-%d'))
    latest_close = float(latest_row['Close'].values[0])

    X_latest = latest_row[feature_cols]
    pred_class = int(model.predict(X_latest)[0])
    probs = model.predict_proba(X_latest)[0]

    if len(probs) == 2:
        prob_down = float(probs[0])
        prob_up = float(probs[1])
    else:
        prob_up = 1.0 if pred_class == 1 else 0.0
        prob_down = 1.0 - prob_up

    pred_dir = "UP" if pred_class == 1 else "DOWN"

    # --- Random Forest Detection Suite ---
    # 1. Tree consensus
    tree_consensus = detect_tree_consensus(model, X_latest)

    # 2. Breakout Anomaly detection using Isolation Forest
    eval_df = full_df if full_df is not None and not full_df.empty else latest_df
    _, anomaly_info = detect_breakout_anomalies(eval_df)

    # 3. Market Regime classification
    market_regime = detect_market_regime(eval_df, tree_consensus)

    # 4. Driving features explanation
    feat_imp_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    driving_factors = explain_tree_prediction(latest_row, feat_imp_df, top_n=4)

    return {
        "ticker": ticker,
        "signal_date": signal_date,
        "latest_close": latest_close,
        "predicted_direction": pred_dir,
        "probability_up": prob_up,
        "probability_down": prob_down,
        "forecast_horizon_sessions": horizon,
        "rf_detection": {
            "tree_consensus": tree_consensus,
            "market_regime": market_regime,
            "breakout_anomaly": anomaly_info,
            "driving_factors": driving_factors
        },
        "disclaimer": "Probabilities represent statistical model outputs based on historical patterns and are not guaranteed predictions."
    }


def _prepare_feature_frame(
    clean_df: pd.DataFrame,
    horizon: int,
    feature_cols: List[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Builds the technical feature + horizon target frame used by the
    walk-forward history engine. Returns the enriched frame and the usable feature list.
    """
    from feature_engineering import build_features, create_target, FEATURE_COLUMNS

    df = clean_df.copy().sort_values('Date').reset_index(drop=True)

    required_raw = ['Open', 'High', 'Low', 'Close', 'Volume']
    has_features = bool(feature_cols) and all(c in df.columns for c in feature_cols)

    if not has_features or not all(c in df.columns for c in required_raw):
        df = build_features(df)

    df = create_target(df, horizon=horizon)

    cols = list(feature_cols) if feature_cols else list(FEATURE_COLUMNS)
    usable = [c for c in cols if c in df.columns]
    if not usable:
        usable = [c for c in FEATURE_COLUMNS if c in df.columns]

    df[usable] = df[usable].replace([np.inf, -np.inf], np.nan).ffill().bfill().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df, usable


def walk_forward_prediction_history(
    clean_df: pd.DataFrame,
    feature_cols: List[str] = None,
    horizon: int = 5,
    years: int = 5,
    min_train_rows: int = 252,
    refit_every: int = 21
) -> pd.DataFrame:
    """
    Replays the Random Forest classifier over the last `years` of history using an
    expanding walk-forward window (model refit every `refit_every` sessions, with an
    embargo gap equal to `horizon` so no signal date can see its own outcome).

    Returns a per-signal-date prediction log:
    Signal_Date, Reference_Close, Predicted_Direction, Prob_UP_Pct,
    Actual_Direction, Actual_Return_Pct, Result (HIT/MISS), Year, Fold.
    """
    if clean_df is None or clean_df.empty or len(clean_df) < 60:
        return pd.DataFrame()

    df, feats = _prepare_feature_frame(clean_df, horizon, feature_cols)
    target_col = f"Target_{horizon}D_UP"
    future_col = f"Future_Close_{horizon}D"

    df['Date'] = pd.to_datetime(df['Date'])
    cutoff = df['Date'].max() - pd.DateOffset(years=years)
    start_idx = int(df.index[df['Date'] >= cutoff][0]) if (df['Date'] >= cutoff).any() else 0

    min_train_rows = int(min(min_train_rows, max(60, start_idx)))
    if start_idx < min_train_rows:
        min_train_rows = max(40, int(start_idx * 0.6))

    if start_idx < 40:
        return pd.DataFrame()

    last_idx = len(df) - horizon - 1  # final signal date whose outcome is observable
    if last_idx <= start_idx:
        return pd.DataFrame()

    records = []
    fold = 0
    cursor = start_idx

    while cursor <= last_idx:
        fold += 1
        block_end = min(cursor + refit_every - 1, last_idx)

        train_end = max(cursor - horizon, 40)
        train_block = df.iloc[:train_end].dropna(subset=[target_col])
        val_block = df.iloc[cursor:block_end + 1]

        if len(train_block) < 40 or val_block.empty:
            cursor = block_end + 1
            continue

        y_tr = train_block[target_col].astype(int)
        if y_tr.nunique() < 2:
            cursor = block_end + 1
            continue

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            random_state=42 + fold,
            class_weight='balanced',
            n_jobs=-1
        )
        model.fit(train_block[feats], y_tr)

        X_val = val_block[feats]
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1] if len(model.classes_) > 1 else np.full(len(X_val), 0.5)

        for pos in range(len(val_block)):
            row = val_block.iloc[pos]
            actual_up = row[target_col]
            if pd.isna(actual_up) or pd.isna(row.get(future_col, np.nan)):
                continue
            ref_close = float(row['Close'])
            fut_close = float(row[future_col])
            if ref_close <= 0:
                continue
            actual_ret = ((fut_close / ref_close) - 1.0) * 100.0
            pred_up = int(preds[pos])
            records.append({
                "Signal_Date": row['Date'].strftime('%Y-%m-%d'),
                "Year": row['Date'].year,
                "Fold": fold,
                "Reference_Close": round(ref_close, 2),
                "Predicted_Direction": "UP" if pred_up == 1 else "DOWN",
                "Prob_UP_Pct": round(float(probs[pos]) * 100.0, 1),
                "Actual_Direction": "UP" if int(actual_up) == 1 else "DOWN",
                "Actual_Return_Pct": round(actual_ret, 2),
                "Result": "HIT" if (pred_up == int(actual_up)) else "MISS"
            })

        cursor = block_end + 1

    history = pd.DataFrame(records)
    if not history.empty:
        history = history.sort_values("Signal_Date").reset_index(drop=True)
    return history


def summarize_prediction_history(history: pd.DataFrame) -> Dict[str, Any]:
    """
    Aggregates a walk-forward prediction log into overall + per-year statistics.
    """
    if history is None or history.empty:
        return {"valid": False, "overall": pd.DataFrame(), "yearly": pd.DataFrame()}

    def _group(g: pd.DataFrame) -> Dict[str, Any]:
        hits = int((g["Result"] == "HIT").sum())
        total = len(g)
        wins = g[g["Result"] == "HIT"]["Actual_Return_Pct"]
        losses = g[g["Result"] == "MISS"]["Actual_Return_Pct"]
        up_calls = int((g["Predicted_Direction"] == "UP").sum())
        return {
            "Signals": total,
            "Hits": hits,
            "Hit_Rate_Pct": round(hits / total * 100.0, 1) if total else 0.0,
            "Baseline_Pct": round(max(up_calls, total - up_calls) / total * 100.0, 1) if total else 0.0,
            "Avg_Return_When_Right_Pct": round(float(wins.mean()), 2) if len(wins) else 0.0,
            "Avg_Return_When_Wrong_Pct": round(float(losses.mean()), 2) if len(losses) else 0.0,
        }

    yearly_rows = []
    for year, grp in history.groupby("Year"):
        row = {"Year": int(year)}
        row.update(_group(grp))
        yearly_rows.append(row)
    yearly_df = pd.DataFrame(yearly_rows)

    overall = pd.DataFrame([{"Year": "All", **_group(history)}])

    first_dt = pd.to_datetime(history["Signal_Date"].min())
    last_dt = pd.to_datetime(history["Signal_Date"].max())
    baseline = max((history["Predicted_Direction"] == "UP").mean(), 1 - (history["Predicted_Direction"] == "UP").mean())

    return {
        "valid": True,
        "overall": overall,
        "yearly": yearly_df,
        "signals": len(history),
        "hit_rate_pct": round((history["Result"] == "HIT").mean() * 100.0, 1),
        "baseline_pct": round(float(baseline) * 100.0, 1),
        "edge_pct": round(float((history["Result"] == "HIT").mean() - baseline) * 100.0, 1),
        "predicted_up_share_pct": round(float((history["Predicted_Direction"] == "UP").mean()) * 100.0, 1),
        "start_date": first_dt.strftime("%Y-%m-%d"),
        "end_date": last_dt.strftime("%Y-%m-%d"),
        "avg_return_pct": round(float(history["Actual_Return_Pct"].mean()), 2),
    }


def predict_next_week(
    clean_df: pd.DataFrame,
    feature_cols: List[str] = None,
    sessions: int = 5
) -> pd.DataFrame:
    """
    Produces an explicit UP/DOWN call for each of the next `sessions` trading days.

    The 1-session classifier is trained on all available history. Each future session is
    then projected bar-by-bar (ATR-bounded synthetic OHLCV), features are rebuilt on the
    extended frame, and the model votes on that session. This keeps the call anchored to
    real model probabilities instead of a flat extrapolation.
    """
    if clean_df is None or clean_df.empty or len(clean_df) < 60:
        return pd.DataFrame()

    from feature_engineering import build_features

    df, feats = _prepare_feature_frame(clean_df, horizon=1, feature_cols=feature_cols)
    target_col = "Target_1D_UP"

    train_block = df.dropna(subset=[target_col])
    if len(train_block) < 40:
        return pd.DataFrame()

    y_tr = train_block[target_col].astype(int)
    if y_tr.nunique() < 2:
        return pd.DataFrame()

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(train_block[feats], y_tr)

    atr = float(df['ATR_14'].iloc[-1]) if 'ATR_14' in df.columns else float(df['Close'].iloc[-1] * 0.015)
    if not np.isfinite(atr) or atr <= 0:
        atr = float(df['Close'].iloc[-1] * 0.015)
    vol_ma = float(df['Volume'].tail(20).mean()) if 'Volume' in df.columns else 0.0

    last_date = pd.to_datetime(df['Date'].iloc[-1])
    last_close = float(df['Close'].iloc[-1])

    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=sessions + 5, freq='B')
    future_dates = [d for d in future_dates if d > last_date][:sessions]

    working = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    working['Date'] = pd.to_datetime(working['Date'])

    prev_close = last_close
    prev_prob_up = float(model.predict_proba(df[feats].iloc[[-1]])[0][1]) if len(model.classes_) > 1 else 0.5
    rows = []

    for i, fdate in enumerate(future_dates):
        prob_up = prev_prob_up if i == 0 else rows[-1]["Prob_UP_Pct"] / 100.0
        expected_move = (prob_up - 0.5) * 2.0 * atr
        proj_close = max(prev_close + expected_move, 1.0)

        synth_open = prev_close
        synth_high = max(synth_open, proj_close) + atr * 0.35
        synth_low = min(synth_open, proj_close) - atr * 0.35
        synth_vol = vol_ma * (1.0 + abs(prob_up - 0.5) * 1.5)

        working = pd.concat([working, pd.DataFrame([{
            'Date': fdate, 'Open': synth_open, 'High': synth_high,
            'Low': synth_low, 'Close': proj_close, 'Volume': synth_vol
        }])], ignore_index=True)

        ext_feat = build_features(working)
        ext_feat[feats] = ext_feat[feats].replace([np.inf, -np.inf], np.nan).ffill().bfill().replace([np.inf, -np.inf], 0.0).fillna(0.0)

        x_row = ext_feat[feats].iloc[[-1]]
        if len(model.classes_) > 1:
            prob_up = float(model.predict_proba(x_row)[0][1])
        else:
            prob_up = 0.5

        direction = "UP" if prob_up >= 0.5 else "DOWN"
        rows.append({
            "Trading_Day": f"Day +{i + 1}",
            "Trading_Date": fdate.strftime("%Y-%m-%d"),
            "Weekday": fdate.strftime("%a"),
            "Predicted_Direction": direction,
            "Prob_UP_Pct": round(prob_up * 100.0, 1),
            "Prob_DOWN_Pct": round((1.0 - prob_up) * 100.0, 1),
            "Projected_Close_INR": round(proj_close, 2),
            "Change_vs_Prev_Pct": round(((proj_close / prev_close) - 1.0) * 100.0, 2) if prev_close else 0.0
        })
        prev_close = proj_close

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    weekly_up = out["Prob_UP_Pct"].mean()
    out.attrs["week_direction"] = "UP" if weekly_up >= 50 else "DOWN"
    out.attrs["week_prob_up"] = round(float(weekly_up), 1)
    out.attrs["reference_close"] = round(last_close, 2)
    out.attrs["reference_date"] = last_date.strftime("%Y-%m-%d")
    return out


def predict_5day_trajectory(
    clean_df: pd.DataFrame,
    feature_cols: List[str],
    model: RandomForestClassifier = None
) -> pd.DataFrame:
    """
    Generates a session-by-session 5-day future projected path (Day +1 to Day +5)
    using Random Forest techniques, calibrated tree probabilities, and technical momentum.
    Returns a pandas DataFrame with Day +1 to Day +5 projected dates, predicted close,
    day-over-day return %, direction (UP/DOWN), probability, and estimated ATR range.
    """
    if clean_df is None or clean_df.empty or len(clean_df) < 20:
        return pd.DataFrame()

    df_calc = clean_df.copy()

    # Check if feature columns exist, otherwise build features
    missing_cols = [c for c in feature_cols if c not in df_calc.columns]
    if missing_cols:
        from feature_engineering import build_features
        df_calc = build_features(df_calc)

    latest_row = df_calc.iloc[-1]
    last_dt = pd.to_datetime(latest_row['Date'])
    last_close = float(latest_row['Close'])

    atr = float(df_calc['ATR_14'].iloc[-1]) if 'ATR_14' in df_calc.columns else (last_close * 0.018)
    rsi = float(df_calc['RSI_14'].iloc[-1]) if 'RSI_14' in df_calc.columns else 50.0

    # Train or fit 1-day model if not provided
    if model is None:
        if 'Target_1D_UP' not in df_calc.columns:
            df_calc['Target_1D_UP'] = np.where(df_calc['Close'].shift(-1) > df_calc['Close'], 1, 0)
        valid_train = df_calc.dropna(subset=['Target_1D_UP'] + feature_cols)
        X_tr = valid_train[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        y_tr = valid_train['Target_1D_UP']
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
        model.fit(X_tr, y_tr)

    X_latest = df_calc.iloc[[-1]][feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    prob_1d_up = float(model.predict_proba(X_latest)[0][1]) if len(model.classes_) > 1 else 0.5

    # Generate 5 future trading business days
    future_dates = pd.date_range(start=last_dt + pd.Timedelta(days=1), periods=10, freq='B')
    future_dates = [d for d in future_dates if d > last_dt][:5]

    trajectory = []
    current_proj_close = last_close
    decay_factor = [1.0, 0.92, 0.85, 0.78, 0.72]

    for i in range(len(future_dates)):
        fut_dt = future_dates[i]
        dt_str = fut_dt.strftime('%Y-%m-%d (%a)')

        adj_prob = prob_1d_up * decay_factor[i] + (0.5 if rsi == 50 else (rsi / 100.0) * 0.1) * (1 - decay_factor[i])
        adj_prob = min(max(adj_prob, 0.15), 0.88)

        daily_expected_return_pct = (adj_prob - 0.5) * 2.0 * (atr / last_close * 100.0)
        day_dir = "UP" if daily_expected_return_pct >= 0 else "DOWN"

        current_proj_close = current_proj_close * (1.0 + (daily_expected_return_pct / 100.0))
        upper_bound = current_proj_close + (atr * (1.0 + i * 0.2))
        lower_bound = max(current_proj_close - (atr * (1.0 + i * 0.2)), 1.0)

        trajectory.append({
            "Day": f"Day +{i+1}",
            "Trading_Date": dt_str,
            "Predicted_Close_INR": round(current_proj_close, 2),
            "Expected_Return_Pct": round(daily_expected_return_pct, 2),
            "Cumulative_Return_Pct": round(((current_proj_close / last_close) - 1.0) * 100.0, 2),
            "Direction": day_dir,
            "RF_Probability_UP": round(adj_prob * 100.0, 1),
            "Lower_Bound_INR": round(lower_bound, 2),
            "Upper_Bound_INR": round(upper_bound, 2)
        })

    return pd.DataFrame(trajectory)

