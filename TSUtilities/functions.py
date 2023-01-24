# -*- coding: utf-8 -*-
from TSUtilities.TSTrend.trend_dampen import TrendDampen


def dampen_prophet(y, fit_df, forecast_df):
    """
    A function that takes in the forecasted dataframe output of Prophet and
    constrains the trend based on it's percieved strength'

    Parameters
    ----------
    y : pd.Series
        The single time series of actuals that are fitted with Prophet.
    fit_df : pd.DataFrame
        The Fitted DataFrame from Prophet.
    forecast_df : pd.DataFrame
        The future forecast dataframe from prophet which includes the predicted trend.

    Returns
    -------
    forecasts_damped : np.array
        The damped trend forecast.

    """
    predictions = forecast_df.tail(len(forecast_df) - len(fit_df))
    predicted_trend = predictions['trend'].values
    trend_component = fit_df['trend'].values
    if 'multiplicative_terms' in forecast_df.columns:
        seasonality_component = fit_df['trend'].values * \
                                fit_df['multiplicative_terms'].values
        dampener = TrendDampen(damp_factor='auto',
                                damp_style='smooth')
        dampened_trend = dampener.dampen(predicted_trend,
                                         y=y,
                                         trend_component=trend_component,
                                         seasonality_component=seasonality_component)
        forecasts_damped = predictions['additive_terms'].values + \
                           dampened_trend + \
                           (dampened_trend * \
                           predictions['multiplicative_terms'].values)
    else:
        seasonality_component = fit_df['additive_terms'].values
        dampener = TrendDampen(damp_factor='auto',
                                damp_style='smooth')
        dampened_trend = dampener.dampen(predicted_trend,
                                         y=y,
                                         trend_component=trend_component,
                                         seasonality_component=seasonality_component)
        forecasts_damped = predictions['additive_terms'].values + dampened_trend
    return forecasts_damped

