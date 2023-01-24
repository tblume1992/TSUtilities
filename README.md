# TSUtilities v0.0.2

## Recent Changes

pip install TSUtilities:
```
pip install TSUtilities
```

Example of trend dampening:

```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style('darkgrid')
y = np.linspace(0, 100, 100)
plt.plot(y)
plt.show()

y_train = y[:80]
future_y = y[80:]
future_trend = future_y


from TSUtilities.TSTrend.trend_dampen import TrendDampen

dampener = TrendDampen(damp_factor=.7,
                       damp_style='smooth')
dampened_trend = dampener.dampen(future_trend)
```

Example of Prophet Trend Dampening helper function where ts is your input to prophet:

```
from TSUtilities.functions import dampen_prophet

prophet = Prophet()
prophet.fit(ts)
fitted = prophet.predict()

# create a future data frame
future = prophet.make_future_dataframe(periods=len(y_test))
forecast = prophet.predict(future)

#get predictions and required data inputs for auto-damping
predictions = forecast.tail(len(y_test))
predicted_trend = predictions['trend'].values
trend_component = fitted['trend'].values
seasonality_component = fitted['additive_terms'].values
forecasts_no_dampen = predictions['yhat'].values
forecasts_damped = dampen_prophet(y=y.values,
                                  fit_df=fitted,
                                  forecast_df=forecast)
```