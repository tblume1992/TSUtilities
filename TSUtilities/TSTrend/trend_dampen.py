# -*- coding: utf-8 -*-
import numpy as np


class TrendDampen:
    __version__ = '0.1'
    def __init__(self,
                 damp_factor='auto',
                 damp_style='smooth'
                 ):
        self.damp_style = damp_style
        self.damp_factor = damp_factor
         
    @staticmethod
    def get_tau_proposal(zeroed_trend_component,
                         damp_factor):
        N = len(zeroed_trend_component)
        target_value = damp_factor * zeroed_trend_component[-1]
        special_coefficient = np.arange(0.01, 100, 0.01)
        tau = special_coefficient*damp_factor*2*N
        trend_proposal_value = zeroed_trend_component[-1]*np.exp(-N/(tau))
        proposed_tau_idx = np.argmin(np.abs(trend_proposal_value - target_value))
        return tau[proposed_tau_idx]
    
    @staticmethod
    def smooth_dampen(zeroed_trend_component, 
                      tau):
        decay_func = np.exp(-np.array(range(1, len(zeroed_trend_component) + 1))/tau)
        dampened_trend = zeroed_trend_component*decay_func
        return dampened_trend
    
    @staticmethod
    def linear_dampen(zeroed_trend_component, damp_factor):
        target = zeroed_trend_component[-1] * damp_factor
        return np.linspace(0, target, num=len(zeroed_trend_component))

    @staticmethod
    def level_dampen(trend_component, damp_factor):
        local_level = np.abs(trend_component[0])
        target = local_level +  damp_factor * local_level
        return np.tile(target, len(trend_component))

    @staticmethod
    def calc_trend_strength(resids, deseasonalized):
        return max(0, 1-(np.var(resids)/np.var(deseasonalized)))


    def dampen(self,
               predicted_trend,
               y=None,
               trend_component=None,
               seasonality_component=None
               ):
        zeroed_trend_component = predicted_trend - predicted_trend[0]
        if self.damp_factor == 'auto':
            assert trend_component is not None, 'To use auto damp_factor you must pass a fitted trend component'
            assert seasonality_component is not None, 'To use auto damp_factor you must pass a fitted seasonality component, if the series is non-seasonal then pass an array of all zeros with a len of your y'
            resids = y - (trend_component + seasonality_component)
            self.damp_factor = 1 - self.calc_trend_strength(resids,
                                                            y - seasonality_component)
        if self.damp_factor == 0:
            return predicted_trend
        if self.damp_factor == 1:
            return np.tile(predicted_trend[0], len(predicted_trend))
        if self.damp_style == 'smooth':
            tau = TrendDampen.get_tau_proposal(zeroed_trend_component,
                                               self.damp_factor)
            dampened_trend = TrendDampen.smooth_dampen(zeroed_trend_component,
                                                       tau)
        elif self.damp_style == 'linear':            
            dampened_trend = TrendDampen.linear_dampen(zeroed_trend_component,
                                                       self.damp_factor)
        else:
            raise NotImplementedError('That damp style is not implemented!')
        crossing = np.where(np.diff(np.sign(np.gradient(dampened_trend))))[0]
        if crossing.size > 0:
            crossing_point = crossing[0]
            max_idx = np.argmax(np.mean(np.gradient(zeroed_trend_component))*dampened_trend)
            bound = dampened_trend[max_idx]
            dampened_trend[crossing_point:] = bound   
            scale = self.damp_factor*zeroed_trend_component[-1]/dampened_trend[-1]
            dampened_trend = dampened_trend*scale
        return dampened_trend + predicted_trend[0]

#%%
if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('darkgrid')

    # read data from NY times
    df = pd.read_csv("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv")
    
    # engineer new cases
    df['new_cases'] = df.cases - df.cases.shift().fillna(0)

    df.date = pd.to_datetime(df.date)
    df.set_index('date',inplace=True)
    df['rolling_weekly_avg'] = df.new_cases.rolling(window=7).mean().fillna(0)
    
    # create timeseries readable by fbprophet
    ts = pd.DataFrame({'ds':df.index,'y':df.new_cases})
    from prophet import Prophet
    
    # instantiate the model and fit the timeseries
    prophet = Prophet()
    prophet.fit(ts)
    fitted = prophet.predict()
    
    # create a future data frame 
    future = prophet.make_future_dataframe(periods=25)
    forecast = prophet.predict(future)
    
    # display the most critical output columns from the forecast
    forecast[['ds','yhat','yhat_lower','yhat_upper']].head()
    
    # plot
    fig = prophet.plot(forecast)
    dampener = TrendDampen()
    dampened_trend = dampener.dampen(forecast['trend'].values[-25:],
                                     ts['y'].values,
                                     fitted['trend'].values)
    plt.plot(dampened_trend)
    plt.plot(forecast['trend'].values[-25:])
    #%%
    y = np.linspace(0, 100, 100)
    y_train = y[:80]
    future_y = y[80:]
    trend = y_train
    seasonality = np.zeros(len(y_train))
    future_trend = future_y
    plt.plot(future_trend, color='black', alpha=.5,label='Actual Trend', linestyle='dotted')
    for damp_factor in [.1, .3, .5, .7, .9, 'auto']:
        dampener = TrendDampen(damp_factor=damp_factor,
                               damp_style='smooth')
        dampened_trend = dampener.dampen(future_trend,
                                         y=y_train,
                                         trend_component=trend,
                                         seasonality_component=seasonality)
        plt.plot(dampened_trend, label=damp_factor, alpha=.7)
    plt.legend()
    plt.show()

    #%%
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
from prophet import Prophet
import seaborn as sns
sns.set_style('darkgrid')

train_df = pd.read_csv(r'C:\Users\er90614\\downloads\m4-weekly-train.csv')
test_df = pd.read_csv(r'C:\Users\er90614\\downloads\m4-weekly-test.csv')
train_df.index = train_df['V1']
train_df = train_df.drop('V1', axis = 1)
test_df.index = test_df['V1']
test_df = test_df.drop('V1', axis = 1)


def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) +       np.abs(F)))


seasonality = 52
no_damp_smapes = []
damp_smapes = []
naive_smape = []
j = tqdm(range(len(train_df)))
for row in j:
    print(row)
    row = 51
    y = train_df.iloc[row, :].dropna()
    y = y.iloc[-(3*seasonality):]
    y_test = test_df.iloc[row, :].dropna()
    ds = pd.date_range(start='01-01-2000',
                       periods=len(y) + len(y_test),
                       freq='W')
    ts = y.to_frame()
    ts.columns = ['y']
    ts['ds'] = ds[:len(y)]
    j.set_description(f'{np.mean(no_damp_smapes)}, {np.mean(damp_smapes)}')
    prophet = Prophet()
    prophet.fit(ts)
    fitted = prophet.predict()

    # create a future data frame
    future = prophet.make_future_dataframe(freq='W',periods=len(y_test))
    forecast = prophet.predict(future)
    # prophet.plot(forecast)
    # prophet.plot_components(forecast)


    predictions = forecast.tail(len(y_test))
    predicted_trend = predictions['trend'].values
    trend_component = fitted['trend'].values
    seasonality_component = fitted['additive_terms'].values

    dampener = TrendDampen(damp_factor='auto',
                            damp_style='smooth')
    dampened_trend = dampener.dampen(predicted_trend,
                                      y=y,
                                      trend_component=trend_component,
                                      seasonality_component=seasonality_component)

    forecasts_no_dampen = predictions['yhat'].values
    forecasts_damped = predictions['additive_terms'].values + dampened_trend

    plt.plot(np.append(fitted['yhat'].values, forecasts_no_dampen), label='Normal Trend')
    plt.plot(np.append(fitted['yhat'].values, forecasts_damped), label='Dampened Trend')
    plt.plot(y.values)
    plt.legend()
    plt.title(row)
    plt.show()

    plt.plot(np.append(fitted['yhat'].values, dampened_trend), label='Dampened Trend')
    plt.plot(np.append(fitted['yhat'].values, predicted_trend), label='Normal Trend')
    plt.legend()
    plt.show()
    no_damp_smapes.append(smape(y_test.values, forecasts_no_dampen))
    damp_smapes.append(smape(y_test.values, forecasts_damped))
    naive_smape.append(smape(y_test.values, np.tile(y.iloc[-1], len(y_test))))
print(f'Weekly {np.mean(no_damp_smapes)}')
print(f'Weekly {np.mean(damp_smapes)}')
print(f'Naive {np.mean(naive_smape)}')


#weekly
# Weekly 10.38596475293425
# Weekly 10.139633844695672
# Naive 9.161286913982

#monthly
# monthly 4.729624009396322
# monthly 4.358958734903725
# Naive 3.0452517800636607

