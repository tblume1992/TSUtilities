# -*- coding: utf-8 -*-
import numpy as np


class FourierBasisFunction:

    def __init__(self, seasonal_weights=None):
        self.seasonal_weights = seasonal_weights
        if self.seasonal_weights is not None:
            self.seasonal_weights = np.array(self.seasonal_weights).reshape((-1, 1))

    def get_fourier_series(self, length, seasonal_period, fourier_order):
        x = 2 * np.pi * np.arange(1, fourier_order + 1) / seasonal_period
        t = np.arange(1, length + 1)
        x = x * t[:, None]
        fourier_series = np.concatenate((np.cos(x), np.sin(x)), axis=1)
        return fourier_series
    
    def get_basis(self, length, seasonal_period, fourier_order):
        harmonics = self.get_fourier_series(length, seasonal_period, fourier_order)
        if self.seasonal_weights is not None:
            harmonics = harmonics * self.seasonal_weights
        return harmonics

    def get_future_basis(self, length, forecast_horizon, seasonal_period, fourier_order):
        total_length = length + forecast_horizon
        future_harmonics = self.get_fourier_series(total_length, 
                                                   seasonal_period,
                                                   fourier_order)
        if self.seasonal_weights is None:
            return future_harmonics[length:, :]
        else:
            return future_harmonics[length:, :] * self.seasonal_weights[-1]