# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


class LinearBasisFunction:

    def __init__(self,
                 n_changepoints,
                 decay=None,
                 weighted=True,
                 basis_difference=False):
        self.n_changepoints = n_changepoints
        self.decay = decay
        self.weighted = weighted
        self.basis_difference = basis_difference

    def get_basis(self, y):
        y = y.copy()
        y -= y.iloc[0]
        mean_y = np.mean(y)
        self.length = len(y)
        n_changepoints = self.n_changepoints
        array_splits = np.array_split(np.array(y),n_changepoints + 1)[:-1]
        if self.weighted:
            initial_point = y.iloc[0]
            final_point = y.iloc[-1]
        else:
            initial_point = 0
            final_point = 0
        changepoints = np.zeros(shape=(len(y), n_changepoints+1))
        len_splits = 0
        for i in range(n_changepoints):
            len_splits += len(array_splits[i])
            if self.weighted:
                moving_point = array_splits[i][-1]
            else:
                moving_point = 1
            left_basis = np.linspace(initial_point,
                                     moving_point,
                                     len_splits)
            end_point = self.add_decay(moving_point, final_point, mean_y)
            right_basis = np.linspace(moving_point,
                                      end_point,
                                      len(y) - len_splits + 1)
            changepoints[:, i] = np.append(left_basis, right_basis[1:])
        changepoints[:, i+1] = np.arange(0, len(y))
        if self.basis_difference and self.n_changepoints > 1:
            r,c = np.triu_indices(changepoints.shape[1],1)
            changepoints = changepoints[:,r] - changepoints[:,c]
        return changepoints

    def add_decay(self, moving_point, final_point, mean_point):
            if self.decay is None:
                return final_point
            else:
                if self.decay == 'auto':
                    dd = max(.001, min(.99, moving_point**2 / (mean_point**2)))
                    return moving_point - ((moving_point - final_point) * (1 - dd))
                else:
                    return moving_point - ((moving_point - final_point) * (1 - self.decay))

    def get_future_basis(self, basis_functions, forecast_horizon):
            n_components = np.shape(basis_functions)[1]
            slopes = np.gradient(basis_functions)[0][-1, :]
            future_basis = np.array(np.arange(0, forecast_horizon + 1))
            future_basis += len(basis_functions)
            future_basis = np.transpose([future_basis] * n_components)
            future_basis = future_basis * slopes
            future_basis = future_basis + (basis_functions[-1, :] - future_basis[0, :])
            return future_basis[1:, :]