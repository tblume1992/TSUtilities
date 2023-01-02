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

