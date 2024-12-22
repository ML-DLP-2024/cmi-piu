# type: ignore

from typing import Any
import pandas as pd
import numpy as np
from .BaseDailyDataProcessor import BaseDailyDataProcessor

class EngineerFeaturesFromPeriodicActivityLevels(BaseDailyDataProcessor):
    def __init__(self, dfs: tuple[pd.DataFrame, pd.DataFrame]):
        """
        dfs = (df, aldf)

        df: all_days_df as returned by
            CalculateDailyPeriodicActivityRelatedValues.process()
        
        aldf: activity-level-per-days dataframe : {
            "day": [],
            "daily_periodic_activity_levels": [],
        }
        """
        super().__init__()
        self.all_days_df, self.aldf = dfs

    epsilon = 0.1
    SLEEP_AL = -1.0
    MODERATE_AL = 0
    VIGOROUS_AL = 1.5
    NIGHTTIME_SPT_WINDOW_TOLERANCE = 3

    def _detect_nighttime_spt_window(self, periodic_activity_levels: np.ndarray[Any]):
        look_left = look_right = True
        # Because w_start decreases and w_end increases over iterations
        # we set w_start to hour 24 initially, so that when there is an overlap,
        # we have w_start <= w_end.
        w_start = int(24 / self.granularity_in_hours) # 0
        w_end = int(4 / self.granularity_in_hours)

        # if w_start changes then change this one too!
        sleep_periods = np.arange(w_end+1)
        # if w_start or w_end changes then change this one too!
        d = np.sum(np.array(periodic_activity_levels[0:w_end+1]) >= self.SLEEP_AL)

        N = len(periodic_activity_levels)

        def get_max_k_right():
            k = self.NIGHTTIME_SPT_WINDOW_TOLERANCE
            points = []
            indices = []
            for i in range(w_end+1, w_end+k+1):
                points.append( periodic_activity_levels[i % N] )
                indices.append(i % N)
            for i in range(len(points) - 1, -1, -1):
                if points[i] + self.epsilon < self.SLEEP_AL:
                    break
            return np.array(points[:i+1]), np.array(indices[:i+1])

        def get_max_k_left():
            k = self.NIGHTTIME_SPT_WINDOW_TOLERANCE
            points = []
            indices = []
            for i in range(w_start-k, w_start):
                points.append( periodic_activity_levels[i % N] )
                indices.append(i % N)
            for i in range(len(points)):
                if points[i] + self.epsilon < self.SLEEP_AL:
                    break
            return np.array(points[i:]), np.array(indices[i:])

        while (look_left or look_right) and w_start > w_end:
            if look_left:
                points, indices = get_max_k_left()
                if len(points) == 0:
                    look_left = False
                else:
                    num_sleep_points = np.sum(points + self.epsilon < self.SLEEP_AL)
                    if num_sleep_points / len(points) < 0.5:
                        look_left = False
                    else:
                        w_start -= len(points)
                        d += len(points) - num_sleep_points
                        sleep_periods = np.concatenate((indices, sleep_periods))
            if look_right:
                points, indices = get_max_k_right()
                if len(points) == 0:
                    look_right = False
                else:
                    num_sleep_points = np.sum(points + self.epsilon < self.SLEEP_AL)
                    if num_sleep_points / len(points) < 0.5:
                        look_right = False
                    else:
                        w_end += len(points)
                        d += len(points) - num_sleep_points
                        sleep_periods = np.concatenate((sleep_periods, indices))
        
        if w_start <= w_end:
            return np.nan, np.nan, np.array([]), np.nan, np.nan

        w_duration = w_end % N + N - w_start % N

        return w_start % N, w_end % N, sleep_periods, w_duration, d
    
    def _percentage_of_moderate_and_vigorous_physical_movements(self, periodic_activity_levels: np.ndarray):
        """
        Calculates the percentage of moderate and vigorous physical movements in the given activity levels.
        Returns a tuple of (percentage of moderate movements, percentage of vigorous movements).
        """
        arr = periodic_activity_levels
        N_percent = 100 / len(arr)
        moderate = np.sum(arr + self.epsilon > self.MODERATE_AL)
        vigorous = np.sum(arr + self.epsilon > self.VIGOROUS_AL)
        moderate -= vigorous # type: ignore
        return moderate * N_percent, vigorous * N_percent

    def process(self) -> pd.DataFrame:
        data = {
            "day": [],
        }

        for day in self.aldf.index:
            data['day'].append(day)
            
            periodic_activity_levels = np.array(list(self.aldf[self.aldf.index == day]['daily_periodic_activity_levels'].iloc[0]))
            w_start, w_end, sleep_periods, w_duration, d = self._detect_nighttime_spt_window(periodic_activity_levels)
            sleep_start_hour = w_start * self.granularity_in_hours
            sleep_end_hour = w_end * self.granularity_in_hours
            # In case there is too little data on that day, and the data points are all sleep points,
            # then it is totally fine to have a sleep duration of 24 hours.
            # Hope the models can spot it out and learn from it.
            sleep_duration_in_hours = w_duration * self.granularity_in_hours
            if not np.isnan(sleep_duration_in_hours):
                sleep_duration_in_hours = min(24, sleep_duration_in_hours)
    
            sleep_start_earliness = (
                np.nan if np.isnan(sleep_start_hour)
                else -1 if sleep_start_hour < 12 and sleep_start_hour >= 7
                else 10 if sleep_start_hour <= 21
                else 9 if sleep_start_hour <= 22
                else 7 if sleep_start_hour <= 23
                else 5 if sleep_start_hour <= 24
                else 2
            )
    
            sleep_end_earliness = (
                np.nan if np.isnan(sleep_end_hour)
                else -1 if sleep_end_hour <= 3 or sleep_end_hour >= 13
                else 10 if sleep_end_hour <= 8
                else 9 if sleep_end_hour <= 9
                else 7 if sleep_end_hour <= 10
                else 5 if sleep_end_hour <= 11
                else 2
            )

            def aggregate_light(sleep_periods):
                day_df = self.all_days_df[self.all_days_df.index == day]['daily_periodic_df'].iloc[0].copy()
                day_lights = list(day_df['periodic_light_mean'])
                wake_lights = []
                sleep_lights = []
                for i in self.period_range():
                    if i not in sleep_periods:
                        wake_lights.append(day_lights[i])
                    else:
                        sleep_lights.append(day_lights[i])
                wake_lights_aggregates = pd.Series(wake_lights).agg(['mean', 'max'])
                sleep_lights_aggregates = pd.Series(sleep_lights).agg(['mean', 'max'])
                light_mean_diff = wake_lights_aggregates['mean'] - sleep_lights_aggregates['mean']
                light_max_diff = wake_lights_aggregates['max'] - sleep_lights_aggregates['max']
                return light_mean_diff, light_max_diff

            if np.isnan(w_start) or np.isnan(w_end):
                percent_moderate_activities_waketime = percent_vigorous_activities_waketime = np.nan
                # Defaulting sleep time to be 0h - 5h
                light_mean_diff, light_max_diff = aggregate_light(np.arange(0, 5 / self.granularity_in_hours + 1))
            else:
                waketime_activity_levels = []
                for i in self.period_range():
                    if i not in sleep_periods:
                        waketime_activity_levels.append(periodic_activity_levels[i])
                waketime_activity_levels = np.array(waketime_activity_levels)
                percent_moderate_activities_waketime, percent_vigorous_activities_waketime = self._percentage_of_moderate_and_vigorous_physical_movements(waketime_activity_levels)
    
                day_df = self.all_days_df[self.all_days_df.index == day]['daily_periodic_df'].iloc[0].copy()
                day_lights = list(day_df['periodic_light_mean'])
                wake_lights = []
                sleep_lights = []
                for i in self.period_range():
                    if i not in sleep_periods:
                        wake_lights.append(day_lights[i])
                    else:
                        sleep_lights.append(day_lights[i])
                wake_lights_aggregates = pd.Series(wake_lights).agg(['mean', 'max'])
                sleep_lights_aggregates = pd.Series(sleep_lights).agg(['mean', 'max'])
                light_mean_diff = wake_lights_aggregates['mean'] - sleep_lights_aggregates['mean']
                light_max_diff = wake_lights_aggregates['max'] - sleep_lights_aggregates['max']
    
            for col, value in {
                'percent_moderate_activities_waketime': percent_moderate_activities_waketime,
                'percent_vigorous_activities_waketime': percent_vigorous_activities_waketime,
                'sleep_duration_in_hours': sleep_duration_in_hours,
                'sleep_start_earliness': sleep_start_earliness,
                'sleep_end_earliness': sleep_end_earliness,
                'sleep_disturbance_duration_in_hours': d * self.granularity_in_hours,
                'light_mean_diff': light_mean_diff,
                'light_max_diff': light_max_diff,
            }.items():
                data[col] = data.get(col, [])
                data[col].append(value)

        df = pd.DataFrame(data)
        df.set_index('day', inplace=True)
        return df

    @classmethod
    def get_features(cls):
        return [
            'percent_moderate_activities_waketime',
            'percent_vigorous_activities_waketime',
            'sleep_duration_in_hours',
            'sleep_start_earliness',
            'sleep_end_earliness',
            'sleep_disturbance_duration_in_hours',
            'light_mean_diff',
            'light_max_diff'
        ]
