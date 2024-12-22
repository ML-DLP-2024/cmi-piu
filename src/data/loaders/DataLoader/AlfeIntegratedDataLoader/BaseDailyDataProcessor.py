class BaseDailyDataProcessor:
    def __init__(self):
        self.granularity_in_hours = 1.0
        self.period_range = lambda: range(0, int(24 / self.granularity_in_hours))
        self.num_periods = len(self.period_range())
