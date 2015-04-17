"""
No resampling needed for these experiments (this
functionality is built in for other experiments,
but has been removed here).
"""

class NullResamplingConfiguration(object):

    def __init__(self, dataset):
        self.dataset = dataset

    def get_settings(self):
        return [{}]

    def get_resampled(self):
        return self.dataset

    def get_all_resampled(self):
        return [self.get_resampled(**setting)
                for setting in self.get_settings()]
