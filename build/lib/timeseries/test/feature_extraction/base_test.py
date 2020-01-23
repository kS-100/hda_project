import unittest
import pandas as pd
import dask.dataframe as dd

class TimeseriesTest(unittest.TestCase):

    def get_test_dataframe(self):
        data = {
            'A': list(range(10)),
            'B': list(range(10,20)),
            'C': list(range(20, 30))
        }
        return pd.DataFrame(data)

