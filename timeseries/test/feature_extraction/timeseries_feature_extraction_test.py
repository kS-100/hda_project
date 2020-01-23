import unittest

from src.library.test.timeseries import base_test
from src.library.timeseries.timeseries_feature_extraction import get_rolling_timeseries


class TimeseriesFeatureExtractionTestCase(base_test.TimeseriesTest):

    def test_get_rolling_timeseries__window_size_one__same_dataframe_is_returned(self):
        test_df = self.get_test_dataframe()
        rolled_df = get_rolling_timeseries(
            df_x=test_df,
            start_index=0,
            lag=0,
            window_start=0,
            window_end=0)
        assert len(test_df) == len(rolled_df)
        assert (test_df.index == rolled_df.window_id).all()
        assert (list(range(10)) == rolled_df.A).all()
        assert (list(range(10, 20)) == rolled_df.B).all()
        assert (list(range(20, 30)) == rolled_df.C).all()

    def test_get_rolling_timeseries__window_size_one_with_different_offsets__dataframe_minus_offset_is_returned(self):
        test_df = self.get_test_dataframe()
        for i in range(1, len(test_df)):
            rolled_df = get_rolling_timeseries(
                df_x=test_df,
                start_index=i,
                lag=0,
                window_start=i,
                window_end=i)
            assert len(test_df) - i == len(rolled_df)
            assert (test_df.index[:-i] == rolled_df.window_id).all()
            assert (test_df.A[:-i].values == rolled_df.A.values).all()
            assert (test_df.B[:-i].values == rolled_df.B.values).all()
            assert (test_df.C[:-i].values == rolled_df.C.values).all()

    def test_get_rolling_timeseries__changing_start_index_over_zero__sub_dataframe_is_returned(self):
        test_df = self.get_test_dataframe()
        for i in range(1, len(test_df)):
            rolled_df = get_rolling_timeseries(
                df_x=test_df,
                start_index=i,
                lag=0,
                window_start=0,
                window_end=0)
            print(rolled_df)
            print(test_df.index[:-i])
            assert len(test_df) - i == len(rolled_df)
            assert (test_df.index[:-i] == rolled_df.window_id).all()
            assert (test_df.A[i:].values == rolled_df.A.values).all()
            assert (test_df.B[i:].values == rolled_df.B.values).all()
            assert (test_df.C[i:].values == rolled_df.C.values).all()

    def test_get_rolling_timeseries__changing_start_index_zero__sub_dataframe_is_returned(self):
        test_df = self.get_test_dataframe()
        rolled_df = get_rolling_timeseries(
            df_x=test_df,
            start_index=0,
            lag=0,
            window_start=0,
            window_end=0)
        print(rolled_df)
        print(test_df.index[:])
        assert len(test_df) == len(rolled_df)
        assert (test_df.index[:] == rolled_df.window_id).all()
        assert (test_df.A[:].values == rolled_df.A.values).all()
        assert (test_df.B[:].values == rolled_df.B.values).all()
        assert (test_df.C[:].values == rolled_df.C.values).all()

    def test_get_rolling_timeseries__changing_lag_over_zero__sub_dataframe_is_returned(self):
        test_df = self.get_test_dataframe()
        for i in range(1, len(test_df)):
            rolled_df = get_rolling_timeseries(
                df_x=test_df,
                start_index=0,
                lag=i,
                window_start=0,
                window_end=0)
            assert len(test_df) - i == len(rolled_df)
            assert (test_df.index[:-i] == rolled_df.window_id).all()
            assert (test_df.A[:-i].values == rolled_df.A.values).all()
            assert (test_df.B[:-i].values == rolled_df.B.values).all()
            assert (test_df.C[:-i].values == rolled_df.C.values).all()

    def test_get_rolling_timeseries__changing_lag_zero__sub_dataframe_is_returned(self):
        test_df = self.get_test_dataframe()
        rolled_df = get_rolling_timeseries(
            df_x=test_df,
            start_index=0,
            lag=0,
            window_start=0,
            window_end=0)
        assert len(test_df) == len(rolled_df)
        assert (test_df.index[:] == rolled_df.window_id).all()
        assert (test_df.A[:].values == rolled_df.A.values).all()
        assert (test_df.B[:].values == rolled_df.B.values).all()
        assert (test_df.C[:].values == rolled_df.C.values).all()

    def test_get_rolling_timeseries__increasing_window_size(self):
        test_df = self.get_test_dataframe()
        for i in range(1, len(test_df)):
            rolled_df = get_rolling_timeseries(
                df_x=test_df,
                start_index=i,
                lag=0,
                window_start=0,
                window_end=i)
            assert (i+1)*(len(test_df)-i) == len(rolled_df)
            assert (rolled_df.groupby('window_id').size() == i+1).all()
            #todo check if content is right



if __name__ == '__main__':
    unittest.main()
