import pandas as pd

from utils import myenum


def convert_pos_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = myenum.DefaultColumns.get_all_default_columns()
    pos = myenum.PositionColumns.get_all_positions()
    df.rename(columns=dict(zip(columns, pos)), inplace=True)
    return df


def convert_rot_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = myenum.DefaultColumns.get_all_default_columns()
    rots = myenum.RotationColumns.get_all_rotations()
    df.rename(columns=dict(zip(columns, rots)), inplace=True)
    return df
