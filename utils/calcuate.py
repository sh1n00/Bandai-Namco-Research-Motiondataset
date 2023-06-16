import pandas as pd
import settings


# 関節の速度を計算する
def calc_velocity(df: pd.DataFrame) -> pd.DataFrame:
    diff_pos = df[settings.POSITIONCOLUMNS].diff()
    diff_t = df["time"].diff()
    df_velocity = diff_pos.div(diff_t, axis=0)

    # 最初の速度は0
    df_velocity.fillna(0.0, inplace=True)
    df_velocity.columns = settings.VELOCITYCOLUMNS
    return pd.concat([df, df_velocity], axis=1)


# 関節の加速度を計算する
def calc_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    diff_vec = df[settings.VELOCITYCOLUMNS].diff()
    diff_t = df["time"].diff()
    df_acceleration = diff_vec.div(diff_t, axis=0)

    # 最初の加速度は0
    df_acceleration.fillna(0.0, inplace=True)
    df_acceleration.columns = settings.ACCELERATIONCOLUMNS
    return pd.concat([df, df_acceleration], axis=1)


if __name__ == "__main__":
    df = calc_velocity(pd.read_csv(f"../position/dataset-1_walk-back_angry_001_pos.csv"))
    print(df)
    print(calc_acceleration(df))
