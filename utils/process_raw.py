import os
from glob import glob

import pandas as pd

import settings
import myenum
import converter


# poscsvのactionごとにフォルダを生成
def make_action_columns():
    raw_path = os.path.join(settings.ROOT_DIR, "data", "raw")
    pos_path = os.path.join(raw_path, "position")
    pos_files = sorted(glob(pos_path + "/*"))
    for pos_file in pos_files:
        df = pd.read_csv(pos_file)
        filename = pos_file.split("/")[-1]
        for action in myenum.Action.get_all_action():
            if action in pos_file:
                output_folder = os.path.join(settings.ROOT_DIR, "data", "processed", "action", action)
                os.makedirs(output_folder, exist_ok=True)
                df["action"] = action
                df.to_csv(os.path.join(output_folder, filename))
                break


# poscsvのstyleごとにフォルダを生成
def make_style_columns():
    raw_path = os.path.join(settings.ROOT_DIR, "data", "raw")
    pos_path = os.path.join(raw_path, "position")
    pos_files = sorted(glob(pos_path + "/*"))
    for pos_file in pos_files:
        df = pd.read_csv(pos_file)
        filename = pos_file.split("/")[-1]
        for style in myenum.Styles.get_all_styles():
            if style in pos_file:
                output_folder = os.path.join(settings.ROOT_DIR, "data", "processed", "style", style)
                os.makedirs(output_folder, exist_ok=True)
                df["style"] = style
                df.to_csv(os.path.join(output_folder, filename))
                break


def make_master_record_pos():
    raw_path = os.path.join(settings.ROOT_DIR, "data", "raw")
    pos_path = os.path.join(raw_path, "position")
    pos_files = sorted(glob(pos_path + "/*"))
    master = pd.DataFrame()
    for pos_file in pos_files:
        df = pd.read_csv(pos_file)
        action, style = pos_file.split("_")[-4:-2]
        df["action"] = action
        df["style"] = style
        master = pd.concat([master, df], axis=0)
    master.to_csv(os.path.join(settings.ROOT_DIR, "data", "processed", "master_pos.csv"))


def make_master_record_rot():
    raw_path = os.path.join(settings.ROOT_DIR, "data", "raw")
    rot_path = os.path.join(raw_path, "rotation")
    rot_files = sorted(glob(rot_path + "/*"))
    master = pd.DataFrame()
    for rot_file in rot_files:
        df = pd.read_csv(rot_file)
        action, style = rot_file.split("_")[-4:-2]
        df["action"] = action
        df["style"] = style
        master = pd.concat([master, df], axis=0)
    master.to_csv(os.path.join(settings.ROOT_DIR, "data", "processed", "master_rot.csv"))


def make_master():
    raw_path = os.path.join(settings.ROOT_DIR, "data", "raw")
    pos_path = os.path.join(raw_path, "position")
    rot_path = os.path.join(raw_path, "rotation")
    pos_files = sorted(glob(pos_path + "/*"))
    rot_files = sorted(glob(rot_path + "/*"))
    master = pd.DataFrame()
    for pos_file, rot_file in zip(pos_files, rot_files):
        df_pos = pd.read_csv(pos_file)
        df_rot = pd.read_csv(rot_file)
        action, style = pos_file.split("_")[-4:-2]
        df_pos = converter.convert_pos_columns(df_pos)
        df_rot = converter.convert_rot_columns(df_rot)
        df = pd.merge(df_pos, df_rot, on="time", how="inner")
        df["action"] = action
        df["style"] = style
        master = pd.concat([master, df], axis=0)
    master.to_csv(os.path.join(settings.ROOT_DIR, "data", "processed", "master.csv"))


if __name__ == "__main__":
    make_master()
    make_master_record_rot()
