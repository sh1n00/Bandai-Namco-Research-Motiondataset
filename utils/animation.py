import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from functools import singledispatch
import torch

from utils import settings


@singledispatch
def create_gif(filepath: str, is_show: bool = False) -> None:
    _, filename = os.path.split(filepath)
    filename, ext = os.path.splitext(filename)
    df = pd.read_csv(f"../data/raw/position/{filename}{ext}")

    timestamp = df["time"]
    df.drop("time", axis=1, inplace=True)
    root_pos = df[["joint_Root.x", "joint_Root.y", "joint_Root.z"]]
    df.drop(["joint_Root.x", "joint_Root.y", "joint_Root.z"], axis=1, inplace=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter([], [], [], color="green")

    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_zscale("linear")

    x_columns = df.columns.str.contains('x')
    y_columns = df.columns.str.contains('y')
    z_columns = df.columns.str.contains('z')
    x_max, x_min = df.loc[:, x_columns].values.max(), df.loc[:, x_columns].values.min()
    y_max, y_min = df.loc[:, y_columns].values.max(), df.loc[:, y_columns].values.min()
    z_max, z_min = df.loc[:, z_columns].values.max(), df.loc[:, z_columns].values.min()

    ax.set_xlim(x_min - settings.BUFFER, x_max + settings.BUFFER)
    ax.set_ylim(y_min - settings.BUFFER, y_max + settings.BUFFER)
    ax.set_zlim(z_min - settings.BUFFER, z_max + settings.BUFFER)

    ax.set_xlabel("x-axis [cm]")
    ax.set_ylabel("y-axis [cm]")
    ax.set_zlabel("z-axis [cm]")

    def update(frame):
        df_i = df.iloc[frame, :]
        x, y, z = [], [], []
        for index in range(0, len(df_i), 3):
            xi, yi, zi = df_i[index:index + 3]
            x.append(xi)
            y.append(yi)
            z.append(zi)
        sc._offsets3d = (x, y, z)

    ani = animation.FuncAnimation(fig, update, frames=len(df), interval=50)
    ani.save(os.path.join(settings.RESULT_DIR, "gif", f"{filename}_label.gif"), writer="imagemagick")
    if is_show:
        plt.show()


@create_gif.register
def _(df: pd.DataFrame, output_name: str, is_show: bool = False) -> None:
    filename, _ = os.path.splitext(output_name)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter([], [], [], color="green")

    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_zscale("linear")

    x_columns = df.columns.str.contains('x')
    y_columns = df.columns.str.contains('y')
    z_columns = df.columns.str.contains('z')
    x_max, x_min = df.loc[:, x_columns].values.max(), df.loc[:, x_columns].values.min()
    y_max, y_min = df.loc[:, y_columns].values.max(), df.loc[:, y_columns].values.min()
    z_max, z_min = df.loc[:, z_columns].values.max(), df.loc[:, z_columns].values.min()

    ax.set_xlim(x_min - settings.BUFFER, x_max + settings.BUFFER)
    ax.set_ylim(y_min - settings.BUFFER, y_max + settings.BUFFER)
    ax.set_zlim(z_min - settings.BUFFER, z_max + settings.BUFFER)

    ax.set_xlabel("x-axis [cm]")
    ax.set_ylabel("y-axis [cm]")
    ax.set_zlabel("z-axis [cm]")

    def update(frame):
        df_i = df.iloc[frame, :]
        x, y, z = [], [], []
        for index in range(0, len(df_i), 3):
            xi, yi, zi = df_i[index:index + 3]
            x.append(xi)
            y.append(yi)
            z.append(zi)
        sc._offsets3d = (x, y, z)

    ani = animation.FuncAnimation(fig, update, frames=len(df), interval=100)
    ani.save(os.path.join(settings.RESULT_DIR, "gif", f"{filename}_label.gif"), writer="imagemagick")
    if is_show:
        plt.show()


@create_gif.register
def _(pred: torch.Tensor, output_name: str, is_show: bool = False) -> None:
    filename, _ = os.path.splitext(output_name)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter([], [], [], color="blue")

    x = np.array([pred[:, i].detach().numpy() for i in range(0, pred.shape[1], 3)])
    y = np.array([pred[:, i].detach().numpy() for i in range(1, pred.shape[1], 3)])
    z = np.array([pred[:, i].detach().numpy() for i in range(2, pred.shape[1], 3)])
    ax.set_xlim(x.min() - settings.BUFFER, x.max() + settings.BUFFER)
    ax.set_ylim(y.min() - settings.BUFFER, y.max() + settings.BUFFER)
    ax.set_zlim(z.min() - settings.BUFFER, z.max() + settings.BUFFER)

    ax.set_xlabel("x-axis [cm]")
    ax.set_ylabel("y-axis [cm]")
    ax.set_zlabel("z-axis [cm]")

    def update(frame):
        df_i = pred[frame]
        x, y, z = [], [], []
        for index in range(0, len(df_i), 3):
            xi, yi, zi = df_i[index:index + 3]
            x.append(xi.item())
            y.append(yi.item())
            z.append(zi.item())
        sc._offsets3d = (x, y, z)

    ani = animation.FuncAnimation(fig, update, frames=len(pred), interval=100)
    ani.save(os.path.join(settings.RESULT_DIR, "gif", f"{filename}_pred.gif"), writer="imagemagick")
    if is_show:
        plt.show()


def create_git_combine(label: pd.DataFrame, pred: torch.Tensor, output_name: str, is_show: bool = False) -> None:
    filename, _ = os.path.splitext(output_name)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc1 = ax.scatter([], [], [], label="pred", color="blue")
    sc2 = ax.scatter([], [], [], label="label", color="green")

    x = np.array([pred[:, i].detach().numpy() for i in range(0, pred.shape[1], 3)])
    y = np.array([pred[:, i].detach().numpy() for i in range(1, pred.shape[1], 3)])
    z = np.array([pred[:, i].detach().numpy() for i in range(2, pred.shape[1], 3)])

    x_columns = label.columns.str.contains('x')
    y_columns = label.columns.str.contains('y')
    z_columns = label.columns.str.contains('z')
    x_max, x_min = label.loc[:, x_columns].values.max(), label.loc[:, x_columns].values.min()
    y_max, y_min = label.loc[:, y_columns].values.max(), label.loc[:, y_columns].values.min()
    z_max, z_min = label.loc[:, z_columns].values.max(), label.loc[:, z_columns].values.min()

    ax.set_xlim(min(x.min(), x_min) - settings.BUFFER, max(x.max(), x_max) + settings.BUFFER)
    ax.set_ylim(min(y.min(), y_min) - settings.BUFFER, max(y.max(), y_max) + settings.BUFFER)
    ax.set_zlim(min(z.min(), z_min) - settings.BUFFER, max(z.max(), z_max) + settings.BUFFER)

    ax.set_xlabel("x-axis [cm]")
    ax.set_ylabel("y-axis [cm]")
    ax.set_zlabel("z-axis [cm]")

    ax.legend()

    def update(frame):
        df_pred = pred[frame]
        df_label = label.iloc[frame, :]
        x_pred, y_pred, z_pred = [], [], []
        x_label, y_label, z_label = [], [], []
        for index in range(0, len(df_label), 3):
            xi_p, yi_p, zi_p = df_pred[index:index + 3]
            xi_l, yi_l, zi_l = df_label[index:index + 3]
            x_pred.append(xi_p.item())
            y_pred.append(yi_p.item())
            z_pred.append(zi_p.item())
            x_label.append(xi_l)
            y_label.append(yi_l)
            z_label.append(zi_l)
        sc1._offsets3d = (x_pred, y_pred, z_pred)
        sc2._offsets3d = (x_label, y_label, z_label)

    ani = animation.FuncAnimation(fig, update, frames=len(pred), interval=100)
    ani.save(os.path.join(settings.RESULT_DIR, "gif", f"{filename}_combine.gif"), writer="imagemagick")
    if is_show:
        plt.show()
