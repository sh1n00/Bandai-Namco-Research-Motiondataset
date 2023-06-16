import os
from typing import overload

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from functools import singledispatch
import torch


# TODO: matplotlibのフォーマットを指定する
@singledispatch
def create_gif(obj) -> str:
    return "Invalid params"


@create_gif.register
def _(filepath: str) -> None:
    _, filename = os.path.split(filepath)
    filename, ext = os.path.splitext(filename)
    df = pd.read_csv(f"../position/{filename}{ext}")

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

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

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
    ani.save(os.path.join("..", "gif", f"{filename}_label.gif"), writer="imagemagick")
    plt.show()


@create_gif.register
def _(motion_frames: torch.Tensor, output_name: str) -> None:
    root_pos = motion_frames[:, :3]
    body_pos = motion_frames[:, 3:]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter([], [], [], color="blue")

    x = np.array([body_pos[:, i].detach().numpy() for i in range(0, body_pos.shape[1], 3)])
    y = np.array([body_pos[:, i].detach().numpy() for i in range(1, body_pos.shape[1], 3)])
    z = np.array([body_pos[:, i].detach().numpy() for i in range(2, body_pos.shape[1], 3)])
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())

    ax.set_xlabel("x-axis [cm]")
    ax.set_ylabel("y-axis [cm]")
    ax.set_zlabel("z-axis [cm]")

    def update(frame):
        df_i = body_pos[frame]
        x, y, z = [], [], []
        for index in range(0, len(df_i), 3):
            xi, yi, zi = df_i[index:index + 3]
            x.append(xi.item())
            y.append(yi.item())
            z.append(zi.item())
        sc._offsets3d = (x, y, z)

    ani = animation.FuncAnimation(fig, update, frames=len(body_pos), interval=50)
    ani.save(os.path.join("..", "gif", f"{output_name}_pred.gif"), writer="imagemagick")
    plt.show()


@create_gif.register
def _(label: pd.DataFrame, pred: torch.Tensor, output_name: str) -> None:
    root_pos = pred[:, :3]
    body_pos = pred[:, 3:]

    root_pos = label[["joint_Root.x", "joint_Root.y", "joint_Root.z"]]
    label.drop(["joint_Root.x", "joint_Root.y", "joint_Root.z"], axis=1, inplace=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc1 = ax.scatter([], [], [], label="pred", color="blue")
    sc2 = ax.scatter([], [], [], label="label", color="green")

    x = np.array([body_pos[:, i].detach().numpy() for i in range(0, body_pos.shape[1], 3)])
    y = np.array([body_pos[:, i].detach().numpy() for i in range(1, body_pos.shape[1], 3)])
    z = np.array([body_pos[:, i].detach().numpy() for i in range(2, body_pos.shape[1], 3)])
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())

    ax.set_xlabel("x-axis [cm]")
    ax.set_ylabel("y-axis [cm]")
    ax.set_zlabel("z-axis [cm]")

    ax.legend()

    def update(frame):
        df_pred = body_pos[frame]
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

    ani = animation.FuncAnimation(fig, update, frames=len(body_pos), interval=50)
    ani.save(os.path.join("..", "gif", f"{output_name}_combine.gif"), writer="imagemagick")
    plt.show()


if __name__ == "__main__":
    filename = "dataset-1_walk-back_angry_001_pos"
    create_gif(filename)
