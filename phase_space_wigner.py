"""
phase_space_wigner.py

绘制 vacuum state、displaced vacuum state、squeezed state、
displaced squeezed state 的 phase space / Wigner function 示意图。

运行：
    python phase_space_wigner.py

输出：
    phase_space_wigner.png
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Microsoft YaHei"]   # 微软雅黑
rcParams["axes.unicode_minus"] = False              # 解决负号显示问题

def gaussian_wigner(X, P, x0=0.0, p0=0.0, sx=1 / np.sqrt(2), sp=1 / np.sqrt(2), theta=0.0):
    """
    一个归一化到最大值为 1 的 Gaussian Wigner function 示意图。
    这里使用无量纲变量 x, p，并令 ħ = 1。
    """
    dx = X - x0
    dp = P - p0

    # 旋转坐标
    ct, st = np.cos(theta), np.sin(theta)
    u = ct * dx + st * dp
    v = -st * dx + ct * dp

    W = np.exp(-0.5 * ((u / sx) ** 2 + (v / sp) ** 2))
    return W


def add_panel(
    ax,
    X,
    P,
    title,
    x0=0.0,
    p0=0.0,
    gamma=0.0,
    theta=0.0,
    displaced=False,
    orbit=False,
    time_label=None,
    formula=None,
):
    """
    gamma > 0: x 方向压缩，p 方向拉伸
    gamma < 0: p 方向压缩，x 方向拉伸
    """
    sx = np.exp(-gamma) / np.sqrt(2)
    sp = np.exp(gamma) / np.sqrt(2)

    W = gaussian_wigner(X, P, x0=x0, p0=p0, sx=sx, sp=sp, theta=theta)

    levels = np.linspace(0.05, 1.0, 15)
    cf = ax.contourf(X, P, W, levels=levels, cmap="coolwarm")
    ax.contour(X, P, W, levels=levels[::2], linewidths=0.5)

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)

    if orbit:
        radius = np.hypot(x0, p0)
        t = np.linspace(0, 2 * np.pi, 300)
        ax.plot(radius * np.cos(t), radius * np.sin(t), "k--", linewidth=1.1, alpha=0.7)
        angle = np.arctan2(p0, x0)
        arrow_start = angle - 0.35
        ax.annotate(
            "",
            xy=(radius * np.cos(angle), radius * np.sin(angle)),
            xytext=(radius * np.cos(arrow_start), radius * np.sin(arrow_start)),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        )

    ax.plot(x0, p0, marker="*", markersize=10, color="black")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    text = (
        rf"$\Delta x={np.exp(-gamma):.2g}/\sqrt{{2}}$" "\n"
        rf"$\Delta p={np.exp(gamma):.2g}/\sqrt{{2}}$" "\n"
        rf"$\Delta x\Delta p=1/2$"
    )
    if displaced:
        text += "\n" + rf"$\langle x\rangle={x0:.2g}$" + "\n" + rf"$\langle p\rangle={p0:.2g}$"
    if abs(theta) > 1e-12:
        text += "\n" + rf"$\theta={theta/np.pi:.2g}\pi$"
    if time_label is not None:
        text += "\n" + time_label

    text_y = 0.03 if orbit else 0.97
    text_va = "bottom" if orbit else "top"
    ax.text(
        0.97,
        text_y,
        text,
        transform=ax.transAxes,
        va=text_va,
        ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    if formula is not None:
        ax.text(
            0.03,
            0.03,
            formula,
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=8.5,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    return cf


def main():
    x = np.linspace(-8, 8, 500)
    p = np.linspace(-8, 8, 500)
    X, P = np.meshgrid(x, p)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)

    panels = [
        dict(
            title=r"1. Vacuum",
            x0=0,
            p0=0,
            gamma=0,
            theta=0,
            displaced=False,
            formula=r"$|\psi\rangle=|0\rangle$" "\n" r"$W\propto e^{-x^2-p^2}$",
        ),
        dict(
            title=r"2. Coherent (Displaced Vacuum)",
            x0=2,
            p0=1,
            gamma=0,
            theta=0,
            displaced=True,
            formula=r"$|\psi\rangle=D(\alpha)|0\rangle=|\alpha\rangle$" "\n" r"$W\propto e^{-(x-x_0)^2-(p-p_0)^2}$",
        ),
        dict(
            title=r"3. Squeezed Vacuum $(\gamma>0)$",
            x0=0,
            p0=0,
            gamma=0.5,
            theta=0,
            displaced=False,
            formula=r"$|\psi\rangle=S(\gamma)|0\rangle,\ \gamma>0$" "\n" r"$\Delta x=e^{-\gamma}/\sqrt{2}$",
        ),
        dict(
            title=r"4. Squeezed Vacuum $(\gamma<0)$",
            x0=0,
            p0=0,
            gamma=-0.5,
            theta=0,
            displaced=False,
            formula=r"$|\psi\rangle=S(\gamma)|0\rangle,\ \gamma<0$" "\n" r"$\Delta p=e^{\gamma}/\sqrt{2}$",
        ),
        dict(
            title=r"5. Rotated Squeezed Vacuum",
            x0=0,
            p0=0,
            gamma=0.5,
            theta=np.pi / 4,
            displaced=False,
            formula=r"$|\psi\rangle=R(\theta)S(\gamma)|0\rangle$" "\n" r"$\theta=\pi/4$",
        ),
        dict(
            title=r"6. Displaced Squeezed State",
            x0=2,
            p0=1,
            gamma=0.5,
            theta=0,
            displaced=True,
            formula=r"$|\psi\rangle=D(\alpha)S(\gamma)|0\rangle$" "\n" r"$(x_0,p_0)=(2,1)$",
        ),
        dict(
            title=r"7. Coherent State Time Evolution",
            x0=2 * np.cos(np.pi / 3),
            p0=2 * np.sin(np.pi / 3),
            gamma=0,
            theta=0,
            displaced=True,
            orbit=True,
            time_label=rf"$t=\pi/3$",
            formula=r"$|\psi(t)\rangle=|\alpha(t)\rangle$" "\n" r"$\alpha(t)=\alpha_0e^{-i\omega t}$",
        ),
        dict(
            title=r"8. Squeezed State Time Evolution",
            x0=2 * np.cos(np.pi / 3),
            p0=2 * np.sin(np.pi / 3),
            gamma=0.5,
            theta=np.pi / 3,
            displaced=True,
            orbit=True,
            time_label=rf"$t=\pi/3$",
            formula=r"$|\psi(t)\rangle=D[\alpha(t)]R(\omega t)S(\gamma)|0\rangle$" "\n" r"$\theta(t)=\omega t$",
        ),
    ]

    cf = None
    for ax, params in zip(axes.flat, panels):
        cf = add_panel(ax, X, P, **params)

    fig.suptitle("Phase Space Representation of Gaussian States", fontsize=18, fontweight="bold")

    cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.82)
    cbar.set_label(r"$W(x,p)$")

    explanation = (
        "无量纲变量，设 ħ = 1。\n"
        "Vacuum: 圆形 Gaussian，中心在原点。\n"
        "Displaced vacuum: 圆形 Gaussian 被平移，形状不变。\n"
        "Squeezed state: 圆形被压成椭圆，一个方向不确定性变小，另一个方向变大。\n"
        "Displaced squeezed state: 先 squeeze，再 displacement。"
    )
    fig.text(
        0.1,
        -0.03,
        explanation,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    output = "phase_space_wigner.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output}")


if __name__ == "__main__":
    main()
