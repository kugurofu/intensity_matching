#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import filedialog


# ==========================================================
# 設定
# ==========================================================

# YAMLと同じ解像度 [m/pixel]
# 例：0.05 m/pixel
RESOLUTION = 0.05


# ==========================================================
# グローバル変数
# ==========================================================

image = None
display_image = None

# クリックした座標
clicked_x = None
clicked_y = None


# ==========================================================
# 画像ファイル選択
# ==========================================================

def select_image():

    # Tkinterのメインウィンドウを非表示
    root = tk.Tk()
    root.withdraw()

    # ファイル選択ダイアログ
    image_path = filedialog.askopenfilename(
        title="Waypoint画像を選択してください",
        filetypes=[
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

    return image_path


# ==========================================================
# マウスコールバック
# ==========================================================

def mouse_callback(event, x, y, flags, param):

    global display_image
    global clicked_x
    global clicked_y

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    h, w = image.shape[:2]

    # ------------------------------------------------------
    # 画像中心を原点とした座標に変換
    #
    # 右方向 : +X
    # 左方向 : -X
    # 上方向 : +Y
    # 下方向 : -Y
    # ------------------------------------------------------

    center_x = w / 2.0
    center_y = h / 2.0

    clicked_x = (x - center_x) * RESOLUTION
    clicked_y = (center_y - y) * RESOLUTION

    print(
        f"pixel=({x}, {y})  "
        f"XY=({clicked_x:.3f}, {clicked_y:.3f}) m"
    )

    # ------------------------------------------------------
    # 表示画像を更新
    # ------------------------------------------------------

    display_image = image.copy()

    # 中心
    cv2.drawMarker(
        display_image,
        (int(center_x), int(center_y)),
        (255, 0, 0),
        cv2.MARKER_CROSS,
        30,
        2
    )

    # クリック位置
    cv2.drawMarker(
        display_image,
        (x, y),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        30,
        2
    )

    # 座標表示
    text = f"X={clicked_x:.3f} m, Y={clicked_y:.3f} m"

    cv2.putText(
        display_image,
        text,
        (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.imshow(
        "Waypoint XY Selector",
        display_image
    )


# ==========================================================
# メイン
# ==========================================================

def main():

    global image
    global display_image

    # ------------------------------------------------------
    # 起動時に画像を選択
    # ------------------------------------------------------

    image_path = select_image()

    # キャンセルされた場合
    if not image_path:
        print("画像が選択されませんでした。")
        return

    # ------------------------------------------------------
    # 画像読み込み
    # ------------------------------------------------------

    image = cv2.imread(
        image_path,
        cv2.IMREAD_COLOR
    )

    if image is None:
        print("画像を読み込めませんでした:")
        print(image_path)
        return

    h, w = image.shape[:2]

    # ------------------------------------------------------
    # 情報表示
    # ------------------------------------------------------

    print("========================================")
    print(" Waypoint XY Selector")
    print("========================================")
    print(f"Image      : {image_path}")
    print(f"Size       : {w} x {h} pixel")
    print(f"Resolution : {RESOLUTION} m/pixel")
    print()
    print("画像をクリックすると座標を表示します。")
    print("画像中心が (0, 0) です。")
    print("右方向 : +X")
    print("左方向 : -X")
    print("上方向 : +Y")
    print("下方向 : -Y")
    print()
    print("ESC または q : 終了")
    print("========================================")

    # ------------------------------------------------------
    # 初期表示画像
    # ------------------------------------------------------

    display_image = image.copy()

    # 中心座標
    center_x = int(w / 2)
    center_y = int(h / 2)

    cv2.drawMarker(
        display_image,
        (center_x, center_y),
        (255, 0, 0),
        cv2.MARKER_CROSS,
        30,
        2
    )

    cv2.putText(
        display_image,
        "(0, 0)",
        (center_x + 10, center_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    # ------------------------------------------------------
    # ウィンドウ
    # ------------------------------------------------------

    cv2.namedWindow(
        "Waypoint XY Selector"
    )

    cv2.setMouseCallback(
        "Waypoint XY Selector",
        mouse_callback
    )

    cv2.imshow(
        "Waypoint XY Selector",
        display_image
    )

    # ------------------------------------------------------
    # キー入力
    # ------------------------------------------------------

    while True:

        key = cv2.waitKey(20) & 0xFF

        if key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()


# ==========================================================
# 実行
# ==========================================================

if __name__ == "__main__":
    main()
