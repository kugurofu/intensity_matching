import os
from PIL import Image

# 対象のフォルダパスを指定
folder_path = os.path.expanduser('~/ros2_ws/src/map/nakaniwa_0821')

# フォルダ内のpngファイルを検索し、jpegに変換
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".png"):
        png_path = os.path.join(folder_path, filename)
        jpeg_path = os.path.join(folder_path, os.path.splitext(filename)[0] + ".jpeg")

        # PNG画像を開く
        with Image.open(png_path) as img:
            # JPEGはアルファチャンネルをサポートしないためRGBに変換
            img = img.convert("RGB")
            img.save(jpeg_path, "JPEG")

        print(f"Converted {png_path} to {jpeg_path}")

print("All .png files have been converted to .jpeg.")
