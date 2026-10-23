
from pathlib import Path
import re

# ===== 設定 =====
folder = Path("/home/ubuntu/ros2_ws/src/map/test_tsukuba")
start_number = 336

# 最初は True で変更予定を確認
# 問題なければ False に変更して実行
dry_run = False
# =================

pattern = re.compile(
    r"^waypoint_map_rgb_(\d+)\.(png|yaml|jpeg)$"
)

# PNGとYAMLを番号ごとにまとめる
groups = {}

for path in folder.iterdir():
    match = pattern.match(path.name)
    if match:
        number = int(match.group(1))
        ext = match.group(2)
        groups.setdefault(number, {})[ext] = path

if not groups:
    raise SystemExit("対象ファイルが見つかりません。")

# 元の番号が小さい順に、新しい番号を割り当てる
number_map = {
    old: start_number + i
    for i, old in enumerate(sorted(groups))
}

# PNGとYAMLを一時名に変更してから連番にする
plans = []

for old_number in sorted(groups):
    new_number = number_map[old_number]

    for ext, old_path in groups[old_number].items():
        new_path = folder / f"waypoint_map_rgb_{new_number:03d}.{ext}"
        plans.append((old_path, new_path, old_number, new_number, ext))

print("変更予定:")
for old_path, new_path, *_ in plans:
    print(f"{old_path.name} -> {new_path.name}")

if dry_run:
    print("\n確認モードのため、まだ変更していません。")
    print("問題なければ dry_run = False にしてください。")
    raise SystemExit

# 変更先に既存ファイルがある場合は中止
source_paths = {p.resolve() for p, _, *_ in plans}

for _, new_path, *_ in plans:
    if new_path.exists() and new_path.resolve() not in source_paths:
        raise SystemExit(f"変更先が既に存在します: {new_path}")

# 先に一時名へ変更し、名前の衝突を防ぐ
temporary = []

for old_path, new_path, *_ in plans:
    temp_path = old_path.with_name(old_path.name + ".rename_tmp")
    if temp_path.exists():
        raise SystemExit(f"一時ファイルが既に存在します: {temp_path}")
    old_path.rename(temp_path)
    temporary.append((temp_path, new_path))

for temp_path, new_path in temporary:
    temp_path.rename(new_path)

# YAML内のimageパスも新しいPNG名に更新
for old_number in sorted(groups):
    new_number = number_map[old_number]
    yaml_path = folder / f"waypoint_map_rgb_{new_number:03d}.yaml"

    if not yaml_path.exists():
        continue

    content = yaml_path.read_text(encoding="utf-8")

    content = re.sub(
        r"(?m)^(\s*image:\s*).*/waypoint_map_rgb_\d+\.png\s*$",
        lambda m: (
            m.group(1)
            + str(folder / f"waypoint_map_rgb_{new_number:03d}.png")
        ),
        content,
    )

    yaml_path.write_text(content, encoding="utf-8")

print("\nファイル名とYAML内のimageパスを変更しました。")

