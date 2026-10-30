# intensity_matching

## 1. 概要

本リポジトリは、3D-LiDARを用いた自律移動ロボットの自己位置推定および地図生成を行うROS 2パッケージである。

点群データを高さ方向に分類し、複数の特徴を保持したマルチレイヤ地図を生成する。生成したGlobal MapとLiDARから取得したLocal Mapをテンプレートマッチングによって照合することで、ロボットの自己位置を推定する。

また、生成した地図をWaypoint情報と組み合わせて使用し、`orange2026A`の`path_follower`と連携して自律走行中の停止処理を行う。

### 本システムの主な機能

- 3D-LiDAR点群の取得および高さ方向の分類
- Global MapおよびLocal Mapの生成
- マルチレイヤ地図を用いたテンプレートマッチング
- PSRおよび占有率を用いたマッチング結果の評価
- 自己位置推定結果の出力
- PNG及びJPEG画像を用いたWaypoint・仮想障害物の管理
- `orange2026A`の`path_follower`との停止処理の連携

## 2. リポジトリ構成

本リポジトリには、以下のパッケージが含まれる。

| パッケージ | 用途 |
|---|---|
| `intensity_matching` | 本システムで使用する地図生成・マッチング処理 |
| `pcd_filter` | テスト用に作成したパッケージ。現在のシステムでは使用しない |

**注意：** `pcd_filter`はテスト用のパッケージであり、現在の自己位置推定システムの動作には必要ない。通常のビルド・実行・引き継ぎでは、`intensity_matching`を中心に確認すること。

### 2.1 ファイル構成

ここに実際のディレクトリ構成を記載し、各Pythonファイル、C++ファイル、Launchファイル、設定ファイルの役割を説明する。

| ファイル | 役割 |
|---|---|
| `intensity_matching/launch/make_globalmap.launch.xml` | Global Map生成処理(launch) |
| `intensity_matching/launch/map_matching.launch.xml` | Local Map生成,マッチング処理(launch) |
| `intensity_matching/intensity_matching/reflection_intensity_globalmap_multi_sub.py` |  Global Map生成処理(node) |
| `intensity_matching/intensity_matching/reflection_intensity_localmap_multi_sub.py` | Local Map生成処理(node) |
| `intensity_matching/intensity_matching/png_map_matching.py` | マッチング処理(node) |
| `intensity_matching/intensity_matching/png_to_jpeg.py` | PNGファイルをJPEGファイルに変換(python) |
| `intensity_matching/intensity_matching/rename_number.py` | PNG,jpeg,yamlファイルの番号を変更(python) |
| `intensity_matching/intensity_matching/check_xy.py` | PNGファイル上で停止地点の確認(python) |

※各ファイルの詳細は「3. ファイル・ノードの役割」を参照。

※ここに記載のないファイルは基本的に使用しない。

## 3. ファイル・ノードの役割

各ノードの役割、処理内容、Publisher、Subscriberを記載する。

### 3.1 ノード一覧

| ノード名 | 役割 |
|---|---|
| `/reflection_global_map` | Global Map生成 |
| `/reflection_local_map` | Local Map生成 |
| `/waypoint_manager_maprun_node` | 自己位置推定,WaypointManager |

### 3.2 ノードごとの入出力トピック

各ノードについて、以下の形式で整理する。

#### ノード名：/reflection_global_map

**Subscriber**

| トピック | メッセージ型 | 用途 |
|---|---|---|
| `/pcd_rotation_merge` | PointCloud2 | LiDAR点群 |
| `/odom/combine` | Odometry | ローカルオドメトリ |
| `/fusion/odom` | Odometry | Global座標系のオドメトリ |

**出力**

/ros2_ws/src/map/test2/waypoint_map_rgb_xxx.png

/ros2_ws/src/map/test2/waypoint_map_rgb_xxx.yaml

#### ノード名：/reflection_local_map

**Subscriber**

| トピック | メッセージ型 | 用途 |
|---|---|---|
| `/pcd_rotation_merge` | PointCloud2 | LiDAR点群 |
| `/odom/combine` | Odometry | ローカルオドメトリ |
| `/fusion/odom` | Odometry | Global座標系のオドメトリ |

**Publisher**

| トピック | メッセージ型 | 用途 |
|---|---|---|
| `/rgb_reflect_map_local` | Image | ローカル画像 |

#### ノード名：/waypoint_manager_maprun_node

**Subscriber**

| トピック | メッセージ型 | 用途 |
|---|---|---|
| `/odom/combine` | Odometry | ローカルオドメトリ |
| `/fusion/odom` | Odometry | Global座標系のオドメトリ |
| `/rgb_reflect_map_local` | Image | ローカル画像 |
| `/goal_pose` | Posestamped | Global座標系のpose(手動waypoint設定用) |
| `/skip_matching_stop` | Int32 | stop_num管理(停止位置用) |

**Publisher**

| トピック | メッセージ型 | 用途 |
|---|---|---|
| `/current_waypoint` | PoseArray | Waypoint |
| `/waypoint_path` | Path | Waypoint Path |
| `/waypoint_number` | Int32 | Waypoint Number |
| `/waypoint_markers` | Marker | Waypoint Marker |
| `/waypoint_labels` | MarkerArray | Waypoint Marker Label |
| `/odom_ref_slam` | Odometry | マッチング自己位置推定 |
| `/odom_ekf_match` | Odometry | マッチング自己位置推定+ローカルオドメトリ |
| `/map_obs` | PointCloud2 | 仮想障害物 |
| `/stop_global_position` | PointStamped | 停止位置のPointStamped |
| `/stop_global_marker` | Marker | 停止位置のMarker |
| `/skip_path_stop` | Int32 | stop_num管理(停止位置用) |

**入力**

/ros2_ws/src/map/test2/waypoint_map_rgb_xxx.png

/ros2_ws/src/map/test2/waypoint_map_rgb_xxx.jpeg

/ros2_ws/src/map/test2/waypoint_map_rgb_xxx.yaml

**出力**

stop_flag アクション(path_follower)

## 4. 地図生成処理

### 4.1 座標系

Local点群はTFを用いてGlobal座標系へ変換する。

地図の座標はGlobalの`odom`座標系を基準として保存する。そのため、Local点群の位置情報をGlobal座標系に変換したうえで、地図上に反映する。

**重要：** Local MapとGlobal Mapの座標系が一致していることを確認すること。TFの設定やオドメトリの入力元を変更する場合は、地図生成処理とマッチング処理の両方への影響を確認する。

### 4.2 マルチレイヤ地図

点群を高さ方向に分類し、以下の3種類のレイヤを使用する。

| レイヤ | 高さ範囲 | 使用する情報 |
|---|---|---|
| Ground | -0.15 ～ 0.15 m | 反射強度 |
| Middle | 0.50 ～ 1.00 m | 距離情報 |
| High | 2.00 ～ 4.00 m | 距離情報 |

各レイヤをRGB画像として統合する。

- R：High
- G：Middle
- B：Ground

### 4.3 Global Mapの保存場所

Global Mapは以下のディレクトリに保存する。

保存先：`/ros2_ws/src/map/test2/`

地図生成ノード,Launchの`save_dir`などのパラメータを確認すること。

地図の保存先を変更する場合は、後述するマッチングノードの地図読み込み先も合わせて変更する。

### 4.4 PNGからJPEGへの変換

Global Mapの生成後、Waypoint情報を付加するためにPNG画像をJPEG画像へ変換する必要がある。

変換には`png_to_jpeg.py`を使用する。

処理の流れ：

1. Global MapをPNG形式で生成する。
2. `png_to_jpeg.py`を実行してJPEG形式に変換する。
3. JPEG画像を開き、必要なWaypoint位置に白色の領域を描画する。
4. 保存したJPEG画像をWaypoint情報を含む地図として使用する。

※PNG形式の地図を生成しただけでは完成しない。地図生成後に必ずJPEG変換を行うこと。

※png_to_jpeg.pyの中で変換する地図のディレクトリを指定する必要がある。

### 4.5 JPEG画像による仮想障害物・Waypoint管理

JPEG画像では、白色で塗られた領域を仮想障害物として利用できる。

WaypointごとにJPEG画像上の白色領域を設定することで、走行経路上に仮想障害物を配置できる。

JPEG画像を編集する場合は以下に注意する。

- 白色領域の位置と大きさを確認する。
- JPEG画像の解像度やサイズを変更しない。

## 5. マッチングアルゴリズム

### 5.1 基本処理

Global MapとLocal Mapを用いたテンプレートマッチングによって自己位置を推定する。

基本的な処理手順は以下のとおり。

1. Global Mapを読み込む。
2. LiDAR点群からLocal Mapを生成する。
3. オドメトリを初期位置として探索範囲を設定する。
4. Global Map上でLocal Mapに対応する領域を探索する。
5. 各レイヤのマッチングスコアを計算する。
6. PSRおよび占有率を用いてマッチング結果を評価する。
7. 最適な候補位置を選択する。
8. 推定位置を出力する。

### 5.2 テンプレートマッチング

各レイヤのGlobal MapとLocal Mapを照合し、画像間の類似度を計算する。

実装では、レイヤに応じた類似度指標を使用する。

- Ground：正規化相互相関（NCC）
- Middle：正規化相互相関（NCC）
- High：正規化相互相関（NCC）

※OpenCVの実際の比較手法（`TM_CCOEFF_NORMED`、`TM_CCORR_NORMED`など）については、実装に合わせて記載する。

### 5.3 PSRによる評価

PSR（Peak-to-Sidelobe Ratio）は、マッチングスコアの最大ピークと周辺領域のスコアの差を利用して、ピークの明瞭さを評価する指標である。

PSRが大きいほど、周辺候補と比較して明確なマッチング候補が存在することを示す。

### 5.4 占有率による評価

占有率は、マッチング対象となる領域にどの程度の特徴が存在するかを評価するために使用する。

特徴が少ない領域では、マッチングスコアが高くても信頼性が低い可能性がある。そのため、PSRだけでなく占有率も利用する。

### 5.5 重み付け

PSRと占有率を組み合わせ、各レイヤのマッチング結果に重みを付ける。

これにより、単純なマッチングスコアだけでは判断できない特徴の信頼性を考慮する。

※具体的な正規化方法、重みの算出式、各レイヤの統合方法は実装に基づいて記載する。

### 5.6 候補位置の探索

オドメトリを基準に位置・姿勢の候補を探索する。

探索範囲、角度の刻み幅、候補数などはパラメータによって調整する。

候補位置の選択では、マッチングスコアだけでなく、オドメトリとの差や前回の推定位置との関係も考慮する。

## 6. Launchファイルとパラメータ

Launchファイルでは、各ノードの起動とパラメータの設定を行う。

使用するLaunchファイルごとに、以下の情報を記載する。

### 6.1 Launchファイル一覧

| Launchファイル | 起動する処理 | 用途 |
|---|---|---|
| `make_globalmap.launch.xml` | `reflection_intensity_globalmap_multi_sub.py` | `Global Map生成処理` |
| `map_matching.launch.xml` | `reflection_intensity_localmap_multi_sub.py, png_map_matching` | `Local Map生成,マッチング処理` |

### 6.2 Launchファイル内のパラメータ

各Launchファイルに設定されているパラメータについて、初期値と役割を説明する。

| パラメータ | 初期値 | 内容 |
|---|---:|---|
| `save_dir` | `~/ros2_ws/src/map/test2` | `Global Map生成ファイル` |
| `folder_path` | `~/ros2_ws/src/map/tsukuba_full_1003_3` | `Global Map読み込みファイル` |
| `waypoint_start_index` | `0` | `開始Waypoint番号` |
| `use_sim_time` | `false` | `simulation時間の使用` |

パラメータを変更する場合は、Launchファイルから渡される値と、ノード側のデフォルト値の両方を確認すること。

### 6.3 主要パラメータ一覧
**reflection_intensity_globalmap_multi_sub.py**
| パラメータ | 初期値 | 内容 |
|---|---:|---|
| `grid_pixel` | 1000/50 | MAP解像度 |
| `MAP_RANGE_GL` | 20 m | GlobalMapの距離閾値(20m×20m) |
| `map_pos_diff_1` | 5 m | Map保存閾値(閾値距離移動→保存) |
| `map_pos_diff_2` | 2 m |  Map保存閾値2(閾値距離移動 and 移動角度閾値→保存) |
| `map_theta_diff` | 40 ° | Map保存閾値2(閾値距離移動 and 移動角度閾値→保存) |
| `RAY_RANGE` | 50 m | Rayの距離 |
| `bin_count` | 720 | Rayの個数 |

**reflection_intensity_localmap_multi_sub.py**
| パラメータ | 初期値 | 内容 |
|---|---:|---|
| `grid_pixel` | 1000/50 | MAP解像度 |
| `MAP_RANGE` | 15 m | LocalMapの距離閾値(15m×15m) |
| `RAY_RANGE` | 50 m | Rayの距離 |
| `bin_count` | 720 | Rayの個数 |

**png_map_matching.py**
| パラメータ | 初期値 | 内容 |
|---|---:|---|
| `match_per_threshold` | 0.4 | マッチング結果の採用判定 |
| `match_dist` | 1.5 m | 候補位置の距離制限 |
| `odom_error_threshold` | 0.5 m | オドメトリとの差の判定 |
| `waypoint_range` | 4.5 m | Waypoint到達判定(基本) |
| `waypoint_range_list` | [23, 1.5] | Waypoint到達判定変更[Waypoint番号, 到達判定距離] |
| `waypoint_offset` | {0: [-1.0, 0.0]} | Waypoint地点変更(加算){Waypoint番号：[x(m), y(m)]} |
| `stop_xy` | [-12.0,  12.0,   0.0,   1.0, -88,  68] | 停止位置設定(Localマッチング)[xmin, xmax, ymin, ymax,目標角度, Waypoint番号] |

※上記は開発時に使用していた値であり、現在のLaunch設定における初期値と一致するか確認すること。

※waypoint_offsetはGlobal座標上のxyで考え、指定した分Waypointをずらす。

※stop_xyはMAP座標上のxyで考える。Global座標上のxyとは異なる。確認の仕方はcheck_xy.pyを動かして停止したいWaypoint番号のファイルを開く。

## 7. 他パッケージとの連携

### 7.1 必要なパッケージ

本システムはROS 2の標準機能に加えて、点群処理、TF変換、画像処理などに関係するパッケージを使用する。

実際に必要なパッケージは、`package.xml`およびソースコードのImport・依存関係を確認して一覧化する。

### 7.2 orange2026Aとの連携

停止処理は別リポジトリである`orange2026A`の`path_follower`と連携している。

`intensity_matching`側のWaypoint・停止位置の管理と、`path_follower`側の走行制御・停止処理を組み合わせて使用する。

そのため、停止位置やWaypointの設定を変更する場合は、両方のパッケージで設定が整合していることを確認する。

特に、停止地点の追加・削除や停止番号の変更を行う場合は、停止処理の二重実行やWaypointのスキップが発生しないよう注意すること。

### 7.3 他ファイル・パッケージの必要性

本リポジトリ単体で完結する処理と、外部パッケージに依存する処理を区別する。

| 外部ファイル・パッケージ | 必要性 | 用途 |
|---|---|---|
| `orange2026A` | 停止処理・走行制御で必要 | `path_follower`との連携 |
| ROS 2依存パッケージ | ソースコードの依存関係に従う | 点群・TF・画像処理など |

## 8. ビルド・実行方法

### 8.1 ビルド

```bash
cd ~/ros2_ws/src
git clone https://github.com/kugurofu/intensity_matching.git
cd ~/ros2_ws
colcon build
source install/setup.bash
```

### 8.2 実行

実際に使用するLaunchファイルを記載する。

**Global Map作成時**
・bagで作成することを想定
・use_sim_time:=true
```bash
ros2 bag play xxx --clock
ros2 launch intensity_matching make_globalmap.launch.xml
```

**シミュレーション上でのマッチング時**
・bagでシミュレーションすることを想定
・use_sim_time:=true
```bash
ros2 bag play xxx --clock
ros2 launch intensity_matching map_matching.launch.xml
```

**実環境上でのマッチング時**
・use_sim_time:=false
```bash
ros2 bag play xxx --clock
ros2 launch intensity_matching map_matching.launch.xml
```

### 8.3 実行前の確認事項

- LiDAR点群が正常に取得できていること
- 必要なTFが配信されていること
- オドメトリが正常に取得できていること
- Global Mapが正しい場所に保存されていること
- PNGからJPEGへの変換が完了していること
- Waypoint・仮想障害物の設定が正しいこと
- 必要な外部パッケージが起動していること

## 9. トラブルシューティング

### マッチングが成立しない場合

- Global MapとLocal Mapの座標系が一致しているか確認する。
- TFが正常に配信されているか確認する。
- 各レイヤの特徴が正常に生成されているか確認する。
- 探索範囲と候補位置の制限を確認する。
- マッチングスコア、PSR、占有率を確認する。

### 推定位置が不安定な場合

- オドメトリの誤差を確認する。
- 各レイヤのマッチング結果を個別に確認する。
- 動的障害物による特徴の変化を確認する。
- マッチング候補の距離制限を確認する。

### Waypoint・仮想障害物が正常に動作しない場合

- PNGからJPEGへの変換が行われているか確認する。
- JPEG画像の白色領域が正しく設定されているか確認する。
- JPEG画像とGlobal Mapの位置関係が一致しているか確認する。
- `orange2026A`の`path_follower`との設定が一致しているか確認する。

## 10. 引き継ぎ時の注意事項

本システムを変更する場合は、以下の点に注意すること。

1. Local点群はTFによってGlobal座標系へ変換して使用する。
2. 地図の座標はGlobalの`odom`座標系を基準としている。
3. Global Map生成後はPNGからJPEGへの変換が必要である。
4. JPEG画像は仮想障害物の管理用。
5. 停止処理は`orange2026A`の`path_follower`と連携している。
6. `pcd_filter`はテスト用であり、現在のシステムでは使用しない。
7. マッチング処理を変更した場合は、地図生成処理との整合性も確認すること。
