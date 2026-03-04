"""
人物検出 → 追跡 → ホモグラフィー変換 の全パイプラインを実行

ステップ:
  1. GCP地図選択     - ブラウザで地図を開きWGS84座標を取得
  2. GCP動画選択     - 動画フレームで対応する画像座標を取得
  3. 人物検出・追跡   - YOLOで人物検出し軌跡を出力
  4. ホモグラフィー変換 - 軌跡をWGS84座標に変換

License: MIT License
Author: Toki Hirose
"""

import os
import sys
import json
import webbrowser
from pathlib import Path

from config import (
    VIDEO_DIR,
    OUTPUT_DIR,
    TRAJECTORY_DIR,
    VIEWER_DIR,
    GCP_CONFIG_PATH,
    GCP_FRAME_SEC,
    MAP_CENTER_LAT,
    MAP_CENTER_LON,
    MAP_ZOOM,
)


def step2_select_gcp_map(gcp_config_path: str) -> bool:
    """Step 2: 地図でGCPのWGS84座標を選択"""
    print("\n" + "=" * 60)
    print("Step 2: 地図でGCPを選択（動画の点に対応する地図上の点）")
    print("=" * 60)

    # 既にWGS84座標がある場合はスキップ
    if os.path.exists(gcp_config_path):
        with open(gcp_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if len(config.get('gcp_wgs84', [])) >= 4:
            print(f"  GCP設定が既にあります ({len(config['gcp_wgs84'])}点)")
            ans = input("  再選択しますか？ (y/N): ").strip().lower()
            if ans != 'y':
                return True

    from gcp_selector_map import create_gcp_selector_map

    # GCPポイント付きフレーム画像があれば地図に表示
    frame_image_path = os.path.join(
        os.path.dirname(gcp_config_path), "gcp_frame.png"
    )
    if not os.path.exists(frame_image_path):
        frame_image_path = None

    html_path = create_gcp_selector_map(
        center_lat=MAP_CENTER_LAT,
        center_lon=MAP_CENTER_LON,
        zoom_start=MAP_ZOOM,
        frame_image_path=frame_image_path,
    )

    webbrowser.open('file://' + os.path.abspath(html_path))

    print("\n  ブラウザで地図が開きました")
    print("  1. 4点以上クリックしてGCPを選択")
    print("  2. 「JSONをエクスポート」ボタンでgcp_config.jsonを保存")
    print(f"  3. 保存先: {os.path.abspath(gcp_config_path)}")

    input("\n  完了したらEnterを押してください...")

    # 確認
    if not os.path.exists(gcp_config_path):
        print("  エラー: gcp_config.jsonが見つかりません")
        return False

    with open(gcp_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    n_points = len(config.get('gcp_wgs84', []))
    if n_points < 4:
        print(f"  エラー: GCPが{n_points}点しかありません（4点以上必要）")
        return False

    print(f"  OK: {n_points}点のGCPを確認")
    return True


def step1_select_gcp_video(video_path: str, gcp_config_path: str) -> bool:
    """Step 1: 動画フレームでGCPの画像座標を選択"""
    print("\n" + "=" * 60)
    print("Step 1: 動画フレームでGCPの画像座標を選択")
    print("=" * 60)

    # 既に画像座標がある場合はスキップ
    if os.path.exists(gcp_config_path):
        with open(gcp_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if len(config.get('gcp_image', [])) >= 4:
            print(f"  画像座標が既にあります ({len(config['gcp_image'])}点)")
            ans = input("  再選択しますか？ (y/N): ").strip().lower()
            if ans != 'y':
                return True

    from gcp_selector_video import GCPVideoSelector, update_gcp_config

    print(f"  動画: {video_path}")
    print(f"  フレーム: {GCP_FRAME_SEC}秒後")

    selector = GCPVideoSelector(video_path, GCP_FRAME_SEC)
    points = selector.select_points()

    if not points:
        print("  キャンセルされました")
        return False

    update_gcp_config(gcp_config_path, points, video_path)

    # GCPポイント付きフレーム画像を保存（Step 2の地図で参照用）
    frame_path = os.path.join(
        os.path.dirname(gcp_config_path), "gcp_frame.png"
    )
    os.makedirs(os.path.dirname(frame_path), exist_ok=True)
    selector.save_frame(frame_path)

    print(f"  OK: {len(points)}点の画像座標を保存")
    print(f"  → 次のステップで地図上の対応点を{len(points)}点選択してください")
    return True


def step3_detect_and_track(video_path: str, output_dir: str) -> str:
    """Step 3: 人物検出・追跡"""
    print("\n" + "=" * 60)
    print("Step 3: 人物検出・追跡")
    print("=" * 60)

    video_name = Path(video_path).stem
    analysis_json = os.path.join(output_dir, f"{video_name}_analysis.json")

    # 既に結果がある場合はスキップ
    if os.path.exists(analysis_json):
        print(f"  検出結果が既にあります: {analysis_json}")
        ans = input("  再実行しますか？ (y/N): ").strip().lower()
        if ans != 'y':
            return analysis_json

    from detects_people import load_yolox_model, track_video
    from config import YOLOX_ONNX

    script_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(script_dir, YOLOX_ONNX)
    net = load_yolox_model(onnx_path)
    if net is None:
        return None

    print(f"  動画: {video_path}")
    print(f"  出力先: {output_dir}")

    track_video(Path(video_path), net, output_dir)

    if not os.path.exists(analysis_json):
        print("  エラー: 検出結果が生成されませんでした")
        return None

    print(f"  OK: {analysis_json}")
    return analysis_json


def step4_homography_transform(
    analysis_json: str,
    gcp_config_path: str,
    output_dir: str
) -> str:
    """Step 4: ホモグラフィー変換"""
    print("\n" + "=" * 60)
    print("Step 4: ホモグラフィー変換")
    print("=" * 60)

    from project_v2wgs84 import HomographyTransformer, transform_tracking_json

    # GCP設定を読み込み
    with open(gcp_config_path, 'r', encoding='utf-8') as f:
        gcp_config = json.load(f)

    # ホモグラフィー行列を計算
    transformer = HomographyTransformer()
    transformer.compute_from_gcp(
        gcp_config['gcp_image'],
        gcp_config['gcp_wgs84']
    )

    # ホモグラフィー行列を保存
    h_path = os.path.join(output_dir, "homography.json")
    transformer.save(h_path)

    # 出力パス
    input_path = Path(analysis_json)
    output_json = str(input_path.parent / f"{input_path.stem}_wgs84.json")

    # 変換を適用
    transform_tracking_json(analysis_json, output_json, transformer)

    print(f"  OK: {output_json}")
    return output_json


def step5_generate_viewer(wgs84_json: str, viewer_dir: str) -> str:
    """Step 5: 軌跡ビューアHTMLを生成"""
    print("\n" + "=" * 60)
    print("Step 5: 軌跡ビューア生成")
    print("=" * 60)

    from generate_viewer import generate_html

    with open(wgs84_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    html = generate_html(data, Path(wgs84_json).name)
    if html is None:
        return None

    os.makedirs(viewer_dir, exist_ok=True)

    stem = Path(wgs84_json).stem
    out_path = os.path.join(viewer_dir, f"{stem}_viewer.html")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"  OK: {out_path}")
    return out_path


def run_all(video_path: str = None, output_dir: str = None, gcp_config_path: str = None):
    """全パイプラインを実行"""
    print("=" * 60)
    print("  人物追跡 → WGS84変換 パイプライン")
    print("=" * 60)

    # デフォルト値の設定
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if gcp_config_path is None:
        gcp_config_path = GCP_CONFIG_PATH

    trajectory_dir = TRAJECTORY_DIR
    viewer_dir = VIEWER_DIR

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(trajectory_dir, exist_ok=True)
    os.makedirs(viewer_dir, exist_ok=True)

    # 動画ファイルの選択
    if video_path is None:
        video_dir = Path(VIDEO_DIR)
        videos = sorted(video_dir.glob("*.mp4")) + sorted(video_dir.glob("*.avi"))

        if not videos:
            print(f"エラー: {VIDEO_DIR} に動画がありません")
            return

        print(f"\n動画ファイル一覧 ({video_dir}):")
        for i, v in enumerate(videos):
            print(f"  {i + 1}. {v.name}")

        idx = input(f"\n番号を選択 (1-{len(videos)}): ").strip()
        try:
            video_path = str(videos[int(idx) - 1])
        except (ValueError, IndexError):
            print("無効な選択です")
            return

    print(f"\n対象動画: {video_path}")

    # Step 1: 動画でGCP画像座標を選択（先に動画で特徴点を決める）
    if not step1_select_gcp_video(video_path, gcp_config_path):
        print("\nStep 1 でエラーが発生しました")
        return

    # Step 2: 地図でGCP WGS84座標を選択（動画の点に対応する地図上の点）
    if not step2_select_gcp_map(gcp_config_path):
        print("\nStep 2 でエラーが発生しました")
        return

    # Step 3: 人物検出・追跡
    analysis_json = step3_detect_and_track(video_path, trajectory_dir)
    if analysis_json is None:
        print("\nStep 3 でエラーが発生しました")
        return

    # Step 4: ホモグラフィー変換
    output_json = step4_homography_transform(
        analysis_json, gcp_config_path, trajectory_dir
    )
    if output_json is None:
        print("\nStep 4 でエラーが発生しました")
        return

    # Step 5: ビューア生成
    viewer_path = step5_generate_viewer(output_json, viewer_dir)

    # 完了
    print("\n" + "=" * 60)
    print("  全ステップ完了!")
    print("=" * 60)
    print(f"  入力動画:     {video_path}")
    print(f"  GCP設定:      {gcp_config_path}")
    print(f"  検出結果:     {analysis_json}")
    print(f"  WGS84変換結果: {output_json}")
    if viewer_path:
        print(f"  ビューア:     {viewer_path}")
    print("=" * 60)

    # ビューアをブラウザで開く
    if viewer_path:
        ans = input("\nビューアをブラウザで開きますか？ (Y/n): ").strip().lower()
        if ans != 'n':
            webbrowser.open(Path(viewer_path).absolute().as_uri())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='全パイプラインを実行')
    parser.add_argument('--video', '-v', type=str, default=None, help='動画ファイルパス')
    parser.add_argument('--output', '-o', type=str, default=None, help='出力ディレクトリ')
    parser.add_argument('--gcp', '-g', type=str, default=None, help='GCP設定ファイル')

    args = parser.parse_args()

    run_all(
        video_path=args.video,
        output_dir=args.output,
        gcp_config_path=args.gcp
    )
