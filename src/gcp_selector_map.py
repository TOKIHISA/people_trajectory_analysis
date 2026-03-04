"""
Foliumを使って地図上でGCP（Ground Control Points）を選択するツール

使い方:
1. このスクリプトを実行
2. ブラウザで生成されたHTMLを開く
3. 地図上で4点以上をクリック
4. 右側のパネルに表示される座標をコピー
5. gcp_config.jsonを編集して座標を追加

License: MIT License
Author: Toki Hirose
"""

import folium
from folium.plugins import MousePosition
import json
import os
import base64
import webbrowser
from pathlib import Path


def create_gcp_selector_map(
    center_lat: float = 35.6812,  # 東京駅付近
    center_lon: float = 139.7671,
    zoom_start: int = 18,
    output_path: str = None,
    frame_image_path: str = None
) -> str:
    """
    GCP選択用の地図HTMLを生成

    Args:
        center_lat: 地図の中心緯度
        center_lon: 地図の中心経度
        zoom_start: 初期ズームレベル
        output_path: 出力HTMLパス
        frame_image_path: GCPポイント付き動画フレーム画像パス

    Returns:
        str: 生成されたHTMLファイルのパス
    """
    # 地図を作成（ベースレイヤーなしで開始）
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles=None
    )

    # OpenStreetMapレイヤー
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='OpenStreetMap',
        control=True
    ).add_to(m)

    # 航空写真レイヤー（Esri World Imagery）
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='航空写真 (Esri)',
        control=True
    ).add_to(m)

    # 国土地理院 航空写真
    folium.TileLayer(
        tiles='https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg',
        attr='国土地理院',
        name='航空写真 (国土地理院)',
        control=True
    ).add_to(m)

    # 国土地理院 標準地図
    folium.TileLayer(
        tiles='https://cyberjapandata.gsi.go.jp/xyz/std/{z}/{x}/{y}.png',
        attr='国土地理院',
        name='標準地図 (国土地理院)',
        control=True
    ).add_to(m)

    # レイヤーコントロールを追加
    folium.LayerControl(position='topleft').add_to(m)

    # マウス位置表示
    MousePosition(
        position='topright',
        separator=' | ',
        prefix='座標:',
        lat_formatter="function(lat) {return lat.toFixed(8);}",
        lng_formatter="function(lng) {return lng.toFixed(8);}"
    ).add_to(m)

    # フレーム画像をbase64に変換
    frame_img_b64 = ""
    if frame_image_path and os.path.exists(frame_image_path):
        with open(frame_image_path, "rb") as f:
            frame_img_b64 = base64.b64encode(f.read()).decode("utf-8")

    # クリックで座標を取得するJavaScript
    click_js = """
    <script>
    var gcpPoints = [];
    var markers = [];

    function updateGCPList() {
        var listHtml = '<h4>GCP Points (WGS84)</h4>';
        listHtml += '<p>クリックで点を追加（4点以上必要）</p>';
        listHtml += '<table style="width:100%; font-size:12px;">';
        listHtml += '<tr><th>#</th><th>Lat</th><th>Lon</th><th></th></tr>';

        for (var i = 0; i < gcpPoints.length; i++) {
            listHtml += '<tr>';
            listHtml += '<td>' + (i+1) + '</td>';
            listHtml += '<td>' + gcpPoints[i].lat.toFixed(8) + '</td>';
            listHtml += '<td>' + gcpPoints[i].lon.toFixed(8) + '</td>';
            listHtml += '<td><button onclick="removePoint(' + i + ')">×</button></td>';
            listHtml += '</tr>';
        }
        listHtml += '</table>';

        if (gcpPoints.length >= 4) {
            listHtml += '<br><button onclick="exportGCP()" style="width:100%; padding:10px; background:#4CAF50; color:white; border:none; cursor:pointer;">JSONをエクスポート</button>';
        }

        listHtml += '<br><br><h4>JSON出力:</h4>';
        listHtml += '<textarea id="jsonOutput" style="width:100%; height:150px; font-size:10px;">' + JSON.stringify(gcpPoints, null, 2) + '</textarea>';

        document.getElementById('gcpList').innerHTML = listHtml;
    }

    function removePoint(index) {
        gcpPoints.splice(index, 1);
        map.removeLayer(markers[index]);
        markers.splice(index, 1);

        // マーカーの番号を更新
        for (var i = 0; i < markers.length; i++) {
            markers[i].setIcon(L.divIcon({
                className: 'gcp-marker',
                html: '<div style="background:#e74c3c; color:white; width:24px; height:24px; border-radius:50%; text-align:center; line-height:24px; font-weight:bold;">' + (i+1) + '</div>',
                iconSize: [24, 24],
                iconAnchor: [12, 12]
            }));
        }
        updateGCPList();
    }

    function exportGCP() {
        var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify({
            "gcp_wgs84": gcpPoints,
            "gcp_image": [],
            "note": "gcp_imageには動画フレームから取得した対応点を追加してください"
        }, null, 2));
        var dlAnchorElem = document.createElement('a');
        dlAnchorElem.setAttribute("href", dataStr);
        dlAnchorElem.setAttribute("download", "gcp_config.json");
        dlAnchorElem.click();
    }

    // 地図クリックイベント
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            var mapElement = document.querySelector('.folium-map');
            if (mapElement && mapElement._leaflet_id) {
                var map = window['map_' + mapElement.id.split('_')[1]] ||
                          Object.values(window).find(v => v instanceof L.Map);

                if (map) {
                    window.map = map;
                    map.on('click', function(e) {
                        var point = {
                            lat: e.latlng.lat,
                            lon: e.latlng.lng
                        };
                        gcpPoints.push(point);

                        var marker = L.marker([e.latlng.lat, e.latlng.lng], {
                            icon: L.divIcon({
                                className: 'gcp-marker',
                                html: '<div style="background:#e74c3c; color:white; width:24px; height:24px; border-radius:50%; text-align:center; line-height:24px; font-weight:bold;">' + gcpPoints.length + '</div>',
                                iconSize: [24, 24],
                                iconAnchor: [12, 12]
                            })
                        }).addTo(map);
                        markers.push(marker);

                        updateGCPList();
                    });
                }
            }
        }, 1000);

        updateGCPList();
    });
    </script>

    <div id="gcpList" style="position:fixed; top:10px; right:10px; width:280px; max-height:90vh; overflow-y:auto; background:white; padding:15px; border-radius:8px; box-shadow:0 2px 10px rgba(0,0,0,0.2); z-index:1000;">
        <h4>GCP Points</h4>
        <p>読み込み中...</p>
    </div>
    """

    # フレーム画像パネル
    if frame_img_b64:
        click_js += """
    <style>
    #framePanel {
        position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
        z-index: 1000; background: rgba(0,0,0,0.85); border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.4); cursor: move; user-select: none;
    }
    #framePanel img {
        display: block; max-width: 480px; max-height: 300px;
        border-radius: 0 0 8px 8px;
    }
    #frameHeader {
        color: #fff; font: bold 12px sans-serif; padding: 6px 10px;
        display: flex; justify-content: space-between; align-items: center;
    }
    #frameHeader button {
        background: none; border: none; color: #fff; font-size: 16px; cursor: pointer;
    }
    #framePanel.collapsed img { display: none; }
    </style>
    <div id="framePanel">
        <div id="frameHeader">
            <span>Video GCP Points</span>
            <button onclick="var p=document.getElementById('framePanel'); p.classList.toggle('collapsed'); this.textContent = p.classList.contains('collapsed') ? '+' : '-';">-</button>
        </div>
        <img src="data:image/png;base64,""" + frame_img_b64 + """" draggable="false">
    </div>
    <script>
    // ドラッグ移動
    (function() {
        var panel = document.getElementById('framePanel');
        var header = document.getElementById('frameHeader');
        var dx = 0, dy = 0, startX = 0, startY = 0;
        header.addEventListener('mousedown', function(e) {
            startX = e.clientX; startY = e.clientY;
            document.addEventListener('mousemove', drag);
            document.addEventListener('mouseup', function() {
                document.removeEventListener('mousemove', drag);
            }, {once: true});
        });
        function drag(e) {
            dx = e.clientX - startX; dy = e.clientY - startY;
            startX = e.clientX; startY = e.clientY;
            panel.style.top = (panel.offsetTop + dy) + 'px';
            panel.style.left = (panel.offsetLeft + dx) + 'px';
            panel.style.transform = 'none';
        }
    })();
    </script>
    """

    # HTMLに追加
    m.get_root().html.add_child(folium.Element(click_js))

    # 出力パス
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__),
            'gcp_selector_map.html'
        )

    # 保存
    m.save(output_path)
    print(f"地図を保存しました: {output_path}")

    return output_path


def load_gcp_config(config_path: str) -> dict:
    """GCP設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_gcp_config(config: dict, config_path: str):
    """GCP設定ファイルを保存"""
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"GCP設定を保存しました: {config_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GCP選択用地図を生成')
    parser.add_argument('--lat', type=float, default=35.6812, help='中心緯度')
    parser.add_argument('--lon', type=float, default=139.7671, help='中心経度')
    parser.add_argument('--zoom', type=int, default=18, help='ズームレベル')
    parser.add_argument('--output', type=str, default=None, help='出力HTMLパス')
    parser.add_argument('--open', action='store_true', help='ブラウザで開く')

    args = parser.parse_args()

    html_path = create_gcp_selector_map(
        center_lat=args.lat,
        center_lon=args.lon,
        zoom_start=args.zoom,
        output_path=args.output
    )

    if args.open:
        webbrowser.open('file://' + os.path.abspath(html_path))
