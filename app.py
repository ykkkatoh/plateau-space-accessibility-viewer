"""
CityGML Viewer — Streamlit app
依存: streamlit>=1.35, geopandas, pydeck, folium, streamlit-folium, shapely, pandas, numpy, pyproj
"""
from __future__ import annotations
from pathlib import Path
import re

import numpy as np
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import streamlit as st
from shapely.geometry import box, MultiPolygon
from streamlit_folium import st_folium
from pyproj import Transformer
import folium
from folium.plugins import Draw
from shapely.ops import unary_union

# ── 定数 ──────────────────────────────────────────────────────────────────────
EPSG_WGS84 = 4326
EPSG_JGD   = 6677

DATA_DIR  = Path(__file__).parent / "data"
BLDG_DIR  = DATA_DIR / "bldg"
LUSE_DIR  = DATA_DIR / "luse"
NAME_MAP_CSV        = DATA_DIR / "landuse_name_map.csv"
CATEGORY_ROOTS_CSV  = DATA_DIR / "landuse_category_roots.csv"
CATEGORY_GROUPS_CSV = DATA_DIR / "landuse_category_groups.csv"

VST_CATEGORIES = {"自然地", "商業用地", "公共用地", "交通用地"}
LCL_CATEGORIES = {"自然地", "農林地", "住居用地", "商業用地", "農工業用地", "公共用地", "交通用地"}

CATEGORY_COLORS: dict[str, list[int]] = {
    # ── グループ1: 観光客アクセス可（暖色）──
    "商業用地":   [205, 90, 75],   # 赤
    "公共用地":   [230, 140,  50],   # オレンジ
    "交通用地":   [190, 170, 90],   # 黄
    "自然地":     [100, 190,  90],   # 黄緑（自然らしさを残す）

    # ── グループ2: 地域関係者のみ（寒色）──
    "住居用地":   [95, 145, 210],   # 青
    "農工業用地": [ 80, 160, 170],   # シアン寄り
    "農林地":     [ 60, 150, 120],   # 緑

    # ── グループ3: アクセス不可（モノクロ）──
    "その他":     [120, 120, 120],   # 中グレー
    "水面":       [160, 160, 180],   # グレー（やや青みがかり）
    "未細分":     [200, 200, 200],   # 薄グレー
}

HIGHLIGHT_COLOR = [255, 220, 50]
WIREFRAME_COLOR = [255, 220, 50, 230]

MAP_CENTER = (35.6812, 139.7671)
DEFAULT_LABELS = ["エリア A", "エリア B", "エリア C"]

COLORMAP_PRESETS = {
    "Cividis": [(0.0,(0,32,76)),(0.25,(42,60,102)),(0.5,(114,100,93)),
                (0.75,(178,151,88)),(1.0,(253,231,37))],
    "Viridis": [(0.0, (68,1,84)), (0.25, (59,82,139)), (0.5, (33,145,140)),
                (0.75, (94,201,98)), (1.0, (253,231,37))],
    "Plasma":  [(0.0, (13,8,135)), (0.25, (126,3,168)), (0.5, (204,71,120)),
                (0.75, (248,149,64)), (1.0, (240,249,33))],
}


def normalize_bb_size(bb_wgs: tuple, ref_bb: tuple) -> tuple:
    ref_w = ref_bb[2] - ref_bb[0]
    ref_h = ref_bb[3] - ref_bb[1]
    cx = (bb_wgs[0] + bb_wgs[2]) / 2
    cy = (bb_wgs[1] + bb_wgs[3]) / 2
    return (cx - ref_w/2, cy - ref_h/2, cx + ref_w/2, cy + ref_h/2)


# ── メッシュコード → BBox ──────────────────────────────────────────────────────
def mesh_to_bbox(code: str) -> tuple[float, float, float, float] | None:
    if not code.isdigit() or len(code) not in (8, 9):
        return None
    p, q = int(code[0:2]), int(code[2:4])
    r, s = int(code[4]), int(code[5])
    t, u = int(code[6]), int(code[7])
    lat0 = p / 1.5 + r * 5.0/60 + t * 30.0/3600
    lon0 = q + 100.0 + s * 7.5/60 + u * 45.0/3600
    d_lat, d_lon = 30.0/3600, 45.0/3600
    if len(code) == 9:
        v = int(code[8])
        if v not in (1, 2, 3, 4):
            return None
        lat0 += ((v-1)//2) * (d_lat/2)
        lon0 += ((v-1) %2) * (d_lon/2)
        d_lat /= 2; d_lon /= 2
    return (lon0, lat0, lon0+d_lon, lat0+d_lat)


def find_overlapping_files(bb_wgs: tuple) -> list[Path]:
    bb_box = box(*bb_wgs)
    result = []
    for f in sorted(BLDG_DIR.glob("*.geojson")):
        s = f.stem
        code = s[:9] if len(s) >= 9 and s[:9].isdigit() else s[:8]
        bbox = mesh_to_bbox(code)
        if bbox and box(*bbox).intersects(bb_box):
            result.append(f)
    return result


# ── データ読み込み ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="建物データを読み込み中…")
def load_buildings_for_bb(file_paths: tuple[str, ...]) -> gpd.GeoDataFrame:
    parts = []
    for p in file_paths:
        gdf = gpd.read_file(p)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=EPSG_WGS84)
        parts.append(gdf.to_crs(epsg=EPSG_JGD))
    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{EPSG_JGD}")
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=f"EPSG:{EPSG_JGD}")


@st.cache_data(show_spinner="土地利用データを読み込み中…")
def load_landuse() -> gpd.GeoDataFrame:
    files = list(LUSE_DIR.glob("*.gpkg"))
    if not files:
        st.error(f"土地利用GPKGが見つかりません: {LUSE_DIR}"); st.stop()
    parts = []
    for f in files:
        gdf = gpd.read_file(f)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=EPSG_WGS84)
        parts.append(gdf.to_crs(epsg=EPSG_JGD))
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=f"EPSG:{EPSG_JGD}")


@st.cache_data
def load_mappings() -> tuple[dict, dict]:
    nm = pd.read_csv(NAME_MAP_CSV, encoding="utf-8-sig", dtype=str).dropna()
    name_map = dict(zip(nm["raw"].str.strip(), nm["short"].str.strip()))
    cr = pd.read_csv(CATEGORY_ROOTS_CSV, encoding="utf-8-sig", dtype=str).dropna()
    root_map = dict(zip(cr["root"].str.strip(), cr["category"].str.strip()))
    return name_map, root_map

@st.cache_data
def get_available_bboxes() -> list[tuple]:
    """dataフォルダにあるGeoJSONファイルのBBox一覧を返す"""
    result = []
    for f in sorted(BLDG_DIR.glob("*.geojson")):
        s = f.stem
        code = s[:9] if len(s) >= 9 and s[:9].isdigit() else s[:8]
        bbox = mesh_to_bbox(code)
        if bbox:
            result.append(bbox)
    return result

@st.cache_data
def get_coverage_shape():
    """全データBBoxをunionした外形ポリゴンを返す"""
    boxes = [box(*bb) for bb in get_available_bboxes()]
    if not boxes:
        return None
    return unary_union(boxes)

def _interpolate_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def alpha_to_hex(value: float, vmin: float, vmax: float, preset: str) -> str:
    if np.isnan(value) or np.isinf(value):
        return "#cccccc"
    norm = max(0.0, min(1.0, (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5))
    pts  = COLORMAP_PRESETS.get(preset, COLORMAP_PRESETS["Cividis"])
    for i in range(len(pts) - 1):
        p1, c1 = pts[i]
        p2, c2 = pts[i + 1]
        if p1 <= norm <= p2:
            r, g, b = _interpolate_color(c1, c2, (norm - p1) / (p2 - p1))
            return f"#{r:02x}{g:02x}{b:02x}"
    r, g, b = pts[-1][1]
    return f"#{r:02x}{g:02x}{b:02x}"


def render_alpha_heatmap(result_df: pd.DataFrame, bb_list: list, colormap: str = "Cividis"):
    df = result_df.copy()
    df["α_val"] = pd.to_numeric(df["α"], errors="coerce")
    valid = df["α_val"].dropna()
    if valid.empty:
        st.info("有効なα値がありません。")
        return

    vmin, vmax = float(valid.min()), float(valid.max())
    if vmax - vmin < 1e-6:
        vmin -= 0.1; vmax += 0.1

    center_lat = float(df["緯度"].mean())
    center_lon = float(df["経度"].mean())

    m = folium.Map(location=[center_lat, center_lon],
                   zoom_start=14, tiles="CartoDB positron")

    id_to_bb = {e["id"]: e["bb_wgs"] for e in bb_list}

    for _, row in df.iterrows():
        bb = id_to_bb.get(row["id"])
        if bb is None:
            continue
        alpha_val = row["α_val"]
        if np.isnan(alpha_val):
            fill_color, fill_opacity, line_color = "#cccccc", 0.3, "#999999"
            tooltip_val = "計算不可"
        else:
            fill_color   = alpha_to_hex(alpha_val, vmin, vmax, colormap)
            fill_opacity = 0.75
            line_color   = "#ffffff"
            tooltip_val  = f"{alpha_val:.3f}"

        folium.Rectangle(
            bounds=[[bb[1], bb[0]], [bb[3], bb[2]]],
            color=line_color, weight=1,
            fill=True, fill_color=fill_color, fill_opacity=fill_opacity,
            tooltip=folium.Tooltip(
                f"<b>{row['id']}</b><br>α = {tooltip_val}"
                f"<br><small>{'地域色が強い' if not np.isnan(alpha_val) and alpha_val >= (vmin+vmax)/2 else '観光色が強い'}</small>",
                style="font-size:12px;"
            ),
        ).add_to(m)

    pts  = COLORMAP_PRESETS.get(colormap, COLORMAP_PRESETS["Cividis"])
    grad = ", ".join([f"rgb({r},{g},{b}) {p*100:.0f}%" for p, (r,g,b) in pts])
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;bottom:20px;right:10px;width:240px;
                background:white;border:2px solid #aaa;border-radius:5px;
                padding:10px;font-size:12px;z-index:9999;
                box-shadow:2px 2px 6px rgba(0,0,0,0.25);">
        <div style="margin-bottom:6px;font-weight:bold;">補正係数 α</div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:11px;">{vmin:.2f}<br>観光色</span>
            <div style="flex:1;height:14px;
                        background:linear-gradient(to right,{grad});
                        margin:0 8px;border:1px solid #ccc;"></div>
            <span style="font-size:11px;">{vmax:.2f}<br>地域色</span>
        </div>
        <div style="margin-top:5px;font-size:10px;color:#888;">
            高α = 地域利用が優勢 → 観光流入時の負荷大
        </div>
    </div>"""))

    st_folium(m, width="100%", height=460, returned_objects=[], key="alpha_heatmap")

# ── ユーティリティ ─────────────────────────────────────────────────────────────
def to_category(class_val: str, name_map: dict, root_map: dict) -> str:
    short = name_map.get(str(class_val).strip(), str(class_val).strip())
    return root_map.get(short, "その他")


def geom_to_coords(g):
    if isinstance(g, MultiPolygon):
        g = max(g.geoms, key=lambda p: p.area)
    return [[c[0], c[1]] for c in g.exterior.coords]


def suggest_label_from_bldg(bldg_in: gpd.GeoDataFrame) -> str:
    if "address" not in bldg_in.columns:
        return ""
    def extract(addr: str) -> str:
        s = str(addr).strip()
        s = re.sub(r'.+?[都道府県]', '', s)
        s = s.replace(" ", "")
        m = re.search(r'([^\s]+?[区市町村])(.*)', s)
        if m:
            ku   = m.group(1)
            rest = m.group(2)
            town = re.split(r'[0-9０-９一二三四五六七八九十]', rest)[0]
            return f"{ku} {town}".strip() if town else ku
        return s[:10]

    series = bldg_in["address"].dropna().apply(extract)
    if series.empty:
        return ""
    ku_series = series.apply(lambda x: x.split()[0] if x.split() else x)
    top_ku = ku_series.value_counts().idxmax()
    town_series = series[ku_series == top_ku].apply(
        lambda x: x.split()[1] if len(x.split()) > 1 else "")
    top_town = town_series.value_counts().idxmax() if not town_series.empty else ""
    return f"{top_ku} {top_town}".strip()


# ── 補正係数計算 ───────────────────────────────────────────────────────────────
def calc_correction_factor(bldg, luse, bb_jgd, name_map, root_map):
    if luse.empty:
        return float("nan"), pd.DataFrame()

    lu = luse.copy()
    lu["_category"] = lu["class"].fillna("Unknown").apply(
        lambda x: to_category(x, name_map, root_map))

    bb_gdf = gpd.GeoDataFrame(geometry=[bb_jgd], crs=luse.crs)
    lu_clipped = gpd.overlay(lu, bb_gdf, how="intersection")
    lu_clipped["_luse_area"] = lu_clipped.geometry.area
    luse_by_cat = lu_clipped.groupby("_category")["_luse_area"].sum()

    b = bldg.copy()
    s = pd.to_numeric(b.get("storeysAboveGround"), errors="coerce")
    h = pd.to_numeric(b.get("measuredHeight"),     errors="coerce")
    b["_storeys"] = np.where(s.notna() & (s>0) & (s<=50), s,
                    np.where(h.notna() & (h>0), np.ceil(h/3.0), 3.0))
    b["_fp_area"] = b.geometry.area

    lu_sindex = lu.sindex
    extra_by_cat: dict[str, float] = {}
    for _, row in b.iterrows():
        pt = row.geometry.representative_point()
        cands = list(lu_sindex.intersection(pt.bounds))
        cat = "その他"
        for ci in cands:
            if lu.geometry.iloc[ci].contains(pt):
                cat = lu["_category"].iloc[ci]; break
        else:
            if cands:
                cat = lu["_category"].iloc[
                    lu.geometry.iloc[cands].distance(pt).values.argmin()]
        extra = float(row["_fp_area"]) * max(float(row["_storeys"])-1.0, 0.0)
        extra_by_cat[cat] = extra_by_cat.get(cat, 0.0) + extra

    all_cats = set(luse_by_cat.index) | set(extra_by_cat.keys())
    rows = [{"category": c,
             "floor_area_m2": luse_by_cat.get(c, 0.0) + extra_by_cat.get(c, 0.0)}
            for c in sorted(all_cats)]
    summary = pd.DataFrame(rows)
    summary["floor_area_km2"] = summary["floor_area_m2"] / 1e6

    s_vst = summary.loc[summary["category"].isin(VST_CATEGORIES), "floor_area_m2"].sum()
    s_lcl = summary.loc[summary["category"].isin(LCL_CATEGORIES), "floor_area_m2"].sum()
    alpha = s_lcl / s_vst if s_vst > 0 else float("nan")
    return alpha, summary


# ── カテゴリ割り当て ───────────────────────────────────────────────────────────
def assign_category_for_display(bldg_jgd, luse_jgd, name_map, root_map):
    if luse_jgd.empty:
        return ["その他"] * len(bldg_jgd)
    lu = luse_jgd.copy()
    lu["_cat"] = lu["class"].fillna("Unknown").apply(
        lambda x: to_category(x, name_map, root_map))
    sindex = lu.sindex
    cats = []
    for geom in bldg_jgd.geometry:
        pt = geom.representative_point()
        cands = list(sindex.intersection(pt.bounds))
        cat = "その他"
        for ci in cands:
            if lu.geometry.iloc[ci].contains(pt):
                cat = lu["_cat"].iloc[ci]; break
        else:
            if cands:
                cat = lu["_cat"].iloc[
                    lu.geometry.iloc[cands].distance(pt).values.argmin()]
        cats.append(cat)
    return cats


# ── ワイヤーフレームレイヤー生成 ───────────────────────────────────────────────
def build_wireframe_layers(bldg_row: dict, storeys: int) -> list[pdk.Layer]:
    coords_2d = bldg_row["coordinates"]
    total_h   = float(bldg_row["height"])
    floor_h   = total_h / max(storeys, 1)
    slab_h    = max(floor_h * 0.25, 1.0)

    records = []
    for i in range(storeys):
        z_bot = floor_h * i
        ring = [[c[0], c[1], z_bot] for c in coords_2d]
        if ring[0][:2] != ring[-1][:2]:
            ring.append(ring[0])
        records.append({
            "polygon": ring,
            "elevation": slab_h,
        })

    return [pdk.Layer(
        "PolygonLayer",
        data=records,
        id="floor-wireframe-layer",
        get_polygon="polygon",
        get_elevation="elevation",
        extruded=True,
        wireframe=True,
        stroked=True,
        filled=True,
        get_fill_color=[245, 245, 242, 180],
        get_line_color=[110, 110, 110, 220],
        line_width_min_pixels=1,
        pickable=False,
        opacity=0.8,
        **{"positionFormat": "XYZ"},
    )]


# ── pydeckレイヤー生成 ────────────────────────────────────────────────────────
def build_pydeck_layers(bldg_jgd, luse_jgd, name_map, root_map, sk_sel: str,
                        use_luse_color: bool = True):
    cats = assign_category_for_display(bldg_jgd, luse_jgd, name_map, root_map)
    b = bldg_jgd.copy().to_crs(epsg=EPSG_WGS84)
    b["category"] = cats

    h_raw = pd.to_numeric(bldg_jgd.get("measuredHeight"),     errors="coerce")
    s_raw = pd.to_numeric(bldg_jgd.get("storeysAboveGround"), errors="coerce")
    height_vals = np.where(h_raw.notna()&(h_raw>=1.5)&(h_raw<=200), h_raw,
                  np.where(s_raw.notna()&(s_raw>0)&(s_raw<=50), s_raw*3.0, 9.0))
    b["height"]  = np.nan_to_num(height_vals, nan=9.0, posinf=9.0, neginf=9.0)
    b["storeys"] = np.where(s_raw.notna()&(s_raw>0)&(s_raw<=50), s_raw,
                   np.ceil(b["height"] / 3.0)).astype(int)
    b["fp_area"]     = bldg_jgd.geometry.area.values
    b["coordinates"] = b.geometry.apply(geom_to_coords)
    b["hex_color"] = [
        "#{:02x}{:02x}{:02x}".format(*(
            CATEGORY_COLORS.get(c, [160,160,160]) if use_luse_color else [160,160,160]
        ))
        for c in cats
    ]
    for col in ["buildingId", "usage", "measuredHeight", "storeysAboveGround"]:
        if col in bldg_jgd.columns:
            b[col] = bldg_jgd[col].values

    def _coords_key(coords):
        return tuple(coords[0]) if coords else ()

    sel_key = st.session_state.get(sk_sel)

    GRAY = [160, 160, 160]
    b["color"] = [
        HIGHLIGHT_COLOR if (_coords_key(row["coordinates"]) == sel_key)
        else (CATEGORY_COLORS.get(row["category"], [160,160,160])
              if use_luse_color else GRAY)
        for _, row in b[["coordinates", "category"]].iterrows()
    ]
    b["hex_color"] = [
        "#{:02x}{:02x}{:02x}".format(*(
            CATEGORY_COLORS.get(c, GRAY) if use_luse_color else GRAY
        ))
        for c in cats
    ]

    keep = ["coordinates", "height", "color", "hex_color", "category",
            "storeys", "fp_area"] + \
           [c for c in ["buildingId", "usage", "measuredHeight", "storeysAboveGround"]
            if c in b.columns]
    records = b[keep].to_dict("records")

    def _coords_key(coords):
        return tuple(coords[0]) if coords else ()

    sel_key = st.session_state.get(sk_sel)

    base_records = [
        r for r in records
        if _coords_key(r.get("coordinates", [])) != sel_key
    ]

    base_layer = pdk.Layer(
        "PolygonLayer",
        data=base_records,
        id="building-layer",
        get_polygon="coordinates",
        get_elevation="height",
        get_fill_color="color",
        elevation_scale=1,
        extruded=True,
        pickable=True,

        # 見やすさ改善
        filled=True,
        stroked=True,
        wireframe=True,
        get_line_color=[155,155,155,140],
        line_width_min_pixels=1,

        # ベタ塗り感を弱める
        opacity=0.8,
    )

    layers = [base_layer]

    if sel_key:
        sel_rows = b[[_coords_key(c) == sel_key for c in b["coordinates"]]]
        if not sel_rows.empty:
            sel_row = sel_rows.iloc[0]
            layers += build_wireframe_layers(sel_row.to_dict(), int(sel_row["storeys"]))

    return layers, records


# ── 土地利用2Dマップ描画 ──────────────────────────────────────────────────────
def render_2d_map(bb_wgs: tuple, bldg_in: gpd.GeoDataFrame,
                  luse_in: gpd.GeoDataFrame, name_map: dict, root_map: dict,
                  area_key: str, show_luse: bool):
    """2Dマップ。show_luse=Trueなら土地利用ポリゴン、常に建物フットプリントをオーバーレイ"""
    center_lat = (bb_wgs[1] + bb_wgs[3]) / 2
    center_lon = (bb_wgs[0] + bb_wgs[2]) / 2
    m = folium.Map(location=[center_lat, center_lon],
                   zoom_start=16, tiles="CartoDB positron")

    # BB外枠
    folium.Rectangle(
        bounds=[[bb_wgs[1], bb_wgs[0]], [bb_wgs[3], bb_wgs[2]]],
        color="#333333", weight=2, fill=False,
    ).add_to(m)

    # 土地利用ポリゴン（show_luse=Trueのみ）
    if show_luse and not luse_in.empty:
        lu = luse_in.copy().to_crs(epsg=EPSG_WGS84)
        lu["_category"] = lu["class"].fillna("Unknown").apply(
            lambda x: to_category(x, name_map, root_map))
        DRAW_ORDER = ["交通用地", "水面", "自然地", "農林地", "農工業用地",
                      "公共用地", "住居用地", "商業用地", "未細分", "その他"]
        lu["_draw_order"] = lu["_category"].apply(
            lambda c: DRAW_ORDER.index(c) if c in DRAW_ORDER else len(DRAW_ORDER))
        lu = lu.sort_values("_draw_order")
        for _, row in lu.iterrows():
            cat = row["_category"]
            rgb = CATEGORY_COLORS.get(cat, [160, 160, 160])
            hex_c = "#{:02x}{:02x}{:02x}".format(*rgb)
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
            for poly in polys:
                coords_ll = [[c[1], c[0]] for c in poly.exterior.coords]
                folium.Polygon(
                    locations=coords_ll,
                    color="#ffffff", weight=0.5,
                    fill=True, fill_color=hex_c, fill_opacity=0.7,
                    tooltip=folium.Tooltip(f"<b>{cat}</b>", style="font-size:12px;"),
                ).add_to(m)
    elif show_luse:
        st.info("この範囲に土地利用データがありません")

    # 建物フットプリント
    if not bldg_in.empty:
        bldg_wgs = bldg_in.copy().to_crs(epsg=EPSG_WGS84)
        if show_luse:
            # 2D土地利用ONのとき：カテゴリ色で不透明に重ねる
            cats = assign_category_for_display(bldg_in, luse_in, name_map, root_map)
        for geom, cat in zip(bldg_wgs.geometry,
                             cats if show_luse else [None] * len(bldg_wgs)):
            if geom is None or geom.is_empty:
                continue
            if show_luse:
                rgb = CATEGORY_COLORS.get(cat, [160, 160, 160])
                fill_color   = "#{:02x}{:02x}{:02x}".format(*rgb)
                fill_opacity = 1.0
                line_color   = "#ffffff"
                line_weight  = 0.5
            else:
                fill_color   = "#aaaaaa"
                fill_opacity = 0.5
                line_color   = "#555555"
                line_weight  = 0.8
            polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
            for poly in polys:
                coords_ll = [[c[1], c[0]] for c in poly.exterior.coords]
                folium.Polygon(
                    locations=coords_ll,
                    color=line_color, weight=line_weight,
                    fill=True, fill_color=fill_color, fill_opacity=fill_opacity,
                ).add_to(m)

    st_folium(m, width="100%", height=460,
              returned_objects=[], key=f"2d_map_{area_key}")


# ── 凡例（コンパクト版）────────────────────────────────────────────────────────
def render_legend_vertical():
    """カテゴリ凡例を縦リストで表示する"""
    items = "".join([
        f'<div style="display:flex;align-items:center;gap:5px;margin-bottom:5px;">'
        f'<span style="width:11px;height:11px;border-radius:2px;flex-shrink:0;'
        f'background:#{"{:02x}{:02x}{:02x}".format(*col)};display:inline-block;"></span>'
        f'<span style="font-size:0.75rem;white-space:nowrap;">{cat}</span>'
        f'</div>'
        for cat, col in CATEGORY_COLORS.items()
    ])
    st.markdown(
        f'<div style="padding:6px 4px;">{items}</div>',
        unsafe_allow_html=True,
    )


# ── エリア1つ分のビュー描画 ───────────────────────────────────────────────────
def render_area(bb_wgs, label, gdf_luse, name_map, root_map,
                area_key: str, col_idx: int,
                is_3d: bool = True, show_luse: bool = False):
    sk_sel  = f"sel_bldg_{area_key}"
    sk_data = f"bldg_data_{area_key}"

    target_files = find_overlapping_files(bb_wgs)
    if not target_files:
        st.warning(f"{label}: 対応ファイルなし"); return

    gdf_bldg = load_buildings_for_bb(tuple(str(f) for f in target_files))
    bb_jgd   = (gpd.GeoDataFrame(geometry=[box(*bb_wgs)], crs=EPSG_WGS84)
                .to_crs(epsg=EPSG_JGD).geometry[0])
    bldg_in  = gdf_bldg[gdf_bldg.intersects(bb_jgd)].copy()
    luse_in  = gdf_luse[gdf_luse.intersects(bb_jgd)].copy()

    if bldg_in.empty:
        st.warning(f"{label}: 建物なし"); return

    # ── 表示切替 ──────────────────────────────────────────────────────────────
    if is_3d:
        # 3D: show_luse=Trueなら土地利用色、Falseならグレー
        layers, records = build_pydeck_layers(
            bldg_in, luse_in, name_map, root_map, sk_sel,
            use_luse_color=show_luse)
        st.session_state[sk_data] = records

        view = pdk.ViewState(
            latitude=(bb_wgs[1]+bb_wgs[3])/2,
            longitude=(bb_wgs[0]+bb_wgs[2])/2,
            zoom=15, pitch=50, bearing=0,
        )
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view,
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            tooltip={
                "html": (
                    "<div style='background:{hex_color};border:2px solid #fff;"
                    "border-radius:6px;padding:6px 10px;box-shadow:0 2px 6px rgba(0,0,0,0.3);'>"
                    "<b style='color:#fff;text-shadow:0 1px 2px rgba(0,0,0,0.6);'>{category}</b>"
                    "<br><span style='color:#fff;font-size:0.85em;'>高さ: {height}m</span>"
                    "<br><span style='color:#fff;font-size:0.85em;'>階数: {storeys}階</span>"
                    "</div>"
                ),
            },
        )
        event = st.pydeck_chart(
            deck, height=460,
            on_select="rerun",
            selection_mode="single-object",
            key=f"pydeck_{area_key}",
        )
        if event and event.selection:
            objs = event.selection.objects or {}
            bldg_objs = objs.get("building-layer", [])
            if bldg_objs:
                clicked_coords = bldg_objs[0].get("coordinates", [])
                clicked_key = tuple(clicked_coords[0]) if clicked_coords else None
                if clicked_key and clicked_key != st.session_state.get(sk_sel):
                    st.session_state[sk_sel] = clicked_key
                    st.rerun()
    else:
        # 2D: show_luse=Trueなら土地利用ポリゴン＋建物輪郭、Falseなら建物輪郭のみ
        render_2d_map(bb_wgs, bldg_in, luse_in, name_map, root_map,
                      area_key, show_luse=show_luse)

    st.caption(f"表示建物数: {len(bldg_in):,} 棟")

    # 選択ビル情報パネル（3Dモードのみ）
    if is_3d:
        selected_id = st.session_state.get(sk_sel)
        if selected_id:
            records = st.session_state.get(sk_data, [])
            def _coords_key(coords):
                return tuple(coords[0]) if coords else ()
            sel_records = [r for r in records
                           if _coords_key(r.get("coordinates", [])) == selected_id]
            if sel_records:
                r = sel_records[0]
                fp   = r.get("fp_area", 0.0)
                st_n = int(r.get("storeys", 1))
                fa   = fp * st_n
                usage = r.get("usage", "—")
                st.markdown(
                    f"<div style='background:#fffbe6;border-left:4px solid #f5c400;"
                    f"border-radius:4px;padding:6px 10px;margin-top:4px;font-size:0.85rem;'>"
                    f"用途: {usage}<br>"
                    f"建築投影面積: <b>{fp:,.1f} m²</b>　"
                    f"階数: <b>{st_n} 階</b>　"
                    f"推定延床面積: <b>{fa:,.1f} m²</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if st.button("選択解除", key=f"desel_{area_key}"):
                    st.session_state[sk_sel] = None
                    st.rerun()

    # α値
    alpha, summary = calc_correction_factor(
        bldg_in, luse_in, bb_jgd, name_map, root_map)
    s_vst = summary.loc[summary["category"].isin(VST_CATEGORIES), "floor_area_m2"].sum()
    s_lcl = summary.loc[summary["category"].isin(LCL_CATEGORIES), "floor_area_m2"].sum()
    alpha_str = f"{alpha:.3f}" if not np.isnan(alpha) else "計算不可"

    st.markdown(
        f"<div style='margin-top:6px;'>"
        f"<span style='font-size:0.8rem;color:#666;'>"
        f"S_Vst: {s_vst/1e4:.2f} 万m²　／　S_Lcl: {s_lcl/1e4:.2f} 万m²</span><br>"
        f"<span style='font-size:0.8rem;color:#666;'>補正係数</span><br>"
        f"<span style='font-size:2rem;font-weight:700;line-height:1.1;'>α = {alpha_str}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if not summary.empty:
        with st.expander("土地利用カテゴリ別 延床面積"):
            summary["Vst"] = summary["category"].isin(VST_CATEGORIES)
            summary["Lcl"] = summary["category"].isin(LCL_CATEGORIES)
            st.dataframe(
                summary[["category","floor_area_m2","Vst","Lcl"]]
                .sort_values("floor_area_m2", ascending=False)
                .rename(columns={"category":"カテゴリ", "floor_area_m2":"延床面積 (m²)"}),
                use_container_width=True, hide_index=True,
            )


# ── center_to_bb ──────────────────────────────────────────────────────────────
def center_to_bb(lat, lon, size_m):
    tf_to   = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG_JGD}", always_xy=True)
    tf_from = Transformer.from_crs(f"EPSG:{EPSG_JGD}", "EPSG:4326", always_xy=True)
    cx, cy  = tf_to.transform(float(lon), float(lat))
    half    = float(size_m) / 2
    lon_min, lat_min = tf_from.transform(cx - half, cy - half)
    lon_max, lat_max = tf_from.transform(cx + half, cy + half)
    coords = (float(lon_min), float(lat_min), float(lon_max), float(lat_max))
    if any(not np.isfinite(v) for v in coords):
        return None
    return coords


# ── 詳細モード ────────────────────────────────────────────────────────────────

def detail_mode(gdf_luse, name_map, root_map):
    center_lat, center_lon = MAP_CENTER

    if "bb_list" not in st.session_state:
        st.session_state.bb_list = []
    if "detail_base_lat" not in st.session_state:
        st.session_state.detail_base_lat = center_lat
    if "detail_base_lon" not in st.session_state:
        st.session_state.detail_base_lon = center_lon

    with st.sidebar:
        st.divider()
        st.header("グリッド設定")
        size_m = st.number_input("ボックスサイズ (m)", min_value=100, max_value=5000,
                                 value=200, step=50)
        n_rows = st.number_input("行数", min_value=1, max_value=20, value=5, step=1)
        n_cols = st.number_input("列数", min_value=1, max_value=20, value=5, step=1)
        st.markdown("**基準点 BB(1-1) の中心**")
        _lat_key = f"base_lat_{st.session_state.detail_base_lat:.6f}"
        _lon_key = f"base_lon_{st.session_state.detail_base_lon:.6f}"
        base_lat = st.number_input("緯度", format="%.6f",
                                   value=st.session_state.detail_base_lat, key=_lat_key)
        base_lon = st.number_input("経度", format="%.6f",
                                   value=st.session_state.detail_base_lon, key=_lon_key)
        st.session_state.detail_base_lat = base_lat
        st.session_state.detail_base_lon = base_lon

        if st.button("🔲 グリッド生成", use_container_width=True):
            tf_to   = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG_JGD}", always_xy=True)
            tf_from = Transformer.from_crs(f"EPSG:{EPSG_JGD}", "EPSG:4326", always_xy=True)
            cx0, cy0 = tf_to.transform(base_lon, base_lat)
            new_list = []
            for r in range(int(n_rows)):
                for c in range(int(n_cols)):
                    cx = cx0 + c * size_m
                    cy = cy0 - r * size_m
                    lon_c, lat_c = tf_from.transform(cx, cy)
                    bb = center_to_bb(lat_c, lon_c, size_m)
                    if bb:
                        new_list.append({"id": f"{r+1}-{c+1}", "lat": lat_c,
                                         "lon": lon_c, "size_m": size_m, "bb_wgs": bb})
            if not new_list:
                st.error("グリッド生成に失敗しました。")
            else:
                st.session_state.bb_list = new_list
                st.session_state.pop("detail_result", None)
                st.rerun()

    col_map, col_heatmap = st.columns([1, 1])

    with col_map:
        st.subheader("マップ")
        if st.session_state.bb_list:
            map_center_lat = float(np.mean([e["lat"] for e in st.session_state.bb_list]))
            map_center_lon = float(np.mean([e["lon"] for e in st.session_state.bb_list]))
        else:
            map_center_lat = st.session_state.detail_base_lat
            map_center_lon = st.session_state.detail_base_lon
        m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=14,
                       tiles="CartoDB positron")
        folium.CircleMarker(
            location=[st.session_state.detail_base_lat, st.session_state.detail_base_lon],
            radius=1, color="black", fill=True, fill_opacity=1.0, tooltip="基準点",
        ).add_to(m)
        for entry in st.session_state.bb_list:
            bb = entry["bb_wgs"]
            folium.Rectangle(
                bounds=[[bb[1], bb[0]], [bb[3], bb[2]]],
                color="steelblue", fill=True, fill_opacity=0.15, tooltip=entry["id"],
            ).add_to(m)

        shape = get_coverage_shape()
        if shape:
            folium.GeoJson(
                shape.__geo_interface__,
                style_function=lambda _: {
                    "color": "#3388ff",
                    "weight": 1.5,
                    "fillColor": "#3388ff",
                    "fillOpacity": 0.06,
                },
            ).add_to(m)
        
        map_data = st_folium(m, width="100%", height=460, key="detail_map")

        click = map_data.get("last_clicked") if map_data else None
        if click and isinstance(click, dict):
            clat, clng = click.get("lat"), click.get("lng")
            if clat and clng:
                if (abs(clat - st.session_state.detail_base_lat) > 1e-7 or
                        abs(clng - st.session_state.detail_base_lon) > 1e-7):
                    st.session_state.detail_base_lat = clat
                    st.session_state.detail_base_lon = clng
                    st.rerun()

        if st.session_state.bb_list:
            st.caption(f"登録 BB 数: {len(st.session_state.bb_list)}")

            del_col1, del_col2 = st.columns([3, 1])
            with del_col1:
                del_id = st.selectbox(
                    "削除するBB",
                    [e["id"] for e in st.session_state.bb_list],
                    key="del_select",
                    label_visibility="collapsed",
                )
            with del_col2:
                if st.button("🗑️ 削除", use_container_width=True):
                    st.session_state.bb_list = [
                        e for e in st.session_state.bb_list if e["id"] != del_id]
                    st.session_state.pop("detail_result", None)
                    st.rerun()

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🗑️ 全削除", use_container_width=True):
                    st.session_state.bb_list = []
                    st.session_state.pop("detail_result", None)
                    st.rerun()
            with c2:
                run_calc = st.button("▶️ 全BB一括計算", use_container_width=True)
        else:
            st.info("サイドバーからBBを追加してください")
            run_calc = False

    if run_calc:
        results = []
        prog = st.progress(0)
        for i, entry in enumerate(st.session_state.bb_list):
            bb_wgs = entry["bb_wgs"]
            files  = find_overlapping_files(bb_wgs)
            if not files:
                results.append({"id": entry["id"], "緯度": entry["lat"],
                                 "経度": entry["lon"], "サイズ(m)": entry["size_m"],
                                 "建物数": 0, "S_Vst (m²)": 0, "S_Lcl (m²)": 0,
                                 "α": None})
                prog.progress((i+1) / len(st.session_state.bb_list))
                continue
            gdf_b   = load_buildings_for_bb(tuple(str(f) for f in files))
            bb_jgd  = (gpd.GeoDataFrame(geometry=[box(*bb_wgs)], crs=EPSG_WGS84)
                       .to_crs(epsg=EPSG_JGD).geometry[0])
            bldg_in = gdf_b[gdf_b.intersects(bb_jgd)].copy()
            luse_in = gdf_luse[gdf_luse.intersects(bb_jgd)].copy()
            alpha, summary = calc_correction_factor(
                bldg_in, luse_in, bb_jgd, name_map, root_map)
            s_vst = summary.loc[summary["category"].isin(VST_CATEGORIES),
                                 "floor_area_m2"].sum()
            s_lcl = summary.loc[summary["category"].isin(LCL_CATEGORIES),
                                 "floor_area_m2"].sum()
            cat_areas = {
                f"{row['category']} (m²)": round(row["floor_area_m2"], 1)
                for _, row in summary.iterrows()
            } if not summary.empty else {}

            results.append({"id": entry["id"], "緯度": entry["lat"],
                             "経度": entry["lon"], "サイズ(m)": entry["size_m"],
                             "建物数": len(bldg_in),
                             "S_Vst (m²)": round(s_vst, 1),
                             "S_Lcl (m²)": round(s_lcl, 1),
                             "α": round(alpha, 4) if not np.isnan(alpha) else None,
                             **cat_areas})
            prog.progress((i+1) / len(st.session_state.bb_list))
        st.session_state["detail_result"] = pd.DataFrame(results)
        prog.empty()
        st.rerun()

    with col_heatmap:
        st.subheader("補正係数ヒートマップ")
        if "detail_result" in st.session_state:
            hm_colormap = st.session_state.get("hm_colormap", "Cividis")
            render_alpha_heatmap(
                st.session_state["detail_result"],
                st.session_state.bb_list,
                hm_colormap,
            )
            st.selectbox(
                "カラーマップ",
                options=list(COLORMAP_PRESETS.keys()),
                index=list(COLORMAP_PRESETS.keys()).index(hm_colormap),
                key="hm_colormap",
                label_visibility="collapsed",
            )
        else:
            st.info("「▶️ 全BB一括計算」を実行するとここにヒートマップが表示されます")

    if "detail_result" in st.session_state:
        st.divider()
        with st.expander("📋 計算結果テーブル", expanded=False):
            df = st.session_state["detail_result"]
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("⬇️ CSVダウンロード", data=csv,
                               file_name="correction_factors.csv",
                               mime="text/csv", use_container_width=True)


# ── メインUI ──────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="PLATEAU Space Accessibility Viewer", layout="wide")

    name_map, root_map = load_mappings()
    gdf_luse = load_landuse()
    center_lat, center_lon = MAP_CENTER

    with st.sidebar:
        st.header("モード")
        app_mode = st.radio("", ["比較", "詳細"], label_visibility="collapsed")

    # ══════════════════════════════════════════════════════
    # 比較モード
    # ══════════════════════════════════════════════════════
    if app_mode == "比較":
        if "compare_bbs" not in st.session_state:
            st.session_state.compare_bbs    = []
        if "compare_labels" not in st.session_state:
            st.session_state.compare_labels = []

        # 地図パネル
        col_map, col_info = st.columns([3, 1])
        with col_map:
            m = folium.Map(location=[center_lat, center_lon], zoom_start=14,
                           tiles="CartoDB positron")
            Draw(export=False, draw_options={
                "rectangle": True, "polygon": False, "polyline": False,
                "circle": False, "marker": False, "circlemarker": False,
            }).add_to(m)

            shape = get_coverage_shape()
            if shape:
                folium.GeoJson(
                    shape.__geo_interface__,
                    style_function=lambda _: {
                        "color": "#3388ff",
                        "weight": 1.5,
                        "fillColor": "#3388ff",
                        "fillOpacity": 0.06,
                    },
                ).add_to(m)
            
            map_data = st_folium(m, width="100%", height=600, key="compare_map")

        drawings = (map_data or {}).get("all_drawings", []) or []
        raw_bbs = []
        for d in drawings[:3]:
            if d.get("geometry", {}).get("type") == "Polygon":
                coords = d["geometry"]["coordinates"][0]
                bb = (
                    min(c[0] for c in coords), min(c[1] for c in coords),
                    max(c[0] for c in coords), max(c[1] for c in coords),
                )
                raw_bbs.append(bb)

        new_bbs = []
        for i, bb in enumerate(raw_bbs):
            new_bbs.append(bb)

        if new_bbs != [b for b, _ in st.session_state.compare_bbs]:
            old_labels = [lbl for _, lbl in st.session_state.compare_bbs]
            new_entries = []
            for i, bb in enumerate(new_bbs):
                if i < len(old_labels):
                    lbl = old_labels[i]
                else:
                    files = find_overlapping_files(bb)
                    auto_lbl = ""
                    if files:
                        gdf_tmp  = load_buildings_for_bb(tuple(str(f) for f in files))
                        bb_jgd   = (gpd.GeoDataFrame(geometry=[box(*bb)], crs=EPSG_WGS84)
                                    .to_crs(epsg=EPSG_JGD).geometry[0])
                        bldg_tmp = gdf_tmp[gdf_tmp.intersects(bb_jgd)]
                        auto_lbl = suggest_label_from_bldg(bldg_tmp)
                    lbl = auto_lbl if auto_lbl else DEFAULT_LABELS[i]
                new_entries.append((bb, lbl))
            st.session_state.compare_bbs = new_entries

        # ラベル編集
        if st.session_state.compare_bbs:
            label_cols = st.columns(len(st.session_state.compare_bbs))
            new_labels = []
            for i, (bb, lbl) in enumerate(st.session_state.compare_bbs):
                with label_cols[i]:
                    new_lbl = st.text_input(f"エリア{i+1}のラベル", value=lbl,
                                            key=f"lbl_{i}", label_visibility="collapsed")
                    new_labels.append(new_lbl)
            st.session_state.compare_bbs = [
                (bb, new_labels[i])
                for i, (bb, _) in enumerate(st.session_state.compare_bbs)
            ]
        
        with col_info:
            st.caption("地図左側の ▭ ボタンで矩形を描いてエリアを指定（最大3つ）")
            if st.session_state.compare_bbs:
                for _, lbl in st.session_state.compare_bbs:
                    st.success(f"✓ {lbl}")
                st.info("↓ スクロールして結果を確認")
            else:
                st.info("エリア未選択")

        st.divider()

        # ── 3Dビュー / 土地利用2D 切替 ──────────────────────────────────────
        n = len(st.session_state.compare_bbs)
        if n == 0:
            st.info("上の地図で矩形を描くとエリアが追加されます")
        else:
            # ── ヘッダー行：タイトル／2D・3D切替／土地利用チェック ──────────
            seg_col, chk_col = st.columns([1, 1])
            with seg_col:
                st.markdown("<div style='margin-top:6px;'>", unsafe_allow_html=True)
                is_2d = st.toggle(
                    "2D",
                    value=st.session_state.get("view_mode_compare", False),
                    key="view_mode_compare",
                )
                st.markdown("</div>", unsafe_allow_html=True)
            with chk_col:
                st.markdown("<div style='margin-top:10px;'>", unsafe_allow_html=True)
                show_luse = st.checkbox(
                    "土地利用",
                    value=st.session_state.get("show_luse_compare", True),
                    key="show_luse_compare",
                )
                st.markdown("</div>", unsafe_allow_html=True)

            is_3d = not is_2d

            if n == 1:
                col_view, col_leg = st.columns([9, 1])
                with col_view:
                    bb, lbl = st.session_state.compare_bbs[0]
                    st.markdown(f"**{lbl}**")
                    render_area(bb, lbl, gdf_luse, name_map, root_map,
                                area_key="area_0", col_idx=0,
                                is_3d=is_3d, show_luse=show_luse)
                with col_leg:
                    render_legend_vertical()
            elif n == 2:
                col0, col1, col_leg = st.columns([4, 4, 1])
                for col, (bb, lbl), key in zip(
                    [col0, col1],
                    st.session_state.compare_bbs,
                    ["area_0", "area_1"],
                ):
                    with col:
                        st.markdown(f"**{lbl}**")
                        render_area(bb, lbl, gdf_luse, name_map, root_map,
                                    area_key=key, col_idx=0,
                                    is_3d=is_3d, show_luse=show_luse)
                with col_leg:
                    render_legend_vertical()
            else:
                col0, col1, col2, col_leg = st.columns([3, 3, 3, 1])
                for col, (bb, lbl), key, idx in zip(
                    [col0, col1, col2],
                    st.session_state.compare_bbs,
                    ["area_0", "area_1", "area_2"],
                    [0, 1, 2],
                ):
                    with col:
                        st.markdown(f"**{lbl}**")
                        render_area(bb, lbl, gdf_luse, name_map, root_map,
                                    area_key=key, col_idx=idx,
                                    is_3d=is_3d, show_luse=show_luse)
                with col_leg:
                    render_legend_vertical()

    # ══════════════════════════════════════════════════════
    # 詳細モード
    # ══════════════════════════════════════════════════════
    else:
        detail_mode(gdf_luse, name_map, root_map)


if __name__ == "__main__":
    main()
