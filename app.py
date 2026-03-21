# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, url_for, jsonify
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, mapping
from shapely.ops import unary_union
import folium
import os
import sys
import unicodedata
from io import BytesIO
from datetime import datetime
from xhtml2pdf import pisa
from flask import make_response, send_file, current_app
from flask import flash, session
import yaml
from pyproj import Geod
import hashlib
import re
from datetime import timedelta



# Detectar si se ejecuta como .exe
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

# ---- ConfiguraciÃ³n (config.yaml) ----
CFG = {}
CFG_PATH = os.path.join(BASE_DIR, "config.yaml")
if os.path.exists(CFG_PATH):
    try:
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            CFG = yaml.safe_load(f) or {}
    except Exception as e:
        print("[config] Error cargando config.yaml:", e)
else:
    print("[config] config.yaml no encontrado; usando rutas por defecto relativas a BASE_DIR")

# ---- Carga de datos base ----
paths_cfg = (CFG.get("paths") or {})
path_est = paths_cfg.get("establecimientos") or os.path.join(BASE_DIR, "establecimientos.geojson")
es = gpd.read_file(path_est); es = es.to_crs("EPSG:4326")

# Asegurar un GeoDataFrame de establecimientos con geometrÃ­a en EPSG:4326
if "geometry" in es.columns and es.geometry.notna().any():
    # El GeoJSON ya trae geometrÃ­a
    es_gdf = es.to_crs("EPSG:4326")
elif {"x_gis", "y_gis"}.issubset(es.columns):
    # Construir puntos a partir de columnas x_gis (longitud), y_gis (latitud)
    es_gdf = gpd.GeoDataFrame(
        es.copy(),
        geometry=gpd.points_from_xy(es["x_gis"], es["y_gis"]),
        crs="EPSG:4326",
    )
else:
    raise ValueError(
        "No se encontrÃ³ 'geometry' ni columnas x_gis/y_gis para establecimientos. "
        "Agrega la geometrÃ­a o las columnas necesarias."
    )

# (opcional) estandariza nombre para evitar problemas de tipos
es_gdf["est_nombre"] = es_gdf["est_nombre"].astype(str)


path_iso = paths_cfg.get("isocronas_todas") or os.path.join(BASE_DIR, "isocronas_todas.geojson")
gdf_iso = gpd.read_file(path_iso); gdf_iso = gdf_iso.to_crs("EPSG:4326")
path_pop = paths_cfg.get("poblacion_puntos") or os.path.join(BASE_DIR, "poblacion_puntos.csv")
pop_rp = pd.read_csv(path_pop)

geometry = [Point(xy) for xy in zip(pop_rp['longitude'], pop_rp['latitude'])]
pop_gdf = gpd.GeoDataFrame(pop_rp, geometry=geometry, crs="EPSG:4326")

if "poblacion" not in gdf_iso.columns:
    sindex = pop_gdf.sindex
    poblaciones = []
    for idx, row in gdf_iso.iterrows():
        poligono = row["geometry"]
        posibles_idx = list(sindex.intersection(poligono.bounds))
        posibles_puntos = pop_gdf.iloc[posibles_idx]
        dentro = posibles_puntos[posibles_puntos.geometry.within(poligono)]
        total_poblacion = dentro["per_general_2020"].sum()
        poblaciones.append(total_poblacion)
    gdf_iso["poblacion"] = poblaciones

establecimientos = sorted(es["est_nombre"].unique())
tiempos = [5, 10, 15, 20, 25, 30]
colores = {5: "green", 10: "orange", 15: "gold", 20: "orangered", 25: "red", 30: "purple"}

# ===== Recorte a 4 distritos (definiciÃ³n global) =====
# (Partes previas tuyas)
DIST_UNION_4D = None

# Reemplazo: cargar distritos desde config (dinÃ¡mico)
try:
    path_dist = (CFG.get('paths') or {}).get('distritos') or os.path.join(BASE_DIR, 'data', 'admin', 'piura_distritos.geojson')
    if os.path.exists(path_dist):
        distritos_piura = gpd.read_file(path_dist).to_crs('EPSG:4326')
    else:
        print(f"[admin] No existe capa de distritos: {path_dist}. Se operarÃ¡ sin recorte.")
        distritos_piura = None
except Exception as e:
    print('[admin] Error cargando distritos:', e)
    distritos_piura = None

# Normalizador

def _norm(txt):
    if pd.isna(txt):
        return ""
    s = str(txt)
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    return " ".join(s.upper().strip().split())

# distritos (ajusta el nombre si tu shapefile no usa 'DISTRITO')
DIST_NAME_COL = "DISTRITO"
if distritos_piura is not None and (DIST_NAME_COL in distritos_piura.columns):
    distritos_piura["DIST_NORM"] = distritos_piura[DIST_NAME_COL].map(_norm)

# establecimientos: viene en la columna nom_dist
if "nom_dist" in es_gdf.columns:
    es_gdf["NOM_DIST_NORM"] = es_gdf["nom_dist"].map(_norm)
else:
    es_gdf["NOM_DIST_NORM"] = ""



# ðŸ‘‰ AHORA sÃ­ define las constantes:
DIST_NAME_COL = "DISTRITO"
ALL_DISTRICTS = sorted(distritos_piura[DIST_NAME_COL].unique().tolist()) if (distritos_piura is not None and DIST_NAME_COL in distritos_piura.columns) else []
DEFAULT_DISTRICTS = ["PIURA", "CASTILLA", "CATACAOS", "VEINTISEIS DE OCTUBRE"]


# ===== Universo total de poblaciÃ³n en los 4 distritos (cÃ¡lculo automÃ¡tico) =====
# Requiere que pop_gdf exista y estÃ© en EPSG:4326
try:
    # Join espacial puntos -> 4 distritos
    pts_4d = gpd.sjoin(
        pop_gdf[['per_general_2020', 'geometry']],
        distritos_piura[['DISTRITO', 'geometry']],
        how='inner',
        predicate='intersects'
    )
    TOTAL_POBLACION_4D = int(pts_4d['per_general_2020'].sum())
    print("TOTAL_POBLACION_4D =", TOTAL_POBLACION_4D)
except Exception as e:
    print("âš ï¸ No se pudo calcular TOTAL_POBLACION_4D automÃ¡ticamente:", e)
    TOTAL_POBLACION_4D = 514_653  # fallback manual

def build_district_union_and_total(selected_names):
    if distritos_piura is None or "DIST_NORM" not in (distritos_piura.columns if hasattr(distritos_piura, 'columns') else []):
        return None, 0, gpd.GeoDataFrame(columns=["geometry"])  # fallback sin recorte
    sel_norm = [_norm(x) for x in selected_names]
    sel = distritos_piura[distritos_piura["DIST_NORM"].isin(sel_norm)].copy()
    if sel.empty:
        return None, 0, sel
    dist_union = sel.union_all() if hasattr(sel, "union_all") else sel.unary_union
    pts_sel = gpd.sjoin(
        pop_gdf[["per_general_2020", "geometry"]],
        sel[["geometry"]],
        how="inner",
        predicate="intersects",
    )
    total = int(pts_sel["per_general_2020"].sum())
    return dist_union, total, sel


def establecimientos_en_distritos(selected_names):
    sel_norm = {_norm(x) for x in selected_names}

    # 1) Filtro por atributo 'nom_dist'
    out = []
    if "NOM_DIST_NORM" in es_gdf.columns and es_gdf["NOM_DIST_NORM"].ne("").any():
        out = (
            es_gdf.loc[es_gdf["NOM_DIST_NORM"].isin(sel_norm), "est_nombre"]
                  .astype(str)
                  .dropna()
                  .unique()
                  .tolist()
        )
        out.sort()

    # 2) Fallback espacial si no hubo match por atributo
    if not out:
        sel = distritos_piura[distritos_piura["DIST_NORM"].isin(sel_norm)][["geometry"]] if (distritos_piura is not None and "DIST_NORM" in distritos_piura.columns) else gpd.GeoDataFrame(columns=["geometry"])
        if not sel.empty:
            join = gpd.sjoin(
                es_gdf[["est_nombre","geometry"]],
                sel,
                how="inner",
                predicate="intersects",
            )
            out = sorted(join["est_nombre"].astype(str).unique().tolist())

    print(f"[Filtro-ES] distritos_sel={selected_names} -> {len(out)} E.S.")
    # debug opcional: ver valores normales disponibles en ES
    # print("ES NOM_DIST_NORM Ãºnicos (ejemplo):", es_gdf["NOM_DIST_NORM"].dropna().unique()[:10])
    return out



def _cleanup_old_maps(dir_path: str, days: int = 7):
    try:
        import time
        now = time.time()
        max_age = days * 24 * 3600
        if not os.path.isdir(dir_path):
            return
        for fn in os.listdir(dir_path):
            if not fn.lower().endswith('.html'):
                continue
            fp = os.path.join(dir_path, fn)
            try:
                if now - os.path.getmtime(fp) > max_age:
                    os.remove(fp)
            except Exception:
                pass
    except Exception:
        pass


def _deterministic_map_name(prov_code, dist_codes, estabs, tiempos):
    prov_part = (str(prov_code).zfill(2)) if prov_code else 'XX'
    dist_part = '-'.join(sorted([str(c) for c in dist_codes])) if dist_codes else 'ALL'
    base_str = '|'.join([
        prov_part,
        dist_part,
        ','.join(sorted([str(e) for e in (estabs or [])])),
        ','.join(sorted([str(t) for t in (tiempos or [])])),
    ])
    h = hashlib.sha1(base_str.encode('utf-8')).hexdigest()[:8]
    return f"mapas/mapa_{prov_part}_{dist_part}_{h}.html"


def calcular_cobertura_multiple(estabs_seleccionados, tiempos_seleccionados, distritos_seleccionados):
    dist_union, total_universo, gdf_dist_sel = build_district_union_and_total(
        distritos_seleccionados or DEFAULT_DISTRICTS
    )

    from datetime import datetime
    from shapely.prepared import prep

    # ---- Ãndice espacial ----
    SINDEX = pop_gdf.sindex

    # 1) Unir isÃ³cronas por establecimiento (solo tiempos seleccionados)
    union_por_estab = {}
    for est in estabs_seleccionados:
        sub = gdf_iso[(gdf_iso["est_nombre"] == est) & (gdf_iso["tiempo"].isin(tiempos_seleccionados))]
        if sub.empty:
            continue
        geom_est = unary_union(sub.geometry)
        if geom_est.is_empty:
            continue
        union_por_estab[est] = geom_est

    if not union_por_estab:
        return "<b>No hay geometrÃ­as para los filtros seleccionados.</b>", "default_map.html"

    # 2) Ãrea y poblaciÃ³n por establecimiento (sin doble conteo intra-estab)
    poblacion_por_estab = {}
    area_por_estab_m2 = {}
    geometries = []
    for est, geom in union_por_estab.items():
        geometries.append(geom)

        # Calcular Ã¡rea geodÃ©sica en m^2 sobre el elipsoide WGS84
        geod = Geod(ellps="WGS84")
        def _area_geod_m2(g):
            if g.is_empty:
                return 0.0
            if g.geom_type == "Polygon":
                x, y = g.exterior.xy
                a, _ = geod.polygon_area_perimeter(x, y)
                total = abs(a)
                for ring in g.interiors:
                    xi, yi = ring.xy
                    a2, _ = geod.polygon_area_perimeter(xi, yi)
                    total -= abs(a2)
                return float(abs(total))
            if g.geom_type == "MultiPolygon":
                return float(sum(_area_geod_m2(p) for p in g.geoms))
            return 0.0
        area_m2 = _area_geod_m2(geom)
        area_por_estab_m2[est] = float(area_m2)

        prep_geom = prep(geom)
        cand_idx = list(SINDEX.intersection(geom.bounds))
        cand_pts = pop_gdf.iloc[cand_idx]
        dentro = cand_pts[cand_pts.geometry.apply(prep_geom.intersects)]
        poblacion_por_estab[est] = int(dentro["per_general_2020"].sum())

        # 3) UniÃ³n global de isÃ³cronas
    union_global = unary_union(geometries)
    if union_global.is_empty:
        return "<b>Las geometrÃ­as seleccionadas estÃ¡n vacÃ­as.</b>", "default_map.html"

    # Arreglar posibles geometrÃ­as invÃ¡lidas
    union_global = union_global.buffer(0)

    # --- Recorte geomÃ©trico a los distritos seleccionados ---
    aviso_recorte = ""
    union_recortada = union_global

    if dist_union is not None and not getattr(dist_union, "is_empty", False):
        try:
            union_recortada = union_global.intersection(dist_union.buffer(0))
        except Exception as e:
            print("âš ï¸ Error al intersectar union_global con dist_union:", e)
            union_recortada = None
    else:
        union_recortada = None

    # Preparar nombre determinÃ­stico (prov/dist/hash) y limpieza
    prov_code = None
    dist_codes = []
    try:
        if gdf_dist_sel is not None and not gdf_dist_sel.empty and all(c in gdf_dist_sel.columns for c in ['CCPP','CCDI']):
            codes_series = gdf_dist_sel['CCPP'].astype(str).str.zfill(2) + gdf_dist_sel['CCDI'].astype(str).str.zfill(2)
            dist_codes = sorted(codes_series.unique().tolist())
            provs = gdf_dist_sel['CCPP'].astype(str).str.zfill(2).unique().tolist()
            prov_code = provs[0] if len(provs) == 1 else None
    except Exception:
        pass
    mapas_dir = os.path.join(BASE_DIR, 'static', 'mapas')
    os.makedirs(mapas_dir, exist_ok=True)
    _cleanup_old_maps(mapas_dir, days=7)

    if (union_recortada is None) or getattr(union_recortada, "is_empty", False):
       # Fallback visual para no romper la UI
       center = [es["y_gis"].mean(), es["x_gis"].mean()]
       m = folium.Map(location=center, zoom_start=13)
       for est in union_por_estab.keys():
           fila = es[es["est_nombre"] == est].iloc[0]
           folium.Marker(location=[fila["y_gis"], fila["x_gis"]], tooltip=est).add_to(m)
       rel_name = _deterministic_map_name(prov_code, dist_codes, list(union_por_estab.keys()), tiempos_seleccionados)
       m.save(os.path.join(BASE_DIR, 'static', rel_name))
       return "<b>No hay cobertura dentro de los distritos seleccionados.</b>", rel_name

    # etiqueta amigable de recorte
    aviso_recorte = " (recortado a: " + ", ".join(distritos_seleccionados) + ")"
    

    # 4) Cobertura Ãºnica total usando la uniÃ³n recortada
    geod = Geod(ellps="WGS84")
    def _area_geod_m2(g):
        if g.is_empty:
            return 0.0
        if g.geom_type == "Polygon":
            x, y = g.exterior.xy
            a, _ = geod.polygon_area_perimeter(x, y)
            total = abs(a)
            for ring in g.interiors:
                xi, yi = ring.xy
                a2, _ = geod.polygon_area_perimeter(xi, yi)
                total -= abs(a2)
            return float(abs(total))
        if g.geom_type == "MultiPolygon":
            return float(sum(_area_geod_m2(p) for p in g.geoms))
        return 0.0
    union_area_m2 = _area_geod_m2(union_recortada)

    from shapely.prepared import prep
    prep_union_recort = prep(union_recortada)

    # âš ï¸ Usa el bbox de la GEOMETRÃA RECORTADA (no el de union_global)
    cand_idx_all = list(SINDEX.intersection(union_recortada.bounds))
    cand_pts_all = pop_gdf.iloc[cand_idx_all]

    # QuÃ©datate solo con puntos dentro del recorte
    pts_en_union = cand_pts_all[cand_pts_all.geometry.apply(prep_union_recort.intersects)]
    cobertura_total_unica = int(pts_en_union["per_general_2020"].sum())

    print(f"[Diag] candidatos:{len(cand_pts_all)} | en_recorte:{len(pts_en_union)} | suma={cobertura_total_unica}")

    # 5) PoblaciÃ³n en solape (â‰¥2 E.S.) dentro del recorte
    def cuenta_cover_por_establecimiento(pt):
        c = 0
        for _, geom in union_por_estab.items():
            if geom.bounds[0] <= pt.x <= geom.bounds[2] and geom.bounds[1] <= pt.y <= geom.bounds[3]:
                if geom.intersects(pt):
                    c += 1
        return c

    covers = pts_en_union.geometry.apply(cuenta_cover_por_establecimiento)
    mask_inter = covers >= 2
    poblacion_inter = int(pts_en_union.loc[mask_inter, "per_general_2020"].sum())
    porcentaje_inter = (poblacion_inter / cobertura_total_unica * 100.0) if cobertura_total_unica > 0 else 0.0

    # 5.1) GeometrÃ­a de solape (â‰¥2) recortada a 4D
    intersecciones = []
    keys = list(union_por_estab.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            gi = union_por_estab[keys[i]]
            gj = union_por_estab[keys[j]]
            inter = gi.intersection(gj)
            if dist_union is not None:
                inter = inter.intersection(dist_union.buffer(0))
            if inter.is_valid and not inter.is_empty:
                intersecciones.append(inter)
    inter_geom = unary_union(intersecciones) if intersecciones else None
    area_inter_m2 = 0.0
    if inter_geom and not inter_geom.is_empty:
        area_inter_m2 = _area_geod_m2(inter_geom)

    # 7) Mapa
    center = [es["y_gis"].mean(), es["x_gis"].mean()]
    m = folium.Map(location=center, zoom_start=13)

    # LÃ­mite de 4 distritos
    if dist_union is not None and not dist_union.is_empty:
        folium.GeoJson(
            mapping(dist_union),
            style_function=lambda x: {"color": "black", "weight": 2, "fillOpacity": 0.0},
            tooltip="LÃ­mite distritos"
        ).add_to(m)

    # Marcadores E.S.
    for est in union_por_estab.keys():
        fila = es[es["est_nombre"] == est].iloc[0]
        folium.Marker(location=[fila["y_gis"], fila["x_gis"]], tooltip=est).add_to(m)

    # Capa por tiempo (solo visual)
    colores = {5: "green", 10: "orange", 15: "gold", 20: "orangered", 25: "red", 30: "purple"}
    for est in estabs_seleccionados:
        for t in tiempos_seleccionados:
            zona = gdf_iso[(gdf_iso["est_nombre"] == est) & (gdf_iso["tiempo"] == t)]
            if zona.empty:
                continue
            geom_t = unary_union(zona.geometry)
            # recortar cada capa al lÃ­mite (para que lo visual coincida con el cÃ¡lculo)
            if dist_union is not None:
                geom_t = geom_t.intersection(dist_union)
                if geom_t.is_empty:
                    continue
            folium.GeoJson(
                mapping(geom_t),
                style_function=lambda x, t=t: {"fillColor": colores.get(t, "gray"),
                                               "color": colores.get(t, "gray"),
                                               "weight": 1, "fillOpacity": 0.25}
            ).add_to(m)

    # Sombreado de cobertura mÃºltiple (â‰¥2)
    if inter_geom and not inter_geom.is_empty:
        folium.GeoJson(
            mapping(inter_geom),
            style_function=lambda x: {"fillColor": "blue", "color": "blue", "weight": 2, "fillOpacity": 0.30},
            tooltip="Cobertura mÃºltiple (â‰¥2 E.S.)"
        ).add_to(m)

    # 8) Guardar mapa determinÃ­stico
    rel_name = _deterministic_map_name(prov_code, dist_codes, list(union_por_estab.keys()), tiempos_seleccionados)
    m.save(os.path.join(BASE_DIR, 'static', rel_name))

    # 9) PoblaciÃ³n sin cobertura (comparada con universo 4D)
    total_4d = total_universo
    # (fallback: si no definiste TOTAL_POBLACION_4D, lo puedes calcular una sola vez fuera con sjoin)
    sin_cobertura = None
    if isinstance(total_4d, (int, float)):
        sin_cobertura = max(int(total_4d - cobertura_total_unica), 0)

    # 10) Texto
    texto = "<b>Resultados{}:</b><br>".format(aviso_recorte)
    for est in sorted(union_por_estab.keys()):
        texto += f"- {est}: {poblacion_por_estab.get(est,0)} personas, {area_por_estab_m2.get(est,0)/1e6:.2f} km²<br>"
    texto += "<br><b>Cobertura única total:</b> {:,} personas".format(cobertura_total_unica).replace(",", ".")
    texto += "<br><b>En solape (≥2 E.S.):</b> {:,} personas ({:.1f}%)".format(poblacion_inter, porcentaje_inter).replace(",", ".")
    texto += "<br><b>Área total única (unión):</b> {:.2f} km²".format(union_area_m2/1e6)
    texto += "<br><b>Área con solape (≥2):</b> {:.2f} km²".format(area_inter_m2/1e6)

    label_distritos = ", ".join(distritos_seleccionados) if distritos_seleccionados else "distritos seleccionados"
    num_distritos = len(distritos_seleccionados) if distritos_seleccionados else 0
    if sin_cobertura is not None:
        texto += "<br><b>Población sin cobertura ({}):</b> {:,} personas".format(
            label_distritos, sin_cobertura
        ).replace(",", ".")
        texto += "<br><small>Universo {} distrito(s): {:,} personas.</small>".format(
            num_distritos, int(total_4d)
        ).replace(",", ".")

    # Preparar métricas resumidas para reporte
    # Override: usar contexto persistido en sesión
    # (Función pura) No tocar sesión ni renderizar plantillas aquí.

    try:
        porc_inter = (poblacion_inter / cobertura_total_unica * 100.0) if cobertura_total_unica > 0 else 0.0
    except Exception:
        porc_inter = 0.0
    metricas = {
        "poblacion_unica": int(cobertura_total_unica),
        "poblacion_solape": int(poblacion_inter),
        "porcentaje_solape": float(porc_inter),
        "area_unica_km2": float(union_area_m2/1e6),
        "area_solape_km2": float(area_inter_m2/1e6),
        "resumen_texto": _strip_emojis(texto) if '_strip_emojis' in globals() else texto,
    }
    return texto, rel_name, metricas



@app.route('/situacion-actual', methods=['GET', 'POST'])
def situacion_actual():
    mapas = [
        "mapa_piura_pob.html",
        "mapa_interactivo2.html",
        "isocronas_ORS.html"
    ]
    nombre_amigable = {
        "mapa_piura_pob.html": "Densidad poblacional",
        "mapa_interactivo2.html": "Establecimientos por distrito",
        "isocronas_ORS.html": "DistribuciÃ³n actual de E.S."
    }

    static_dir = os.path.join(BASE_DIR, "static")
    disponibles = set(os.listdir(static_dir))
    mapas_filtrados = [f for f in mapas if f in disponibles]

    return render_template("situacion_actual.html", mapas=mapas_filtrados, nombres=nombre_amigable, active_page="situacion_actual")

@app.route('/diagnostico', methods=['GET', 'POST'])
def diagnostico():
    form = request.form
    if request.method == 'POST':
        try:
            print("[Diag] form_raw:", form.to_dict(flat=False))
        except Exception:
            pass
    prov = form.get('prov') or form.get('provincia')
    dists = form.getlist('dist') or form.getlist('dist[]') or form.getlist('distritos') or form.getlist('distritos[]')
    ests_param  = form.getlist('est') or form.getlist('est[]') or form.getlist('establecimientos') or form.getlist('establecimientos[]')
    # Admite CSV en 'dist' (p.ej.: '0101,0104')
    if isinstance(dists, list) and len(dists) == 1 and (',' in str(dists[0])):
        dists = [s.strip() for s in str(dists[0]).split(',') if s.strip()]
    # Leer y validar tiempos desde el formulario (sin defaults en POST)
    raw_tiempos = request.form.getlist("tiempos") or request.form.getlist("tiempos[]")
    tiempos_int = sorted({int(t) for t in raw_tiempos if str(t).isdigit()})
    print(f"[Diag] tiempos_recibidos={raw_tiempos} -> usados={tiempos_int}")
    if request.method == 'POST' and not tiempos_int:
        flash("Selecciona uno o más tiempos.", "warn")
        return redirect(url_for("diagnostico"))
    # En GET no preseleccionamos tiempos; el usuario debe elegir

    # 1) Distritos elegidos (por cÃ³digos si vienen, o nombres/defecto)
    if dists and (distritos_piura is not None):
        df = distritos_piura
        codes_norm = [str(c).zfill(4) for c in dists]
        series_code = df['CCPP'].astype(str).str.zfill(2) + df['CCDI'].astype(str).str.zfill(2)
        name_col = DIST_NAME_COL if (DIST_NAME_COL in df.columns) else df.columns[0]
        distritos_sel = df.loc[series_code.isin(codes_norm), name_col].astype(str).tolist()
    else:
        distritos_sel = form.getlist("distritos") or DEFAULT_DISTRICTS

    # 2) Establecimientos disponibles segÃºn distritos
    establecimientos_opts = establecimientos_en_distritos(distritos_sel)

    # 3) Lo que marcÃ³ el usuario
    seleccionados = [e for e in ests_param if e in establecimientos_opts] if ests_param else []
    # tiempos (ya leidos arriba)

    # 3.1 MantÃ©n solo los ES que siguen dentro de la zona seleccionada
    if seleccionados:
      seleccionados = [e for e in seleccionados if e in establecimientos_opts]

    resultados, mapa_html, metricas = "", "default_map.html", None

    # 4) Solo calcula si hay selección de ES y tiempos
    if seleccionados and tiempos_int:
        # Pre-sembrar contexto para evitar early-return dentro de calcular_cobertura_multiple
        try:
            session['ultimo_reporte'] = {
                "provincia": prov,
                "distritos": distritos_sel,
                "establecimientos": seleccionados,
                "tiempos": tiempos_int,
                "metricas": {},
                "mapa_html": None,
            }
        except Exception:
            pass
        _ret = calcular_cobertura_multiple(seleccionados, tiempos_int, distritos_sel)
        if isinstance(_ret, tuple):
            if len(_ret) == 3:
                resultados, mapa_html, metricas = _ret
            elif len(_ret) == 2:
                resultados, mapa_html = _ret
                metricas = {}
        elif hasattr(_ret, 'status_code'):
            return _ret
        # Si no vinieron métricas, intenta extraerlas del HTML de resultados
        if not metricas:
            try:
                metricas = extraer_metricas_de_html(resultados)
            except Exception:
                metricas = {}
        try:
            session['ultimo_reporte'] = {
                "provincia": prov,
                "distritos": distritos_sel,
                "establecimientos": seleccionados,
                "tiempos": tiempos_int,
                "metricas": metricas or {},
                "mapa_html": mapa_html,
            }
            print("[Diag] Contexto guardado en sesión: keys=", list(session.get('ultimo_reporte', {}).keys()))
        except Exception as e:
            print("[Diag] No se pudo guardar contexto en sesión:", e)

    print("[Diagnóstico] prov=", prov, "dists=", dists, "ests=", seleccionados, "tiempos=", tiempos_int, "=> distritos_sel=", distritos_sel)
    return render_template(
        "diagnostico.html",
        prov_sel=prov,
        dist_sel_codes=dists,
        est_sel=seleccionados,
        distritos_opciones=ALL_DISTRICTS,
        distritos_sel=distritos_sel,
        establecimientos=establecimientos_opts,
        tiempos=tiempos,
        seleccionados=seleccionados,
        tiempos_sel=[str(t) for t in tiempos_int],
        resultados=resultados,
        mapa_html=mapa_html,
        active_page="diagnostico",
    )
from flask import make_response, send_file, url_for, current_app
from xhtml2pdf import pisa
import unicodedata
import shutil
from urllib.parse import urlparse

def _strip_emojis(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    # Reemplazos legibles para los Ã­conos que usamos
    repl = {
        "ðŸ“Š": "Resultados:",
        "ðŸ‘¥": "Personas:",
        "ðŸ—ºï¸": "Ãrea:",
        "ðŸš«": "Sin cobertura:",
    }
    for k, v in repl.items():
        txt = txt.replace(k, v)
    # Quita cualquier otro glifo no BMP que pueda romper reportlab
    txt = unicodedata.normalize("NFKD", txt)
    return txt

def _parse_number(s: str) -> float:
    s2 = s.replace('.', '').replace(',', '.')
    try:
        return float(re.findall(r"[-+]?\d*\.?\d+", s2)[0])
    except Exception:
        return 0.0

def extraer_metricas_de_html(html_texto: str) -> dict:
    try:
        txt = re.sub('<[^<]+?>', ' ', html_texto)
        txt = _strip_emojis(txt)
        # Cobertura única total
        m_total = re.search(r"Cobertura\s+\S*\s+total\D+(\d[\d\.,]*)", txt, re.IGNORECASE)
        pobl_total = int(_parse_number(m_total.group(1))) if m_total else 0
        # En solape
        m_sol = re.search(r"En\s+solape\D+(\d[\d\.,]*)\D+\((\d+[\d\.,]*)%\)", txt, re.IGNORECASE)
        pob_sol = int(_parse_number(m_sol.group(1))) if m_sol else 0
        porc_sol = float(_parse_number(m_sol.group(2))) if m_sol else 0.0
        # Áreas
        m_area_u = re.search(r"\b(Ã?rea|Área)\s+total\s+\S*\s*\(unin|unión\)\D+(\d+[\d\.,]*)\s*km", txt, re.IGNORECASE)
        area_u = float(_parse_number(m_area_u.group(2))) if m_area_u else 0.0
        m_area_s = re.search(r"\b(Ã?rea|Área)\s+con\s+solape\D+(\d+[\d\.,]*)\s*km", txt, re.IGNORECASE)
        area_s = float(_parse_number(m_area_s.group(2))) if m_area_s else 0.0
        return {
            "poblacion_unica": int(pobl_total),
            "poblacion_solape": int(pob_sol),
            "porcentaje_solape": float(porc_sol),
            "area_unica_km2": float(area_u),
            "area_solape_km2": float(area_s),
            "resumen_texto": txt.strip(),
        }
    except Exception:
        return {}

@app.route('/reporte-pdf', methods=['POST'])
def reporte_pdf():
    # Asegurar carpeta de fuentes y copiar DejaVuSans.ttf si se localiza
    try:
        fonts_dir = os.path.join(BASE_DIR, "static", "fonts")
        os.makedirs(fonts_dir, exist_ok=True)
        target_font = os.path.join(fonts_dir, "DejaVuSans.ttf")
        if not os.path.exists(target_font):
            posibles = []
            posibles.append(os.path.join(BASE_DIR, "DejaVuSans.ttf"))
            win = os.environ.get("WINDIR", r"C:\\Windows")
            posibles.append(os.path.join(win, "Fonts", "DejaVuSans.ttf"))
            for src in posibles:
                try:
                    if os.path.exists(src):
                        shutil.copyfile(src, target_font)
                        print(f"[PDF] Copiada fuente desde {src} -> {target_font}")
                        break
                except Exception as _e:
                    pass
            if not os.path.exists(target_font):
                print("[PDF] Advertencia: static/fonts/DejaVuSans.ttf no encontrado. Se usará fuente por defecto.")
    except Exception as e:
        print("[PDF] No se pudo preparar carpeta/fuente:", e)
    # 1) Leer selecciÃ³n del mismo form
    distritos_sel = request.form.getlist("distritos") or DEFAULT_DISTRICTS
    seleccionados = request.form.getlist("establecimientos")
    tiempos_sel   = request.values.getlist("tiempos") or request.values.getlist("tiempos[]") or []
    tiempos_int   = [int(t) for t in tiempos_sel if str(t).isdigit()]
    if not tiempos_int:
        tiempos_int = obtener_tiempos_disponibles()

    # 2) Preparar contenido HTML
    if not seleccionados or not tiempos_int:
        resultados_html = "<b>No hay datos suficientes para el informe.</b><br>Selecciona al menos 1 establecimiento y 1 tiempo."
        mapa_url = None
    else:
        resultados_html, mapa_html = calcular_cobertura_multiple(seleccionados, tiempos_int, distritos_sel)
        mapa_url = url_for('static', filename=mapa_html, _external=True)

    # 2.1 Limpiar emojis/Unicode no soportado para PDF
    resultados_html_pdf = _strip_emojis(resultados_html)
    # 2.2 Renderizar plantilla del informe
    # Construye contexto mínimo para reporte y soporta imagen de mapa si existe
    mapa_img = None
    try:
        if 'mapa_html' in locals() and mapa_html:
            candidate = mapa_html.replace('.html', '.png')
            static_path = os.path.join(BASE_DIR, 'static', candidate.replace('/', os.sep))
            if os.path.exists(static_path):
                mapa_img = candidate
    except Exception:
        mapa_img = None

    ctx = {
        'fecha': datetime.now().strftime('%d/%m/%Y %H:%M'),
        'distritos': distritos_sel,
        'establecimientos': seleccionados,
        'tiempos': tiempos_int,
        'metricas': None,
        'resultados_html': resultados_html_pdf,
        'mapa_html': mapa_html if 'mapa_html' in locals() else None,
        'mapa_img': mapa_img,
    }
    html = render_template('reporte.html', **ctx)

    # 2.2 Renderizar plantilla del informe
    # Override: usar contexto persistido para garantizar consistencia con /diagnostico
    #try:
    # (Función pura) No tocar sesión ni renderizar plantillas aquí.

    # 2.3 Guardar HTML de depuraciÃ³n (abre en el navegador si algo sale mal)
    try:
        debug_dir = os.path.join(BASE_DIR, "static")
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, "_debug_reporte.html")
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[PDF] HTML de depuraciÃ³n en: {debug_path}")
    except Exception as e:
        print("[PDF] No se pudo guardar HTML de depuraciÃ³n:", e)

    # Resolver rutas /static/ para xhtml2pdf
    def _xhtml2pdf_link_callback(uri, rel):
        try:
            # Absolutos HTTP(s): intenta resolver si contienen "/static/"
            if uri.startswith("http://") or uri.startswith("https://"):
                parsed = urlparse(uri)
                path = parsed.path or ""
                if "/static/" in path:
                    local_rel = path[path.index("/static/")+1:]
                    candidate = os.path.join(BASE_DIR, local_rel.replace("/", os.sep))
                    if os.path.exists(candidate):
                        return candidate
                return uri
            # Rutas absolutas del app: "/static/..."
            if uri.startswith("/static/"):
                candidate = os.path.join(BASE_DIR, uri.lstrip("/").replace("/", os.sep))
                return candidate
            # Rutas relativas "static/..."
            if uri.startswith("static/"):
                candidate = os.path.join(BASE_DIR, uri.replace("/", os.sep))
                return candidate
            return uri
        except Exception:
            return uri

    # 3) Render a PDF
    pdf_io = BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=pdf_io, encoding="utf-8", link_callback=_xhtml2pdf_link_callback)
    if pisa_status.err:
        print("[PDF] Error xhtml2pdf:", pisa_status.err)
        return "Error al generar PDF", 500

    pdf_io.seek(0)
    try:
        size = len(pdf_io.getvalue())
        print(f"[PDF] Generado OK, bytes={size}")
    except Exception:
        pass
    return send_file(
        pdf_io,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="reporte_cobertura.pdf",
    )
@app.route('/hexagonos')
def hexagonos():
    nombre_amigable = {
        "mapa_interactivo_DemandaES.html": "DuraciÃ³n de recorrido",
        "isocronas_piura3.html": "Cobertura por isocronas",
        "mapa_interactivo_DemandaES_sierra.html": "DuraciÃ³n de recorrido en sierra",
        "isocronas_piura3_sierra.html": "Cobertura por isocronas en sierra",
    }

    static_dir = os.path.join(BASE_DIR, "static")
    orden_deseado = list(nombre_amigable.keys())

    disponibles = set(os.listdir(static_dir))
    mapas = [f for f in orden_deseado if f in disponibles and f in nombre_amigable]


    return render_template("hexagonos.html", mapas=mapas, nombres=nombre_amigable, active_page="hexagonos")

    print("BASE_DIR:", BASE_DIR)
    print("static_dir:", static_dir)
    print("Existe static_dir:", os.path.exists(static_dir))
    print("Es carpeta:", os.path.isdir(static_dir))

    


@app.route('/')
def inicio():
    return redirect(url_for('situacion_actual'))

@app.route('/api/health')
def health():
    paths = (CFG.get('paths') or {})
    def _fstats(p):
        try:
            exists = bool(p and os.path.exists(p))
            size = os.path.getsize(p) if exists else 0
            return {"path": p, "exists": exists, "size": size}
        except Exception:
            return {"path": p, "exists": False, "size": 0}

    n_dist = None
    n_prov = None
    try:
        if paths.get('distritos') and os.path.exists(paths['distritos']):
            gdf_d = gpd.read_file(paths['distritos'])
            n_dist = int(gdf_d.shape[0])
            n_prov = int(gdf_d['CCPP'].nunique()) if 'CCPP' in gdf_d.columns else None
    except Exception:
        pass

    ors_key_present = bool(os.environ.get('ORS_API_KEY') or os.environ.get('OPENROUTESERVICE_API_KEY'))
    return jsonify({
        'status': 'ok',
        'paths': {
            'distritos': _fstats(paths.get('distritos')),
            'provincias': _fstats(paths.get('provincias')),
            'departamento': _fstats(paths.get('departamento')),
        },
        'counts': {
            'distritos': n_dist,
            'provincias': n_prov,
        },
        'ors_key_present': ors_key_present,
    })

@app.route('/api/establecimientos')
def api_establecimientos():
    try:
        prov = request.args.get('prov')
        dist_codes_param = request.args.get('dist')  # CSV de CCPP+CCDI, ej: 0101,0104
        codes = []
        if dist_codes_param:
            codes = [c.strip() for c in dist_codes_param.split(',') if c.strip()]

        # Determinar distritos filtrados
        df_dist = None
        if distritos_piura is not None:
            df_dist = distritos_piura.copy()
            if prov:
                df_dist = df_dist[df_dist['CCPP'].astype(str).str.zfill(2) == str(prov).zfill(2)]
            if codes:
                df_dist = df_dist[(df_dist['CCPP'].astype(str).str.zfill(2) + df_dist['CCDI'].astype(str).str.zfill(2)).isin(codes)]

        # Filtrar establecimientos por intersecciÃ³n espacial si hay distritos
        g = es_gdf
        if df_dist is not None and not df_dist.empty and 'geometry' in df_dist.columns:
            try:
                joined = gpd.sjoin(g[["est_nombre","geometry"]], df_dist[["geometry"]], how='inner', predicate='intersects')
                es_list = sorted(joined['est_nombre'].astype(str).unique().tolist())
            except Exception:
                es_list = sorted(g['est_nombre'].astype(str).unique().tolist())
        else:
            es_list = sorted(g['est_nombre'].astype(str).unique().tolist())

        out = []
        for name in es_list:
            row = g.loc[g['est_nombre'] == name].iloc[0]
            if 'x_gis' in row and 'y_gis' in row:
                x = float(row['x_gis']); y = float(row['y_gis'])
            else:
                geom = row['geometry']
                x = float(getattr(geom, 'x', 0.0)); y = float(getattr(geom, 'y', 0.0))
            out.append({'est_nombre': str(name), 'x_gis': x, 'y_gis': y})
        print(f"[api_establecimientos] prov={prov} codes={len(codes)} -> {len(out)} ES")
        return jsonify({'establecimientos': out})
    except Exception as e:
        return jsonify({'establecimientos': [], 'error': str(e)}), 500

@app.route('/api/provincias')
def api_provincias():
    try:
        if distritos_piura is None:
            return jsonify({'provincias': []})
        df = distritos_piura
        prov_name_col = None
        for c in df.columns:
            if c.upper() in ('PROVINCIA', 'NOM_PROV', 'PROV'):
                prov_name_col = c
                break
        rows = []
        for ccpp, group in df.groupby('CCPP'):
            nombre = None
            if prov_name_col:
                s = group[prov_name_col].dropna()
                if not s.empty:
                    nombre = str(s.iloc[0])
            rows.append({'ccpp': str(ccpp).zfill(2), 'nombre': nombre, 'n_distritos': int(group.shape[0])})
        rows.sort(key=lambda r: r['ccpp'])
        print(f"[api_provincias] {len(rows)} provincias")
        return jsonify({'provincias': rows})
    except Exception as e:
        return jsonify({'provincias': [], 'error': str(e)}), 500

@app.route('/api/distritos')
def api_distritos():
    try:
        if distritos_piura is None:
            return jsonify({'distritos': []})
        prov = request.args.get('prov')
        df = distritos_piura.copy()
        if prov:
            df = df[df['CCPP'].astype(str).str.zfill(2) == str(prov).zfill(2)]
        name_col = DIST_NAME_COL if (DIST_NAME_COL in df.columns) else None
        if name_col is None:
            for c in df.columns:
                if c.upper() in ('DISTRITO', 'NOM_DIST', 'DIST'):
                    name_col = c
                    break
        rows = []
        for _, r in df.iterrows():
            rows.append({
                'ccpp': str(r.get('CCPP', '')).zfill(2),
                'ccdi': str(r.get('CCDI', '')).zfill(2),
                'nombre': str(r.get(name_col, '')) if name_col else ''
            })
        rows.sort(key=lambda x: (x['ccpp'], x['ccdi']))
        print(f"[api_distritos] prov={prov} -> {len(rows)} distritos")
        return jsonify({'distritos': rows})
    except Exception as e:
        return jsonify({'distritos': [], 'error': str(e)}), 500

def obtener_tiempos_disponibles():
    try:
        if 'tiempo' in gdf_iso.columns:
            vals = []
            for v in gdf_iso['tiempo'].dropna().tolist():
                try:
                    vals.append(int(v))
                except Exception:
                    try:
                        vals.append(int(float(v)))
                    except Exception:
                        pass
            s = sorted({int(x) for x in vals})
            return s if s else tiempos
    except Exception:
        pass
    return tiempos

@app.route('/api/tiempos')
def api_tiempos():
    try:
        ts = obtener_tiempos_disponibles()
        print(f"[api_tiempos] {ts}")
        return jsonify(ts)
    except Exception as e:
        return jsonify([]), 500

@app.route('/api/prov_geo')
def api_prov_geo():
    try:
        if distritos_piura is None:
            return jsonify({'type': 'FeatureCollection', 'features': []})
        prov = request.args.get('ccpp')
        if not prov:
            return jsonify({'type': 'FeatureCollection', 'features': []})
        df = distritos_piura
        mask = df['CCPP'].astype(str).str.zfill(2) == str(prov).zfill(2)
        sub = df.loc[mask]
        if sub.empty:
            return jsonify({'type': 'FeatureCollection', 'features': []})
        try:
            geom = sub.unary_union
        except Exception:
            geom = unary_union(sub.geometry)
        feat = {
            'type': 'Feature',
            'geometry': mapping(geom),
            'properties': {'ccpp': str(prov).zfill(2)}
        }
        return jsonify(feat)
    except Exception as e:
        return jsonify({'type': 'FeatureCollection', 'features': [], 'error': str(e)}), 500

@app.route('/api/dist_geo')
def api_dist_geo():
    try:
        if distritos_piura is None:
            return jsonify({'type': 'FeatureCollection', 'features': []})
        codes_param = request.args.get('codes', '')
        codes = [c.strip() for c in codes_param.split(',') if c.strip()]
        if not codes:
            return jsonify({'type': 'FeatureCollection', 'features': []})
        df = distritos_piura
        series_code = df['CCPP'].astype(str).str.zfill(2) + df['CCDI'].astype(str).str.zfill(2)
        mask = series_code.isin([str(c).zfill(4) for c in codes])
        sub = df.loc[mask]
        name_col = DIST_NAME_COL if (DIST_NAME_COL in sub.columns) else None
        features = []
        for _, r in sub.iterrows():
            nm = str(r.get(name_col, '')) if name_col else ''
            geom = r['geometry']
            features.append({
                'type': 'Feature',
                'geometry': mapping(geom),
                'properties': {
                    'ccpp': str(r.get('CCPP','')).zfill(2),
                    'ccdi': str(r.get('CCDI','')).zfill(2),
                    'nombre': nm
                }
            })
        return jsonify({'type': 'FeatureCollection', 'features': features})
    except Exception as e:
        return jsonify({'type': 'FeatureCollection', 'features': [], 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)





