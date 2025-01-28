# Código de las tareas del módulo Environmental Monitoring
# Estudiante: Erika Vargas Sanchez  
# UNIGIS ID: 108318

import geopandas as gpd
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.features import shapes
from rasterio.io import MemoryFile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patches as mpatches
from shapely.geometry import box, mapping

from utils import (
    reproject_raster,
    add_north_arrow,
    get_raster_statistics
)

country_admin_div_geojson = (
    "../data/cali/gadm41_COL_2.json"  # GeoJSON con los límites de la ciudad
)
cali_puerto_mallarino_area_geojson = "../data/cali/area_puerto_mallarino.geojson"  # GeoJSON con el área de Puerto Mallarino
puerto_mallarino_polygon_geojson = "../data/cali/puerto_mallarino_polygon.geojson"  # GeoJSON con el polígono de Puerto Mallarino
reservorios_polygons_geojson = "../data/cali/reservorios_pto_mallarino.geojson"  # GeoJSON con los polígonos de reservorios
humedal_charco_azul_geojson = "../data/cali/humedal_charco_azul.geojson"  # GeoJSON con el polígono de Humedal Charco Azul
points_of_interest_geojson = "../data/cali/poi_cali.geojson"  # GeoJSON con puntos de interés

# Rutas a los archivos JP2 de Sentinel-2 para las bandas de Infrarrojo Cercano (B08) y Rojo (B04)
nir_paths = {
    '2024': "../data/cali/20241017/T18NUJ_20241017T153631_B08_10m.jp2",
    '2020': "../data/cali/20200117/T18NUJ_20200117T153619_B08_10m.jp2"
}

red_paths = {
    '2024': "../data/cali/20241017/T18NUJ_20241017T153631_B04_10m.jp2",
    '2020': "../data/cali/20200117/T18NUJ_20200117T153619_B04_10m.jp2"
}

green_paths = {
    '2024': "../data/cali/20241017/T18NUJ_20241017T153631_B03_10m.jp2",
    '2020': "../data/cali/20200117/T18NUJ_20200117T153619_B03_10m.jp2"
}

true_color_paths = {
    '2024': "../data/cali/20241017/T18NUJ_20241017T153631_TCI_10m.jp2",
    '2020': "../data/cali/20200117/T18NUJ_20200117T153619_TCI_10m.jp2"
}

YEAR = '2024'

years_to_process = [YEAR]

DESIRED_CITY = "Cali"
OFFSET = 0.0  # Definir un offset para el cuadro de limites

#######################################
# Obtener límites de la ciudad
#######################################
gdf_country_admin_div = gpd.read_file(country_admin_div_geojson)
gdf_cali_puerto_mallarino_area = gpd.read_file(cali_puerto_mallarino_area_geojson)
gdf_puerto_mallarino_polygon = gpd.read_file(puerto_mallarino_polygon_geojson)
gdf_reservorios_polygons = gpd.read_file(reservorios_polygons_geojson)
gdf_humedal_charco_azul = gpd.read_file(humedal_charco_azul_geojson)
gdf_points_of_interest = gpd.read_file(points_of_interest_geojson)

# Límites de Puerto Mallarino
cali_puerto_mallarino_area_gdf = gdf_cali_puerto_mallarino_area.to_crs(epsg=4326)

# Polígono de Puerto Mallarino
puerto_mallarino_polygon_gdf = gdf_puerto_mallarino_polygon.to_crs(epsg=4326)

# Polígonos de reservorios
reservorios_polygons_gdf = gdf_reservorios_polygons.to_crs(epsg=4326)

# Polígono de Humedal Charco Azul
humedal_charco_azul_gdf = gdf_humedal_charco_azul.to_crs(epsg=4326)

# Puntos de interés
points_of_interest_gdf = gdf_points_of_interest.to_crs(epsg=4326)

# Calcular el cuadro delimitador con un offset
minx, miny, maxx, maxy = cali_puerto_mallarino_area_gdf.total_bounds
maxx += OFFSET
maxy += OFFSET

# Crear una geometría de cuadro delimitador
bounding_box = box(minx, miny, maxx, maxy)
bounding_box_gdf = gpd.GeoDataFrame({"geometry": [bounding_box]}, crs="EPSG:4326")

#######################################
# Función para calcular NDVI y porcentajes de área
#######################################
def generate_indices(
    nir_path, red_path, green_path, true_color_path, bounding_box_gdf
):
    # Reproyectar las bandas NIR y Rojo a EPSG:4326
    nir_band, nir_memfile = reproject_raster(nir_path, "EPSG:4326")
    red_band, red_memfile = reproject_raster(red_path, "EPSG:4326")
    green_band, green_memfile = reproject_raster(green_path, "EPSG:4326")
    tci_band, tci_memfile = reproject_raster(true_color_path, "EPSG:4326")

    # Recortar las bandas NIR y Rojo usando el cuadro delimitador
    nir_cropped, nir_transform = mask(nir_band, bounding_box_gdf.geometry, crop=True)
    red_cropped, red_transform = mask(red_band, bounding_box_gdf.geometry, crop=True)
    green_cropped, green_transform = mask(
        green_band, bounding_box_gdf.geometry, crop=True
    )
    tci_cropped, tci_transform = mask(
        tci_band, bounding_box_gdf.geometry, crop=True
    )

    # Leer las bandas como matrices
    nir = nir_cropped[0].astype(float)
    red = red_cropped[0].astype(float)
    green = green_cropped[0].astype(float)
    tci = tci_cropped.transpose(1, 2, 0)

    # Cerrar los MemoryFiles
    nir_memfile.close()
    red_memfile.close()
    green_memfile.close()
    tci_memfile.close()

    # Calcular NDVI
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where((nir + red) == 0.0, 0, (nir - red) / (nir + red))

    # Calcular NDWI
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = np.where((green + nir) == 0.0, 0, (green - nir) / (green + nir))

    # Calcular NDTI
    with np.errstate(divide="ignore", invalid="ignore"):
        ndti = np.where((green + red) == 0.0, 0, (red - green) / (red + green))

    # Enmascarar NDVI con el límite de la ciudad
    city_mask = rasterio.features.geometry_mask(
        [
            feature["geometry"]
            for feature in bounding_box_gdf.to_crs(epsg=4326).iterfeatures()
        ],
        transform=nir_transform,
        invert=True,
        out_shape=nir.shape,
    )
    ndvi_masked = np.where(city_mask, ndvi, np.nan)
    ndwi_masked = np.where(city_mask, ndwi, np.nan)
    ndti_masked = np.where(city_mask, ndti, np.nan)

    # Generar máscara de agua basada en los valores de la banda NDWI (umbral = 0.03).
    water_mask = np.where(ndwi_masked > 0.03, 1, 0)

    # Convertir raster binario a polígonos
    water_mask_int = water_mask.astype(np.uint8)
    results = (
        {"properties": {"gridcode": v}, "geometry": s}
        for i, (s, v) in enumerate(shapes(water_mask_int, transform=nir_transform))
    )
    geoms = list(results)
    gdf = gpd.GeoDataFrame.from_features(geoms, crs="EPSG:4326")

    # Extraer polígonos de agua
    water_polygons = gdf[gdf["gridcode"] == 1]

    # Disolver polígonos de agua para crear una única máscara de cuerpo de agua
    dissolved_water_polygons = water_polygons.dissolve(by="gridcode")

    # Crear máscara de geometría rasterio usando los polígonos de agua disueltos
    water_mask = rasterio.features.geometry_mask(
        [
            mapping(shape)
            for shape in dissolved_water_polygons.geometry.to_crs("EPSG:4326")
        ],
        transform=nir_transform,
        invert=True,
        out_shape=nir.shape,
    )

    # Recortar el raster NDTI a la máscara de agua
    ndti_clipped = np.where(water_mask, ndti_masked, np.nan)

    # Definir un mapa de colores personalizado basado en el mapa de colores jet con más detalle entre -0.3 y 0.3
    colors = [
        (0, 0, 1),
        (0, 0.5, 1), 
        (0, 1, 1),  
        (0, 1, 0.5), 
        (0, 1, 0),  
        (0.5, 1, 0),  
        (1, 1, 0), 
        (1, 0.5, 0),
        (1, 0, 0),
    ]

    # Definir las posiciones de los colores en el mapa de colores
    positions = [0, 0.40, 0.45, 0.47, 0.5, 0.53, 0.55, 0.60, 1]

    # Crear el mapa de colores personalizado
    custom_cmap = LinearSegmentedColormap.from_list("custom_jet", list(zip(positions, colors)), N=256)
    
    # Crear objetos raster georreferenciados
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=tci.shape[0],
            width=tci.shape[1],
            count=3,
            dtype=tci.dtype,
            crs="EPSG:4326",
            transform=tci_transform,
        ) as dataset:
            dataset.write(tci.transpose(2, 0, 1))
            tci_georef = dataset.read()

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=ndti_clipped.shape[0],
            width=ndti_clipped.shape[1],
            count=1,
            dtype=ndti_clipped.dtype,
            crs="EPSG:4326",
            transform=nir_transform,
        ) as dataset:
            dataset.write(ndti_clipped, 1) 
            ndti_clipped_georef = dataset.read(1)

    rasters = {
        "nir": nir,
        "red": red,
        "green": green,
        "tci": tci_georef,
        "ndvi": ndvi,
        "ndwi": ndwi,
        "ndti": ndti,
        "ndvi_masked": ndvi_masked,
        "ndwi_masked": ndwi_masked,
        "ndti_masked": ndti_masked,
        "water_mask": water_mask,
        "water_polygons": water_polygons,
        "dissolved_water_polygons": dissolved_water_polygons,
        "ndti_clipped": ndti_clipped,
        "ndti_clipped_georef": ndti_clipped_georef
    }

    return rasters, nir_transform, custom_cmap


rasters_per_year = {}
for year in years_to_process:
    rasters, nir_transform, custom_cmap = generate_indices(
        nir_paths[year],
        red_paths[year],
        green_paths[year],
        true_color_paths[year],
        bounding_box_gdf
    )
    rasters_per_year[year] = rasters

# Enmascarar el raster NDTI a los polígonos de reservorios
reservorios_mask = rasterio.features.geometry_mask(
    [mapping(geom) for geom in reservorios_polygons_gdf.geometry],
    transform=nir_transform,
    invert=True,
    out_shape=rasters_per_year[YEAR]["ndti_clipped"].shape,
)
ndti_reservorios = np.where(reservorios_mask, rasters_per_year[YEAR]["ndti_clipped"], np.nan)

# Enmascarar el raster NDTI al polígono de Humedal Charco Azul
humedal_charco_azul_mask = rasterio.features.geometry_mask(
    [mapping(humedal_charco_azul_gdf.geometry[0])],
    transform=nir_transform,
    invert=True,
    out_shape=rasters_per_year[YEAR]["ndti_clipped"].shape,
)
ndti_humedal_charco_azul = np.where(humedal_charco_azul_mask, rasters_per_year[YEAR]["ndti_clipped"], np.nan)

# Enmascarar el raster NDTI al área fuera de los polígonos de reservorios (es decir, el río Cauca)
ndti_cauca_river = np.where(~reservorios_mask, rasters_per_year[YEAR]["ndti_clipped"], np.nan)
# Enmascarar el raster NDTI al área fuera del polígono de Humedal Charco Azul (es decir, el río Cauca)
ndti_cauca_river = np.where(~humedal_charco_azul_mask, ndti_cauca_river, np.nan)

# Calcular estadísticas para el NDTI dentro de los reservorios
mean_ndti_reservorios, median_ndti_reservorios, min_ndti_reservorios, max_ndti_reservorios, stddev_ndti_reservorios = get_raster_statistics(ndti_reservorios)

# Calcular estadísticas para el NDTI dentro del río Cauca
mean_ndti_cauca_river, median_ndti_cauca_river, min_ndti_cauca_river, max_ndti_cauca_river, stddev_ndti_cauca_river = get_raster_statistics(ndti_cauca_river)



fig, ax = plt.subplots(figsize=(16, 12))

# Graficar los rásteres georreferenciados
show(rasters_per_year[YEAR]["tci"], ax=ax, transform=nir_transform, alpha=0.3)
show(rasters_per_year[YEAR]["ndti_clipped_georef"], cmap=custom_cmap, ax=ax, transform=nir_transform, alpha=1.0, vmin=-1, vmax=1)

ax.set_title(f"Water Turbidity of Cauca River (Using NDTI) in the Vicinity of\nthe Puerto Mallarino Water Purification Plan (17/10/2024)", fontsize=16, weight="bold")
ax.axis("off")

norm = Normalize(vmin=-1, vmax=1)
sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])

cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.03, pad=0.03, anchor=(0.15, 0.0))
cbar.set_label('NDTI Value')

plt.figtext(0.25, 0.03, "Normalized Difference Turbidity Index (NDTI) values:\n-1 (Low Turbidity) to 1 (High Turbidity)", ha="left", fontsize=10, weight="bold")

creator_text = f"This map has been created by:     \nErika Vargas on January 20, 2025."
plt.text(0.995, -0.037, creator_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.2))

ax.grid(True, linestyle='--')

ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')

add_north_arrow(ax, '../data/north_arrow.png', position='top right')

puerto_mallarino_polygon_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.0, linestyle='--', label='Puerto Mallarino Water Purification Plant')
humedal_charco_azul_gdf.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=1.0, linestyle=':', label='Humedal Charco Azul')
points_of_interest_gdf.plot(ax=ax, marker='^', edgecolor='black', color='magenta', markersize=70, label='Juanchito Bridge')
legend_puerto_mallarino = mpatches.Patch(facecolor='none', edgecolor='black', linewidth=1.0, linestyle='--', label='Puerto Mallarino Water Purification Plant')
legend_humedal_charco_azul = mpatches.Patch(facecolor='none', edgecolor='blue', linewidth=1.0, linestyle=':', label='Humedal Charco Azul')
legend_points_of_interest = mpatches.Patch(facecolor='magenta', edgecolor='black', label='Juanchito Bridge')
ax.legend(handles=[legend_puerto_mallarino, legend_humedal_charco_azul, legend_points_of_interest], loc='upper left', title="Points of Interest", title_fontsize='large')

plt.savefig(f'./results/turbidity_overview_{DESIRED_CITY}_{YEAR}.pdf')
plt.show()

###########################################################################

# Datos para visualización
categories = ['Mean', 'Median', 'Min', 'Max', 'Stddev']
reservorios_stats = [mean_ndti_reservorios, median_ndti_reservorios, min_ndti_reservorios, max_ndti_reservorios, stddev_ndti_reservorios]
cauca_river_stats = [mean_ndti_cauca_river, median_ndti_cauca_river, min_ndti_cauca_river, max_ndti_cauca_river, stddev_ndti_cauca_river]


# Generar rásteres NDTI para los años 2024 y 2020
rasters_2024, nir_transform_2024, custom_cmap_2024 = generate_indices(
    nir_paths['2024'],
    red_paths['2024'],
    green_paths['2024'],
    true_color_paths['2024'],
    bounding_box_gdf
)

rasters_2020, nir_transform_2020, custom_cmap_2020 = generate_indices(
    nir_paths['2020'],
    red_paths['2020'],
    green_paths['2020'],
    true_color_paths['2020'],
    bounding_box_gdf
)

# Enmascarar el ráster NDTI a los polígonos de reservorios para ambos años
reservorios_mask_2024 = rasterio.features.geometry_mask(
    [mapping(geom) for geom in reservorios_polygons_gdf.geometry],
    transform=nir_transform_2024,
    invert=True,
    out_shape=rasters_2024["ndti_clipped"].shape,
)
ndti_reservorios_2024 = np.where(reservorios_mask_2024, rasters_2024["ndti_clipped"], np.nan)

reservorios_mask_2020 = rasterio.features.geometry_mask(
    [mapping(geom) for geom in reservorios_polygons_gdf.geometry],
    transform=nir_transform_2020,
    invert=True,
    out_shape=rasters_2020["ndti_clipped"].shape,
)
ndti_reservorios_2020 = np.where(reservorios_mask_2020, rasters_2020["ndti_clipped"], np.nan)

# Enmascarar el ráster NDTI al área fuera de los polígonos de reservorios (es decir, el río Cauca) para ambos años
ndti_cauca_river_2024 = np.where(~reservorios_mask_2024, rasters_2024["ndti_clipped"], np.nan)
ndti_cauca_river_2020 = np.where(~reservorios_mask_2020, rasters_2020["ndti_clipped"], np.nan)

# Calcular estadísticas para el NDTI dentro de los reservorios para ambos años
mean_ndti_reservorios_2024, median_ndti_reservorios_2024, min_ndti_reservorios_2024, max_ndti_reservorios_2024, stddev_ndti_reservorios_2024 = get_raster_statistics(ndti_reservorios_2024)
mean_ndti_reservorios_2020, median_ndti_reservorios_2020, min_ndti_reservorios_2020, max_ndti_reservorios_2020, stddev_ndti_reservorios_2020 = get_raster_statistics(ndti_reservorios_2020)

# Calcular estadísticas para el NDTI dentro del río Cauca para ambos años
mean_ndti_cauca_river_2024, median_ndti_cauca_river_2024, min_ndti_cauca_river_2024, max_ndti_cauca_river_2024, stddev_ndti_cauca_river_2024 = get_raster_statistics(ndti_cauca_river_2024)
mean_ndti_cauca_river_2020, median_ndti_cauca_river_2020, min_ndti_cauca_river_2020, max_ndti_cauca_river_2020, stddev_ndti_cauca_river_2020 = get_raster_statistics(ndti_cauca_river_2020)

# Calcular la tendencia (diferencia) en las estadísticas entre los dos años
trend_mean_reservorios = mean_ndti_reservorios_2024 - mean_ndti_reservorios_2020
trend_median_reservorios = median_ndti_reservorios_2024 - median_ndti_reservorios_2020
trend_min_reservorios = min_ndti_reservorios_2024 - min_ndti_reservorios_2020
trend_max_reservorios = max_ndti_reservorios_2024 - max_ndti_reservorios_2020
trend_stddev_reservorios = stddev_ndti_reservorios_2024 - stddev_ndti_reservorios_2020

trend_mean_cauca_river = mean_ndti_cauca_river_2024 - mean_ndti_cauca_river_2020
trend_median_cauca_river = median_ndti_cauca_river_2024 - median_ndti_cauca_river_2020
trend_min_cauca_river = min_ndti_cauca_river_2024 - min_ndti_cauca_river_2020
trend_max_cauca_river = max_ndti_cauca_river_2024 - max_ndti_cauca_river_2020
trend_stddev_cauca_river = stddev_ndti_cauca_river_2024 - stddev_ndti_cauca_river_2020

# Datos para visualización
categories = ['Mean', 'Median', 'Min', 'Max', 'Stddev']
reservorios_stats_2024 = [mean_ndti_reservorios_2024, median_ndti_reservorios_2024, min_ndti_reservorios_2024, max_ndti_reservorios_2024, stddev_ndti_reservorios_2024]
reservorios_stats_2020 = [mean_ndti_reservorios_2020, median_ndti_reservorios_2020, min_ndti_reservorios_2020, max_ndti_reservorios_2020, stddev_ndti_reservorios_2020]
cauca_river_stats_2024 = [mean_ndti_cauca_river_2024, median_ndti_cauca_river_2024, min_ndti_cauca_river_2024, max_ndti_cauca_river_2024, stddev_ndti_cauca_river_2024]
cauca_river_stats_2020 = [mean_ndti_cauca_river_2020, median_ndti_cauca_river_2020, min_ndti_cauca_river_2020, max_ndti_cauca_river_2020, stddev_ndti_cauca_river_2020]


# Crear diagramas de caja combinados para comparar los valores de NDTI para reservorios y el río Cauca para los años 2024 y 2020

# Preparar datos para diagramas de caja
data_reservorios = [ndti_reservorios_2024[~np.isnan(ndti_reservorios_2024)], ndti_reservorios_2020[~np.isnan(ndti_reservorios_2020)]]
data_cauca_river = [ndti_cauca_river_2024[~np.isnan(ndti_cauca_river_2024)], ndti_cauca_river_2020[~np.isnan(ndti_cauca_river_2020)]]

# Combinar datos para diagramas de caja
data_combined = data_reservorios + data_cauca_river
labels_combined = ['Reservoirs [2024]', 'Reservoirs [2020]', 'Cauca River [2024]', 'Cauca River [2020]']

# Crear un diagrama de caja combinado
fig, ax = plt.subplots(figsize=(12, 8))
ax.boxplot(data_combined, labels=labels_combined)
ax.set_xlabel('Category [Year]')
ax.set_ylabel('NDTI Value')
ax.set_title('Boxplot of NDTI Values for Reservoirs and Cauca River (17/10/2024 vs 17/01/2020)', fontsize=16, weight="bold")

# Guardar la figura
plt.savefig(f'./results/ndti_combined_boxplot_{DESIRED_CITY}_2024_vs_2020.pdf')
plt.show()