# Código de las tareas del módulo Environmental Monitoring
# Estudiante: Erika Vargas Sanchez  
# UNIGIS ID: 108318

import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box

from utils import (
    reproject_raster,
    add_north_arrow
)

country_admin_div_geojson = (
    "../data/cali/gadm41_COL_2.json"  # GeoJSON con los límites de la ciudad
)

barrios_de_cali_shp = (
    "../data/cali/mc_barrios/mc_barrios.shp"  # Shapefile con los límites de los barrios
)

# Graficar barrios_de_cali_shp y zona_urbana_cali_gdf
barrios_de_cali_gdf = gpd.read_file(barrios_de_cali_shp).to_crs(epsg=4326)

# Agrupar polígonos por el atributo 'comuna'
comunas_de_cali_gdf = barrios_de_cali_gdf.dissolve(by='comuna')

# Disolver el barrios_de_cali_gdf para crear un solo polígono para toda la ciudad
zona_urbana_cali_gdf = barrios_de_cali_gdf.dissolve()

OFFSET = 0.01  # Offset para el cuadro delimitador

#######################################
# Obtener límites de la ciudad
#######################################
# Cargar el archivo GeoJSON
gdf_country_admin_div = gpd.read_file(country_admin_div_geojson)

# Filtrar el GeoDataFrame para geometrías con el atributo NAME_2
cali_boundary_gdf = gdf_country_admin_div[
    gdf_country_admin_div["NAME_2"] == "SantiagodeCali"
].to_crs(epsg=4326)

# Calcular el cuadro delimitador con un offset
minx, miny, maxx, maxy = cali_boundary_gdf.total_bounds
maxx += OFFSET
maxy += OFFSET

# Crear una geometría de cuadro delimitador
bounding_box = box(minx, miny, maxx, maxy)
bounding_box_gdf = gpd.GeoDataFrame({"geometry": [bounding_box]}, crs="EPSG:4326")

def calculate_ndvi(nir_path, red_path, bounding_box_gdf):
    # Reproyectar las bandas NIR y Rojo a EPSG:4326
    nir_band, nir_memfile = reproject_raster(nir_path, "EPSG:4326")
    red_band, red_memfile = reproject_raster(red_path, "EPSG:4326")

    # Recortar las bandas NIR y Rojo usando el cuadro delimitador
    nir_cropped, nir_transform = mask(nir_band, bounding_box_gdf.geometry, crop=True)
    red_cropped, red_transform = mask(red_band, bounding_box_gdf.geometry, crop=True)

    # Leer las bandas como matrices
    nir = nir_cropped[0].astype(float)
    red = red_cropped[0].astype(float)

    # Cerrar los MemoryFiles
    nir_memfile.close()
    red_memfile.close()

    # Calcular NDVI
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where((nir + red) == 0.0, 0, (nir - red) / (nir + red))

    return ndvi, nir_transform

def compute_greenness_percentage(comunas_gdf, ndvi, nir_transform):
    # Inicializar una lista para almacenar el porcentaje de puntos críticos de verdor para cada comuna
    greenness_percentage_list = []

    # Iterar sobre cada comuna
    for idx, comuna in comunas_gdf.iterrows():
        # Obtener la geometría de la comuna
        comuna_geom = comuna.geometry

        # Enmascarar la matriz NDVI a la geometría de la comuna actual
        comuna_mask = rasterio.features.geometry_mask(
            [comuna_geom],
            transform=nir_transform,
            invert=True,
            out_shape=ndvi.shape,
        )
        ndvi_comuna = np.where(comuna_mask, ndvi, np.nan)

        # Calcular el porcentaje de puntos críticos de verdor (NDVI > 0.4) dentro de la comuna
        greenness_hotspots = np.sum(ndvi_comuna > 0.4)
        total_area = np.sum(~np.isnan(ndvi_comuna))
        greenness_percentage = (greenness_hotspots / total_area) * 100 if total_area > 0 else 0

        # Añadir el porcentaje de verdor a la lista
        greenness_percentage_list.append(greenness_percentage)

    # Añadir la lista de porcentaje de verdor como una nueva columna al GeoDataFrame de comunas
    comunas_gdf['Greenness_Percentage'] = greenness_percentage_list

    return comunas_gdf

# Rutas a los archivos JP2 de Sentinel-2 para las bandas de Infrarrojo Cercano (B08) y Rojo (B04)
nir_path = "../data/cali/20241017/T18NUJ_20241017T153631_B08_10m.jp2"
red_path = "../data/cali/20241017/T18NUJ_20241017T153631_B04_10m.jp2"

# Calcular NDVI para toda el área
ndvi, nir_transform = calculate_ndvi(nir_path, red_path, bounding_box_gdf)

# Calcular el porcentaje de puntos críticos de verdor para cada comuna
comunas_de_cali_gdf = compute_greenness_percentage(comunas_de_cali_gdf, ndvi, nir_transform)

fig, ax = plt.subplots(figsize=(16, 12))

comunas_de_cali_gdf.plot(column='Greenness_Percentage', legend=False, cmap='Greens', ax=ax)
zona_urbana_cali_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=3.0)

ax.set_title(f"Greenness Percentage for Each Comuna in Cali (17/10/2024)", fontsize=16, weight="bold")

# Añadir barra de color vertical
sm = plt.cm.ScalarMappable(cmap='Greens', norm=plt.Normalize(vmin=comunas_de_cali_gdf['Greenness_Percentage'].min(), vmax=comunas_de_cali_gdf['Greenness_Percentage'].max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.03)
cbar.set_label('Greenness Percentage (%)')

# Añadir números de comuna en el centro de cada comuna con color de borde
for idx, row in comunas_de_cali_gdf.iterrows():
    plt.annotate(
        text=idx,
        xy=row['geometry'].centroid.coords[0],
        xytext=(5, -5),
        textcoords='offset points',
        ha='left',
        va='top',
        fontsize=12,
        weight='bold',
        color='white',
        bbox=dict(facecolor='black', edgecolor='white', boxstyle='round,pad=0.3')
    )

creator_text = f"This map has been created by:\nErika Vargas on January 20, 2025."
plt.text(0.995, -0.037, creator_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.2))

ax.grid(True, linestyle='--')
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')

add_north_arrow(ax, '../data/north_arrow.png', position='top right')

plt.savefig(f'./results/greenness_percentage_comunas_cali_2024.pdf')
plt.show()