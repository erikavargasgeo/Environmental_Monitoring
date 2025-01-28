# Código de las tareas del módulo Environmental Monitoring
# Estudiante: Erika Vargas Sanchez  
# UNIGIS ID: 108318

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from shapely.geometry import box

from utils import filter_gdf_by_intersection, merge_landmark_gdfs, reproject_raster, compute_area_percentage_by_class, add_north_arrow

belgium_admin_div_geojson = '../data/leuven/gadm41_BEL_4.json'  # Archivo GeoJSON con los límites de la ciudad en Bélgica
landmarks_geopkg_file = '../data/leuven/poi_patimonialelements_3812.gpkg' # GeoPackage con puntos de interés en Bélgica

# Rutas a los archivos JP2 de Sentinel-2 para las bandas de Infrarrojo Cercano (B08) y Rojo (B04)
nir_paths = {
    2024: "../data/leuven/T31UFS_20240921T104629_B08.jp2",
    2022: "../data/leuven/T31UFS_20220828T104631_B08_10m.jp2",
    2020: "../data/leuven/T31UFS_20200922T104649_B08_10m.jp2",
    2018: "../data/leuven/T31UFS_20180918T105021_B08.jp2"
}

red_paths = {
    2024: "../data/leuven/T31UFS_20240921T104629_B04.jp2",
    2022: "../data/leuven/T31UFS_20220828T104631_B04_10m.jp2",
    2020: "../data/leuven/T31UFS_20200922T104649_B04_10m.jp2",
    2018: "../data/leuven/T31UFS_20180918T105021_B04.jp2"
}

YEAR = 2024

years_to_process = [2018, 2020, 2022, 2024]

DESIRED_CITY = "Leuven"
OFFSET = 0.0  # Definir un desplazamiento para el cuadro delimitador

#######################################
# Obtener límites de la ciudad
#######################################
# Cargar el archivo GeoJSON
gdf_belgium_admin_div = gpd.read_file(belgium_admin_div_geojson)

# Filtrar el GeoDataFrame para geometrías con el atributo NAME_4 igual al valor en DESIRED_CITY. Reproyectar los datos GeoJSON a WGS84
city_boundary_gdf = gdf_belgium_admin_div[gdf_belgium_admin_div['NAME_4'] == DESIRED_CITY].to_crs(epsg=4326)

# Calcular el cuadro delimitador con un desplazamiento
minx, miny, maxx, maxy = city_boundary_gdf.total_bounds
miny -= OFFSET
maxy += OFFSET

# Crear una geometría de cuadro delimitador
bounding_box = box(minx, miny, maxx, maxy)
bounding_box_gdf = gpd.GeoDataFrame({'geometry': [bounding_box]}, crs='EPSG:4326')

#######################################
# Obtener puntos de referencia/elementos patrimoniales
#######################################
# Filtrar todas las capas en el GeoPackage con el CRS EPSG:4326
filtered_landmarks_layers = filter_gdf_by_intersection(city_boundary_gdf, landmarks_geopkg_file, output_crs_epsg=4326)

# Fusionar todas las capas filtradas en un solo GeoDataFrame
landmarks_gdf = merge_landmark_gdfs(filtered_landmarks_layers)

#######################################
# Función para calcular NDVI y porcentajes de área
#######################################
# Definir etiquetas de clase NDVI
ndvi_class_labels = ["Barren", "Low", "Moderate", "High"]

def calculate_ndvi_and_area_percentages(nir_path, red_path, bounding_box_gdf, city_boundary_gdf, ndvi_class_labels):
    # Reproyectar las bandas NIR y Rojo a EPSG:4326
    nir_band, nir_memfile = reproject_raster(nir_path, 'EPSG:4326')
    red_band, red_memfile = reproject_raster(red_path, 'EPSG:4326')

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
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.where((nir + red) == 0., 0, (nir - red) / (nir + red))

    # Enmascarar NDVI con el límite de la ciudad para una mejor visualización
    city_mask = rasterio.features.geometry_mask(
        [feature["geometry"] for feature in city_boundary_gdf.to_crs(epsg=4326).iterfeatures()],
        transform=nir_transform,
        invert=True,
        out_shape=nir.shape,
    )
    ndvi_masked = np.where(city_mask, ndvi, np.nan)

    # Clasificar valores de NDVI
    ndvi_classes = np.digitize(ndvi_masked, bins=[-1, 0.2, 0.4, 0.6, 1])

    # Calcular porcentaje de área por clase NDVI
    area_percentage_by_class = compute_area_percentage_by_class(ndvi_classes, ndvi_class_labels, city_mask)

    return ndvi_masked, nir_transform, ndvi_classes, area_percentage_by_class, nir

#######################################
# Calcular NDVI y porcentajes de área para cada año
#######################################
ndvi_data = {}
for year in years_to_process:
    nir_path = nir_paths[year]
    red_path = red_paths[year]
    ndvi_masked, nir_transform, ndvi_classes, area_percentage_by_class, nir_band = calculate_ndvi_and_area_percentages(nir_path, red_path, bounding_box_gdf, city_boundary_gdf, ndvi_class_labels)
    ndvi_data[year] = {
        'ndvi_masked': ndvi_masked,
        'nir_transform': nir_transform,
        'ndvi_classes': ndvi_classes,
        'area_percentage_by_class': area_percentage_by_class,
        'nir_band': nir_band
    }

# Obtener los datos del año actual
current_year_data = ndvi_data[YEAR]
current_area_percentage_by_class = current_year_data['area_percentage_by_class']

# Calcular variaciones y añadir a ndvi_data
for i in range(1, len(years_to_process)):
    current_year = years_to_process[i]
    previous_year = years_to_process[i - 1]
    current_area_percentage_by_class = ndvi_data[current_year]['area_percentage_by_class']
    previous_area_percentage_by_class = ndvi_data[previous_year]['area_percentage_by_class']

    # Calcular la variación en los puntos críticos de verdor
    barren_variation = current_area_percentage_by_class["Barren"] - previous_area_percentage_by_class["Barren"]
    low_variation = current_area_percentage_by_class["Low"] - previous_area_percentage_by_class["Low"]
    moderate_variation = current_area_percentage_by_class["Moderate"] - previous_area_percentage_by_class["Moderate"]
    high_variation = current_area_percentage_by_class["High"] - previous_area_percentage_by_class["High"]

    # Añadir variaciones a ndvi_data
    ndvi_data[current_year]['barren_variation'] = barren_variation
    ndvi_data[current_year]['low_variation'] = low_variation
    ndvi_data[current_year]['moderate_variation'] = moderate_variation
    ndvi_data[current_year]['high_variation'] = high_variation

if 'barren_variation' in current_year_data:
    variation_text = f"Variation With Respect to {YEAR-2}: \n\n- Barren: {current_year_data['barren_variation']:.2f}%\n- Low: {current_year_data['low_variation']:.2f}%\n- Moderate: {current_year_data['moderate_variation']:.2f}%\n- High: {current_year_data['high_variation']:.2f}%"
else:
    variation_text = f"Variation With Respect to {YEAR-2}: \n\nNo previous data available for comparison."

# Obtener colores para los marcadores de puntos de referencia
cmap = colormaps['Set3']
colors = [cmap(i) for i in range(len(filtered_landmarks_layers))]

# Mapa de visión general: NDVI con puntos de referencia
fig, ax = plt.subplots(figsize=(14, 12))
im = show(current_year_data['ndvi_masked'], ax=ax, transform=current_year_data['nir_transform'], cmap="Greens", title=f"City Greenness Overview in {DESIRED_CITY}, Belgium on 21/09/2024")
for i, (layer_name, gdf) in enumerate(filtered_landmarks_layers.items()):
    gdf.plot(ax=ax, markersize=75, label=layer_name, marker='^', edgecolor='black', alpha=1.0, color=colors[i % len(colors)])
legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Landmarks")


cbar = plt.colorbar(im.get_images()[0], ax=ax, orientation='horizontal', fraction=0.03, pad=0.1)
cbar.set_label('NDVI')

plt.figtext(0.5, 0.03, f"Level of greenness in indicated by NDVI classes:\nBarren (NDVI < 0.2), Low (0.2 < NDVI < 0.4),\nModerate (0.4 < NDVI < 0.6), High (NDVI > 0.6).", ha="center", fontsize=10, weight="bold")

area_text = "Area Percentage by NDVI Class:    \n\n" + "\n".join([f"- {label}: {current_area_percentage_by_class[label]:.2f}%" for label in ndvi_class_labels])
plt.text(1.075, 0.658, area_text + "\n\n" + variation_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.2))

creator_text = f"This map has been created by:     \nErika Vargas on January 20, 2025."
plt.text(1.075, 0.040, creator_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.2))

ax.grid(True, linestyle='--')

ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')

add_north_arrow(ax, '../data/north_arrow.png', position='top right')

plt.savefig(f'./results/city_greenness_overview_{DESIRED_CITY}_{YEAR}.pdf')
plt.show()

#########################################

print(f"\nArea Percentage by NDVI Class in {YEAR}:")
print(pd.DataFrame(current_area_percentage_by_class, index=["Percentage (%)"]).T)

for year in years_to_process:
    print(f"\nArea Percentage by NDVI Class in {year}:")
    print(pd.DataFrame(ndvi_data[year]['area_percentage_by_class'], index=["Percentage (%)"]).T)

# Graficar la variación del porcentaje de área por clase NDVI para 2020, 2022 y 2024.
years = [2020, 2022, 2024]
barren_variations = [ndvi_data[year]['barren_variation'] for year in years]
low_variations = [ndvi_data[year]['low_variation'] for year in years]
moderate_variations = [ndvi_data[year]['moderate_variation'] for year in years]
high_variations = [ndvi_data[year]['high_variation'] for year in years]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 14))

show(current_year_data['ndvi_masked'] > 0.4, ax=ax1, transform=current_year_data['nir_transform'], cmap="Greens", title=f"Greenness Hotspots (NDVI > 0.4) in {DESIRED_CITY}, Belgium on 21/09/2024")
for i, (layer_name, gdf) in enumerate(filtered_landmarks_layers.items()):
    gdf.plot(ax=ax1, markersize=75, label=layer_name, marker='^', edgecolor='black', alpha=1.0, color=colors[i % len(colors)])
ax1.legend(bbox_to_anchor=(0.01, 0.80), loc='upper left', title="Landmarks", ncol=1, framealpha=0.6)

ax1.grid(True, linestyle='--')

ax1.set_xlabel('Longitude (°)')
ax1.set_ylabel('Latitude (°)')

# Añadir flecha del norte
add_north_arrow(ax1, '../data/north_arrow.png', position='top right')

bar_width = 0.2
bar_positions = np.arange(len(years))
bar_positions_low = bar_positions + bar_width
bar_positions_moderate = bar_positions + 2 * bar_width
bar_positions_high = bar_positions + 3 * bar_width

cmap = colormaps['Paired']
colors = [cmap(i) for i in range(len(ndvi_class_labels))]

bar1 = ax2.bar(bar_positions, barren_variations, width=bar_width, label="Barren", color=colors[0], alpha=1.0, zorder=10)
bar2 = ax2.bar(bar_positions_low, low_variations, width=bar_width, label="Low", color=colors[1], alpha=1.0, zorder=10)
bar3 = ax2.bar(bar_positions_moderate, moderate_variations, width=bar_width, label="Moderate", color=colors[2], alpha=1.0, zorder=10)
bar4 = ax2.bar(bar_positions_high, high_variations, width=bar_width, label="High", color=colors[3], alpha=1.0, zorder=10)

ax2.set_xticks(bar_positions + 1.5 * bar_width)
ax2.set_xticklabels([f"{str(year)} vs {str(year-2)}" for year in years])
ax2.grid(axis='y', linestyle='--', zorder=0)
ax2.set_xlabel("Year")
ax2.set_ylabel("Variation (%)")
ax2.set_title("Variation of Area Percentage by NDVI Class With Respect to Previous Two Years", fontweight='bold')
ax2.legend(title="NDVI Class")

plt.savefig(f'./results/greenness_hotspots_and_variation_{DESIRED_CITY}_{YEAR}.pdf')
plt.show()