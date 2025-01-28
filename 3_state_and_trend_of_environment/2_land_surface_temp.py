# Código de las tareas del módulo Environmental Monitoring
# Estudiante: Erika Vargas Sanchez  
# UNIGIS ID: 108318

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box, mapping
from scipy.interpolate import griddata
import xarray as xr

from utils import (
    add_north_arrow
)

country_admin_div_geojson = (
    "../data/cali/gadm41_COL_2.json"  # Archivo GeoJSON con los límites de la ciudad
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

# Filtrar el GeoDataFrame para geometrías con el atributo NAME_2.
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

def generate_lst_gdf(lst_ds_src, geodetic_ds_src, save_path=None):
    # Abrir el archivo NetCDF de LST
    ds = xr.open_dataset(lst_ds_src)
    LST = ds['LST']

    # Abrir el archivo NetCDF de coordenadas geodésicas
    geodetic_ds = xr.open_dataset(geodetic_ds_src)

    # Extraer latitud y longitud del conjunto de datos geodésicos
    lat = geodetic_ds['latitude_in']
    lon = geodetic_ds['longitude_in']

    # Obtener matriz numpy del conjunto de datos LST. Convertir valores en LST de Kelvin a Celsius
    LST_array = LST.values - 273.15
    lat_array = lat.values
    lon_array = lon.values

    # Crear un GeoDataFrame a partir de LST_array, lat_array y lon_array
    lst_df = pd.DataFrame({'LST': LST_array.flatten(), 'latitude': lat_array.flatten(), 'longitude': lon_array.flatten()}) 
    lst_gdf = gpd.GeoDataFrame(lst_df, geometry=gpd.points_from_xy(lst_df.longitude, lst_df.latitude), crs="EPSG:4326")

    # Enmascarar el GeoDataFrame de LST al cuadro delimitador sin usar filter_gdf_by_intersection.
    lst_gdf = lst_gdf[lst_gdf.within(bounding_box)]

    # Convertir lst_gdf en un ráster usando rasterio

    # Obtener las coordenadas del cuadro delimitador
    minx, miny, maxx, maxy = bounding_box.bounds

    # Obtener el número de filas y columnas
    rows, cols = LST_array.shape

    # Obtener la resolución del ráster
    res = (maxx - minx) / cols

    # Crear un perfil de ráster
    rst_profile = {
        'driver': 'GTiff',
        'height': rows,
        'width': cols,
        'count': 1,
        'dtype': 'float32',
        'crs': 'EPSG:4326',
        'transform': rasterio.Affine.translation(minx, maxy) * rasterio.Affine.scale(res, -res),
        'nodata': -9999
    }

    # Crear un objeto MemoryFile
    memfile = MemoryFile()
    with memfile.open(**rst_profile) as rst:
        # Escribir LST_array en el ráster
        rst.write(LST_array, 1)

        # Enmascarar el ráster al cuadro delimitador
        out_image, out_transform = mask(rst, [mapping(bounding_box)], crop=True)

        # Actualizar el perfil del ráster
        rst_profile.update({
            'height': out_image.shape[1],
            'width': out_image.shape[2],
            'transform': out_transform
        })

        # Crear un nuevo objeto MemoryFile
        memfile2 = MemoryFile()
        with memfile2.open(**rst_profile) as rst2:
            rst2.write(out_image)

            # Obtener la matriz LST enmascarada
            LST_array_masked = rst2.read(1)

    # Crear un GeoDataFrame a partir de la matriz LST enmascarada
    lst_df_masked = pd.DataFrame({'LST': LST_array_masked.flatten(), 'latitude': lat_array.flatten(), 'longitude': lon_array.flatten()})
    lst_gdf_masked = gpd.GeoDataFrame(lst_df_masked, geometry=gpd.points_from_xy(lst_df_masked.longitude, lst_df_masked.latitude), crs="EPSG:4326")

    # Enmascarar el GeoDataFrame de LST a la geometría del límite de Cali
    lst_gdf_masked = lst_gdf_masked[lst_gdf_masked.within(zona_urbana_cali_gdf.union_all())]

    cols = 400
    rows = int(cols * (maxy - miny) / (maxx - minx))

    x = np.linspace(minx, maxx, cols)
    y = np.linspace(miny, maxy, rows)
    xx, yy = np.meshgrid(x, y)

    # Interpolar los valores de LST usando el método del vecino más cercano
    points = np.vstack((lst_gdf_masked.geometry.x, lst_gdf_masked.geometry.y)).T
    values = lst_gdf_masked['LST']
    LST_array_masked_interp = griddata(points, values, (xx, yy), method='nearest')

    # Crear un GeoDataFrame a partir de la matriz LST interpolada
    lst_df_masked_interp = pd.DataFrame({'LST': LST_array_masked_interp.flatten(), 'latitude': yy.flatten(), 'longitude': xx.flatten()})
    lst_gdf_masked_interp = gpd.GeoDataFrame(lst_df_masked_interp, geometry=gpd.points_from_xy(lst_df_masked_interp.longitude, lst_df_masked_interp.latitude), crs="EPSG:4326")

    # Enmascarar el GeoDataFrame de LST a la geometría del límite de Cali
    lst_gdf_masked_interp = lst_gdf_masked_interp[lst_gdf_masked_interp.within(zona_urbana_cali_gdf.union_all())]

    # Guardar el GeoDataFrame en un archivo GeoJSON si se proporciona save_path
    if save_path:
        lst_gdf_masked_interp.to_file(save_path, driver='GeoJSON')

    return lst_gdf_masked_interp

def compute_avg_temp_comuna(comunas_gdf, lst_gdf, year=None):
    # Inicializar una lista para almacenar la LST promedio para cada comuna
    avg_lst_list = []

    # Iterar sobre cada comuna
    for idx, comuna in comunas_gdf.iterrows():
        # Obtener la geometría de la comuna
        comuna_geom = comuna.geometry

        # Enmascarar el GeoDataFrame de LST a la geometría de la comuna actual
        lst_within_comuna = lst_gdf[lst_gdf.within(comuna_geom)]

        # Calcular la LST promedio dentro de la comuna
        avg_lst = lst_within_comuna['LST'].mean()

        # Añadir la LST promedio a la lista
        avg_lst_list.append(avg_lst)

    # Añadir la lista de LST promedio como una nueva columna al GeoDataFrame de comunas
    if year:
        column_name = f'AVG_LST_{year}'
        comunas_gdf[column_name] = avg_lst_list
    else:
        comunas_gdf['AVG_LST'] = avg_lst_list

    return comunas_gdf

def compute_trend(comunas_gdf, lst_gdf_1, lst_gdf_2):
    # Inicializar una lista para almacenar la tendencia de los cambios de LST para cada comuna
    trend_lst_list = []

    # Iterar sobre cada comuna
    for idx, comuna in comunas_gdf.iterrows():
        # Obtener la geometría de la comuna
        comuna_geom = comuna.geometry

        # Enmascarar los GeoDataFrames de LST a la geometría de la comuna actual
        lst_within_comuna_1 = lst_gdf_1[lst_gdf_1.within(comuna_geom)]
        lst_within_comuna_2 = lst_gdf_2[lst_gdf_2.within(comuna_geom)]

        # Calcular la LST promedio dentro de la comuna para cada fecha
        avg_lst_1 = lst_within_comuna_1['LST'].mean()
        avg_lst_2 = lst_within_comuna_2['LST'].mean()

        # Calcular la tendencia de los cambios de LST (diferencia entre las dos fechas)
        trend_lst = avg_lst_2 - avg_lst_1

        # Añadir la tendencia de los cambios de LST a la lista
        trend_lst_list.append(trend_lst)

    # Añadir la lista de tendencias de cambios de LST como una nueva columna al GeoDataFrame de comunas
    comunas_gdf['LST_Trend'] = trend_lst_list

    return comunas_gdf

date_1 = '20241017b'
date_2 = '20231017a'

lst_gdf_masked_interp_1 = generate_lst_gdf(
    lst_ds_src=f'../data/cali/{date_1}/LST_in.nc',
    geodetic_ds_src=f'../data/cali/{date_1}/geodetic_in.nc',
    save_path=f'../data/cali/{date_1}/LST_interp_cali.geojson'
)

lst_gdf_masked_interp_2 = generate_lst_gdf(
    lst_ds_src=f'../data/cali/{date_2}/LST_in.nc',
    geodetic_ds_src=f'../data/cali/{date_2}/geodetic_in.nc',
    save_path=f'../data/cali/{date_2}/LST_interp_cali.geojson'
)

comunas_de_cali_gdf = compute_avg_temp_comuna(comunas_de_cali_gdf, lst_gdf_masked_interp_1, year = 2024)
comunas_de_cali_gdf = compute_avg_temp_comuna(comunas_de_cali_gdf, lst_gdf_masked_interp_2, year = 2023)

comunas_de_cali_gdf = compute_trend(comunas_de_cali_gdf, lst_gdf_masked_interp_1, lst_gdf_masked_interp_2)

# Obtener comunas con tendencias de LST positivas y negativas
positive_trend_comunas = comunas_de_cali_gdf[comunas_de_cali_gdf['LST_Trend'] > 0]
negative_trend_comunas = comunas_de_cali_gdf[comunas_de_cali_gdf['LST_Trend'] < 0]

# Imprimir la comuna con la LST promedio más alta
max_avg_lst_comuna_2024 = comunas_de_cali_gdf[comunas_de_cali_gdf['AVG_LST_2024'] == comunas_de_cali_gdf['AVG_LST_2024'].max()]
print(f'Comuna with the highest average LST in 2024: {max_avg_lst_comuna_2024.index[0]}')

max_avg_lst_comuna_2023 = comunas_de_cali_gdf[comunas_de_cali_gdf['AVG_LST_2023'] == comunas_de_cali_gdf['AVG_LST_2023'].max()]
print(f'Comuna with the highest average LST in 2023: {max_avg_lst_comuna_2023.index[0]}')


# Graficar la tendencia de los cambios de LST para cada comuna
fig, ax = plt.subplots(figsize=(10, 10))
comunas_de_cali_gdf.plot(column='LST_Trend', legend=True, cmap='coolwarm', ax=ax)
zona_urbana_cali_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=3.0)
plt.show()


###############################################

# Graficar la LST promedio para cada comuna
fig, ax = plt.subplots(figsize=(16, 12))

comunas_de_cali_gdf.plot(column='AVG_LST_2024', legend=False, cmap='jet', ax=ax)
zona_urbana_cali_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=3.0)

ax.set_title(f"Average Land Surface Temperature (LST)\nfor Each Comuna in Cali (17/10/2024)", fontsize=16, weight="bold")

# Añadir barra de color vertical
sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=comunas_de_cali_gdf['AVG_LST_2024'].min(), vmax=comunas_de_cali_gdf['AVG_LST_2024'].max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.03)
cbar.set_label('Average LST (°C)')

# Añadir números de comuna en el centro de cada comuna
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

plt.savefig(f'./results/average_lst_comunas_cali_2024.pdf')
plt.show()
