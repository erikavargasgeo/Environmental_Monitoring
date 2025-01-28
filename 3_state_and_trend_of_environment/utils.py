# Código de las tareas del módulo Environmental Monitoring
# Estudiante: Erika Vargas Sanchez  
# UNIGIS ID: 108318

import geopandas as gpd
import pandas as pd
import fiona
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def filter_layer_by_intersection(gdf_to_intersect, gdf_layer):
    # Reproyectar gdf_to_intersect al CRS de gdf_layer
    gdf_to_intersect = gdf_to_intersect.to_crs(gdf_layer.crs)
    # Filtrar la capa para incluir solo elementos que intersectan con gdf_to_intersect
    filtered_gdf_layer = gpd.overlay(gdf_layer, gdf_to_intersect, how='intersection')
    # Devolver el GeoDataFrame filtrado si tiene elementos, de lo contrario devolver None
    return filtered_gdf_layer if not filtered_gdf_layer.empty else None

def filter_gdf_by_intersection(gdf_to_intersect, geopkg_file, output_crs_epsg=None):
    filtered_layers = {}
    
    # Listar todas las capas en el GeoPackage
    with fiona.Env():
        layers = fiona.listlayers(geopkg_file)

    prettified_names_lut = {
        'windmill': 'Windmill',
        'watermill': 'Watermill',
        'castle': 'Castle',
        'fortifiedbuilding': 'Fortified Building',
        'tower': 'Tower',
        'shoring': 'Shoring',
        'nonreligiousmonument': 'Non-Religious Monument',
        'monumentalstairs': 'Monumental Stairs',
        'kiosk': 'Kiosk',
        'icehouse': 'Icehouse',
        'historicmound': 'Historic Mound',
        'commemorationbuilding': 'Commemoration Building',
        'othermill': 'Other Mill',
        'otherhistoricbuilding': 'Historic Building'
    }

    # Filtrar cada capa
    for layer in layers:
        gdf_layer = gpd.read_file(geopkg_file, layer=layer)
        
        # Convertir objetos Timestamp a cadenas
        for col in gdf_layer.select_dtypes(include=['datetime64[ns, UTC]', 'datetime64[ns]']).columns:
            gdf_layer[col] = gdf_layer[col].astype(str)
        
        # Filtrar la capa usando la función
        gdf_layer_filtered = filter_layer_by_intersection(gdf_to_intersect, gdf_layer)
        
        # Solo añadir la capa si tiene elementos
        if gdf_layer_filtered is not None:
            # Reproyectar al CRS deseado si se especifica
            if output_crs_epsg is not None:
                gdf_layer_filtered = gdf_layer_filtered.to_crs(epsg=output_crs_epsg)
            filtered_layers[prettified_names_lut[layer]] = gdf_layer_filtered
    
    return filtered_layers

def merge_landmark_gdfs(filtered_layers):
    merged_gdf_list = []
    
    # Fusionar todas las capas filtradas en un solo GeoDataFrame
    for layer_name, gdf in filtered_layers.items():
        gdf['landmark_type'] = layer_name
        merged_gdf_list.append(gdf)
    
    merged_gdf = pd.concat(merged_gdf_list, ignore_index=True)
    
    return gpd.GeoDataFrame(merged_gdf, crs=merged_gdf_list[0].crs)

def reproject_raster(src_path, dst_crs):
    # Reproyectar un archivo raster a un nuevo CRS
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        memfile = rasterio.MemoryFile()
        dst = memfile.open(**kwargs)
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
        return dst, memfile

def compute_area_percentage_by_class(ndvi_classes, class_labels, city_mask):
    # Calcular el porcentaje de área de cada clase NDVI dentro del límite de la ciudad
    class_areas = {}
    total_area_pixels = np.sum(city_mask)  # Número total de píxeles válidos dentro del límite de la ciudad

    for i, label in enumerate(class_labels):
        class_area_pixels = np.sum((ndvi_classes == i + 1) & city_mask)
        class_area_percentage = (class_area_pixels / total_area_pixels) * 100  # Calcular porcentaje
        class_areas[label] = class_area_percentage
    return class_areas

def add_north_arrow(ax, image_path, position='top right'):
    # Añadir una flecha del norte a un gráfico
    arr_img = plt.imread(image_path, format='png')
    imagebox = OffsetImage(arr_img, zoom=0.1)
    if position == 'top right':
        xy = (0.95, 0.95)
    elif position == 'top left':
        xy = (0.05, 0.95)
    ab = AnnotationBbox(imagebox, xy, xycoords='axes fraction', frameon=False)
    ax.add_artist(ab)

def get_raster_statistics(raster):
    # Calcular la media, mediana, mínimo, máximo y desviación estándar de un raster
    if isinstance(raster, tuple):
        raster = raster[0]
    elif isinstance(raster, np.ndarray):
        raster = raster
    elif isinstance(raster, str):
        with rasterio.open(raster) as src:
            raster = src.read(1)

    # Calcular estadísticas excluyendo valores NaN
    mean = np.nanmean(raster)
    median = np.nanmedian(raster)
    min_val = np.nanmin(raster)
    max_val = np.nanmax(raster)
    stddev = np.nanstd(raster)

    return mean, median, min_val, max_val, stddev