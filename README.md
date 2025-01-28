# Código de las tareas del módulo Environmental Monitoring

Estudiante: Erika Vargas Sanchez  
UNIGIS ID: 108318

## Fuentes de Datos

Se requieren datos adicionales (rásters) para que los scripts funcionen. Estos incluyen datos de Sentinel-2 y Sentinel-3, y se pueden descargar del Navegador Copernicus del Ecosistema de Espacio de Datos Copernicus.

El directorio `data` debe contener archivos ráster correspondientes al área de Cali, Colombia y Leuven, Bélgica.

Para Leuven, utilicé imágenes de Sentinel-2 (Bandas 03, 04 y 08) con los siguientes identificadores:
- `T31UFS_20180918T105021`
- `T31UFS_20200922T104649`
- `T31UFS_20220828T104631`
- `T31UFS_20240921T104629`

Los archivos `.jp2` de estas imágenes deben descargarse en el directorio `./data/leuven/`.

Para Cali, utilicé datos de Sentinel-2 (archivos `.jp2`) y Sentinel-3 (archivos `.nc`).

De Sentinel-2, utilicé la fecha con identificador `T18NUJ_20200117T153619` (en el directorio `./data/cali/20200117/`) y `T18NUJ_20241017T153631` (en el directorio `./data/cali/20241017/`).

De Sentinel-3, utilicé los datos de Temperatura de Superficie Terrestre (LST) del Producto Terrestre de Nivel 2 de SENTINEL-3 SLSTR para las fechas `202031017` (en el directorio `./data/cali/20231017a/`), y `20241017` (en el directorio `./data/cali/20241017b/`).