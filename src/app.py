## Paso 1: Instalar librerías

import os
from bs4 import BeautifulSoup
import requests
import time
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

## Paso 2: Descargar HTML

# URL de la página que es objeto del web scrapping
url = 'https://companies-market-cap-copy.vercel.app/index.html'

# Hacer la solicitud a la página
response = requests.get(url)

# Verificar que la solicitud fue exitosa
if response.status_code == 200:
    html_content = response.text # Guardar el HTML
else:
    print(f'Error al acceder a la página: {response.status_code}')

## Paso 3: Transforma el HTML

# Convertir el HTML en un objeto estructurado con BeautifulSoup

soup = BeautifulSoup(html_content, 'html.parser')  # Parsear el HTML con BeautifulSoup

# Buscar todas las tablas en el HTML

tables = soup.find_all('table')

print(f"Se encontraron {len(tables)} tablas en la página.")

# Si encontramos tablas, analizamos cuál contiene la evolución anual
for i, table in enumerate(tables):
    print(f"\nTabla {i + 1}:")
    print(table.prettify()[:500])  # Muestra los primeros 500 caracteres de cada tabla para análisis

# Seleccionar la Tabla 1
tabla_evolucion = tables[0]  # La primera tabla es la de evolución anual

# Extraer las filas de la tabla
filas = tabla_evolucion.find_all('tr')

# Crear una lista para almacenar los datos
data = []

# Iterar sobre las filas de la tabla
for fila in filas[1:]:  # Omitimos la primera fila (encabezados)
    columnas = fila.find_all('td')
    if len(columnas) >= 3:  # Asegurar que hay al menos 3 columnas
        year = columnas[0].text.strip()
        revenue = columnas[1].text.strip()
        change = columnas[2].text.strip()
        data.append([year, revenue, change])

# Crear un DataFrame de pandas con los datos extraídos
df = pd.DataFrame(data, columns=['Año', 'Ingresos', 'Cambio'])

# Ordenar los datos por la columna "Año" de menor a mayor 
df = df.sort_values("Año")

# Mostrar la tabla de datos
df

## Paso 4: Procesa el DataFrame

# Eliminar filas vacías
df = df.dropna()

# Remover '$' y 'B' de la columna "Ingresos" y convertir a número
df['Ingresos'] = df['Ingresos'].replace({'\$': '', ' B': ''}, regex=True).astype(float)

# Procesar la columna "Cambio" para eliminar valores vacíos y convertir en porcentaje decimal
df['Cambio'] = df['Cambio'].replace({'%': ''}, regex=True)
df['Cambio'] = pd.to_numeric(df['Cambio'], errors='coerce') / 100  # Convierte a float, ignorando valores no numéricos

# Eliminar filas donde "Cambio" sea NaN (valores vacíos que no se pudieron convertir)
df = df.dropna(subset=['Cambio'])

# Mostrar el DataFrame limpio
df

## Paso 5: Almacena los datos en sqlite

# Crear (o conectar a) una base de datos SQLite
conn = sqlite3.connect('tesla_financials.db')

# Crear un cursor para ejecutar comandos SQL
cursor = conn.cursor()

# Crear una tabla para almacenar los datos (si no existe)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS financials (
        Año INTEGER PRIMARY KEY,
        Ingresos REAL,
        Cambio REAL
    )
''')

# Insertar los datos en la tabla
df.to_sql('financials', conn, if_exists='replace', index=False)

# Guardar (commit) los cambios en la base de datos
conn.commit()

# Cerrar la conexión
conn.close()

# Confirmación
"Datos almacenados exitosamente en SQLite"

## Paso 6: Visualiza los datos

# Conectar a la base de datos y recuperar los datos
conn = sqlite3.connect('tesla_financials.db')
df = pd.read_sql('SELECT * FROM financials', conn)
conn.close()

# Crear un gráfico de líneas: Evolución de ingresos a lo largo del tiempo
plt.figure(figsize=(10,5))
plt.plot(df['Año'], df['Ingresos'], marker='o', linestyle='-', label='Ingresos (en B USD)')
plt.xlabel('Año')
plt.ylabel('Ingresos (Billones USD)')
plt.title('Evolución de los Ingresos de Tesla')
plt.legend()
plt.grid()
plt.show()

# Crear un gráfico de barras: Comparación de ingresos por año
plt.figure(figsize=(10,5))
plt.bar(df['Año'].astype(str), df['Ingresos'], color='royalblue', alpha=0.7)
plt.xlabel('Año')
plt.ylabel('Ingresos (Billones USD)')
plt.title('Ingresos anuales de Tesla')
plt.grid(axis='y')
plt.show()

# Crear un gráfico de dispersión: Cambio porcentual de ingresos vs Año

plt.figure(figsize=(10,5))
plt.scatter(df['Año'], df['Cambio'], color='red', alpha=0.7, label='Cambio (%)')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%')) # Convertir los valores del eje Y a formato porcentaje
plt.xlabel('Año')
plt.ylabel('Cambio (%)')
plt.title('Cambio porcentual anual en ingresos de Tesla')
plt.legend()
plt.grid()
plt.show()

##########################################################################################################################################################

## Ejercicio extra

import os
from bs4 import BeautifulSoup
import requests
import time
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# URL de la página que es objeto del web scrapping
url = 'https://companies-market-cap-copy.vercel.app/earnings.html'

# Hacer la solicitud a la página
response = requests.get(url)

# Verificar que la solicitud fue exitosa
if response.status_code == 200:
    html_content = response.text # Guardar el HTML
else:
    print(f'Error al acceder a la página: {response.status_code}')

# Convertir el HTML en un objeto estructurado con BeautifulSoup

soup = BeautifulSoup(html_content, 'html.parser')  # Parsear el HTML con BeautifulSoup

# Buscar todas las tablas en el HTML

tables = soup.find_all('table')

print(f"Se encontraron {len(tables)} tablas en la página.")

# Si encontramos tablas, analizamos cuál contiene ganancias anuales
for i, table in enumerate(tables):
    print(f"\nTabla {i + 1}:")
    print(table.prettify()[:500])  # Muestra los primeros 500 caracteres de cada tabla para análisis

# Seleccionar la Tabla 1
tabla_ganancias = tables[0]  # La primera tabla es la de evolución anual

# Extraer las filas de la tabla
filas = tabla_ganancias.find_all('tr')

# Crear una lista para almacenar los datos
data = []

# Iterar sobre las filas de la tabla
for fila in filas[1:]:  # Omitimos la primera fila (encabezados)
    columnas = fila.find_all('td')
    if len(columnas) >= 3:  # Asegurar que hay al menos 3 columnas
        year = columnas[0].text.strip()
        earnings = columnas[1].text.strip()
        change = columnas[2].text.strip()
        data.append([year, earnings, change])

# Crear un DataFrame de pandas con los datos extraídos
df = pd.DataFrame(data, columns=['Año', 'Ganancias', 'Cambio'])

# Ordenar los datos por la columna "Año" de menor a mayor 
df = df.sort_values("Año")

# Mostrar la tabla de datos
df

# Eliminar filas vacías
df = df.dropna()

# Función para convertir ganancias a números
def convertir_ganancias(valor):
    valor = valor.replace("$", "").replace(",", "").strip()  # Eliminar $ y espacios
    if "Million" in valor or "M" in valor:  # Convertir millones a números
        return float(valor.replace("Million", "").replace("M", "")) * 1e6
    elif "Billion" in valor or "B" in valor:  # Convertir billones a números
        return float(valor.replace("Billion", "").replace("B", "")) * 1e9
    else:
        return float(valor)  # Si no tiene unidades, devolver el número directamente

# Aplicar la función a la columna "Ganancias"
df["Ganancias"] = df["Ganancias"].apply(convertir_ganancias)

# Procesar la columna "Cambio" para eliminar valores vacíos y convertir en porcentaje decimal
df["Cambio"] = df["Cambio"].replace({'%': ''}, regex=True)
df["Cambio"] = pd.to_numeric(df["Cambio"], errors='coerce') / 100  # Convertir a float y manejar errores

# Eliminar filas donde "Cambio" sea NaN (valores vacíos que no se pudieron convertir)
df = df.dropna(subset=['Cambio'])

# Mostrar el DataFrame limpio
df

# Crear (o conectar a) una base de datos SQLite
conn = sqlite3.connect('tesla_earnings.db')

# Crear un cursor para ejecutar comandos SQL
cursor = conn.cursor()

# Crear una tabla para almacenar los datos (si no existe)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS earnings (
        Año INTEGER PRIMARY KEY,
        Ganancias REAL,
        Cambio REAL
    )
''')

# Insertar los datos en la tabla
df.to_sql('earnings', conn, if_exists='replace', index=False)

# Guardar (commit) los cambios en la base de datos
conn.commit()

# Consultar la ganancia del último año disponible
query = "SELECT Año, Ganancias FROM earnings ORDER BY Año DESC LIMIT 1"
ultimo_año = pd.read_sql(query, conn)

# Cerrar la conexión
conn.close()

# Mostrar el resultado
año = ultimo_año.iloc[0, 0]  # Extrae el año
ganancias = ultimo_año.iloc[0, 1]  # Extrae las ganancias

print(f"Las ganancias de Tesla en el año {año} fueron de ${ganancias:,.2f}")
