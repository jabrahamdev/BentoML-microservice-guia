# Microservicio de un modelo de Machine Learning con BentoML
Por [J. Abraham](https://github.com/jabrahamdev)

![](/attachments/bentoml-readme-header.jpeg)

## La guía más concisa para poner en marcha un modelo de Machine Learning en producción con BentoML.

De acuerdo con el sitio de la herramienta: 

"BentoML es una plataforma abierta, una manera más rápida de enviar tus modelos a producción. BentoML combina una mejor experiencia al momento de desarrollar con un enfoque en operar ML en producción.

En BentoML queremos que sus modelos ML se envíen de forma rápida, repetible y escalable. BentoML está diseñado para agilizar la transferencia a la implementación de producción, lo que facilita a los desarrolladores y científicos de datos probar, implementar e integrar sus modelos con otros sistemas.

Con BentoML, los científicos de datos pueden concentrarse principalmente en crear y mejorar sus modelos, al mismo tiempo que brindan a los ingenieros de implementación la tranquilidad de saber que nada en la lógica de implementación cambia y que el servicio de producción es estable.

En BentoML queremos que sus modelos ML se envíen de forma rápida, repetible y escalable. BentoML está diseñado para agilizar la transferencia a la implementación de producción, lo que facilita a los desarrolladores y científicos de datos probar, implementar e integrar sus modelos con otros sistemas.

Con BentoML, los científicos de datos pueden concentrarse principalmente en crear y mejorar sus modelos, al mismo tiempo que brindan a los ingenieros de implementación la tranquilidad de saber que nada en la lógica de implementación cambia y que el servicio de producción es estable."



## BentoML vs Flask vs FastAPI? :confused:

Flask y FastAPI son herramientas sólidas, populares por su gran desempeño, frecuentemente son utilizadas en MLOps para poner modelos de ML en producción y de cierta manera se han vuelto la norma para tal fin.

La diferencia entre BentoML y otras tecnologìas como Flask y FastAPI es que las últimas fueron construidas principalmente para satisfacer las necesidades de los desarrolladores **web** backend, mientras que BentoML es **una herramienta optimizada para trabajos de Machine Learning**, fue de hecho creada por ingenieros de datos y de machine learning vinculados con el desarrollo de [Prefect](https://www.prefect.io/), una herramienta utilizada en la automatización de dataflows.


## Instrucciones :pencil:

### **Repo**

Descarga o clona este repositorio

```
git clone git@github.com:jabrahamdev/BentoML-microservice-guia.git
```


### **Modelo**

![](https://editor.analyticsvidhya.com/uploads/51518iris%20img1.png)

Primero necesitamos un modelo de ML previamente entrenado, para fines prácticos de esta demostración este repositorio incluye un modelo en formato h5. El modelo está entrenado en el famoso dataset iris y clasifica si una flor es _setosa_, _virginica_ o _versicolor_ a partir de 4 datos (sepal_length,	sepal_width,	petal_length,	petal_width). 



### **Ambiente**

Crea un ambiente con tu herramienta predilecta, toma en cuenta que es importante crear el ambiente con python 3.7, 3.8, 3.9 o 3.10.

Ejemplo con `conda`:

```
conda create -n bentoenv python=3.8
```

Activa el ambiente

```
conda activate bentoenv
```


### **Dependencias**

Haz `cd` en el directorio e instala las dependencias

```
pip install -r requirements.txt
```


### **Guardar modelo en el _store_ de BentoML**

Para comenzar con BentoML, debemos guardar nuestro modelo  con la API de BentoML en su _store_ (almacen) de modelos (un directorio local administrado por BentoML). Este store se usa para administrar todos los modelos localmente, así como para acceder a ellos para servirlos.

Para ello ejecuta el archivo `modeltobento.py` que carga un modelo Keras del directorio y lo guarda en el BentoML Store, además imprime en la consola el tag que identifica a nuestro modelo en el store.

```python
from pathlib import Path
from tensorflow import keras
import bentoml


def model_to_bento(model_file: Path) -> None:
    model = keras.models.load_model(model_file)
    bento_model = bentoml.keras.save_model("flower_model", model) # "flower_model" es el nombre que identifica al modelo en el store
    print(f"Bento model tag = {bento_model.tag}")

if __name__ == "__main__":
    model_to_bento(Path('final_iris_model.h5'))

```

Para ejecutar el archivo

```
python3 modeltobento.py
```


Ejemplo de output en consola:

```
Bento model tag = flower_model:cuh4o3gokoxi75fv
```

Es importante que tengas en cuenta este tag, será útil al escribir el archivo `service.py`.

Si olvidas copiarlo, no te preocupes, con la siguiente instrucción puedes listar los modelos del BentoML Store


```
bentoml models list
```

![](/attachments/cap_iris8.png)




### **Servir modelo**

Antes de servir el modelo echemos un vistazo a `service.py` el archivo mediante el cual se pone en marcha el modelo como servicio.

```python
# service.py

import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

runner = bentoml.keras.get("flower_model:cuh4o3gokoxi75fv").to_runner()

svc = bentoml.Service("clasificador_iris", runners=[runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def classify(input_series: str) -> str:
    class_ind = np.argmax(await runner.predict.async_run(input_series), axis=-1)[0]
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    return classes[class_ind]

```


El método .`get` de la API de BentoML para Keras recibe un string con el tag, la referencia a nuestro modelo en el store.

Puedes utilizar el tag `latest` para utilizar siempre la última versión del modelo en el store. Los usuarios de Docker encontrarán familiar esta terminología.

Ejemplo:

```python
runner = bentoml.keras.get("flower_model:latest").to_runner()
```

`to_runner()` convierte el modelo en un **runner**, un concepto importante en BentoML

### Runners :runner:

De acuerdo con la documentación de BentoML:

"En BentoML, Runner representa una unidad de computación que se puede ejecutar en un trabajador remoto de Python y se escala de forma independiente.

Runner permite que `bentoml.Service` paralelice varias instancias de una clase `bentoml.Runnable`, cada una en su propio trabajador de Python. Cuando se inicia un BentoServer, se creará un grupo de procesos de ejecución y las llamadas a métodos de ejecución realizadas desde el código `bentoml.Service` se programarán entre esos ejecutores."

_


En `svc = bentoml.Service("clasificador_iris", runners=[runner])`, `"clasificador_iris"` indica el nombre que vamos a dar al servicio y en runners pasamos todos los runners que va ejecutar el mismo, BentoML está optimizado para la paralelización de estos runners.

`async def classify(input_series: str) -> str:` define a la función como asíncrona aunque la API puede definirse tanto de manera síncrona como asíncrona.

Según la documentación:

"La implementación de API asíncrona es más eficiente porque cuando se invoca un método asíncrono, el bucle de eventos se libera para atender otras solicitudes mientras esta solicitud espera los resultados del método. Además, BentoML configurará automáticamente la cantidad ideal de paralelismo en función de la cantidad disponible de núcleos de CPU. No es necesario ajustar más la configuración del bucle de eventos en casos de uso comunes."

_



```python
      class_ind = np.argmax(await runner.predict.async_run(input_series), axis=-1)[0]
```


En esta línea, `await runner.predict.async_run(input_series)` es la instrucción que está haciendo la predicción, recibe el array de los 4 datos sobre los que se va a hacer la predicción y regresa un array de 3 elementos correspondientes a las probabilidades de que la flor sea setosa, virginica o versicolor.

`np.argmax(await runner.predict.async_run(input_series), axis=-1)[0]` regresa **el índice del valor con la mayor probabilidad**

_


```python
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    return classes[class_ind]
```

Estas dos líneas regresan el string correspondiente al índice `class_ind` obtenido con la línea previa. Por ejemplo, si el índice con la mayor probabilidad (valor) es el '0' entonces el string regresado serà 'setosa', si es '1' regresará 'versicolor' y si es 2 regresará 'virginica'.
__


**Para ejecutar el servicio, utiliza la siguiente instrucción**

```
bentoml serve service:svc --reload
```


Este comando utiliza el archivo `service.py`, incluido en este repo para echar a andar el servicio, es importante que se encuentre en la raíz de tu directorio.



Ahora el modelo está servido localmente desde `http://127.0.0.1:3000/`, ve a la url desde el explorador

![](/attachments/cap_iris.png)

## Si has trabajado con FastAPI, la interfaz te será familiar

![](/attachments/cap_iris2.png)

_____



En el área de texto blanca es donde ingresamos nuestro array con los 4 datos que recibe el servicio.

Estos son algunos ejemplos de datapoints incluidos en el dataset que se utilizó para el entrenamiento del modelo

![](/attachments/cap_iris3.png)


Puedes utilizar alguno de estos ejemplos para probar el servicio

```
[[5.1, 3.5, 1.4, 0.2]]  (setosa)

[[4.9, 3.0, 1.4, 0.2]]  (setosa)

[[6.7, 3.0, 5.2, 2.3]]  (virginica)
```

Copia alguno de los array en el área blanca y presiona `Execute`


![](/home/jabraham/.notable/attachments/cap_iris6.png)

**El output esperado es '_setosa_'**

![](/attachments/cap_iris7.png)

## Excelente! :nerd_face:


No estás convencid@? Prueba con otro de los ejemplos, el tercero del que se sabe la flor es 'virginica' `[[6.7, 3.0, 5.2, 2.3]]` 

![](/attachments/cap_iris9.png)

![](/attachments/cap_iris10.png)

![](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZWUxOTNkYjgyMzY4MmUyODlkMGNlMjZhMzMxMWNiZTg3NDVlYjYwYyZjdD1n/Onz7Kbh3TlY6GVHp2q/giphy.gif)



 ## ¿Cómo convertir el servicio definido en un **Bento**? :bento:


![](/attachments/cap_iris11.png)
Imagen por [Ahmed Besbes](https://ahmedbesbes.medium.com/)

Una vez finalizada la definición del servicio, podemos construir el modelo y el servicio en un **Bento**. Bento es el formato de distribución de un servicio. Es un archivo autónomo que contiene todo el código fuente, el modelo y las especificaciones de dependencia necesarias para ejecutar el servicio.

El nombre viene del japonés, "un bento es la iteración japonesa de una comida para llevar de una sola porción o empacada en casa, a menudo para el almuerzo. Fuera de Japón, es común en otros estilos culinarios del este y sudeste asiático, especialmente en las cocinas china, coreana, singapurense, taiwanesa y más, ya que el arroz es un alimento básico común en la región."

Bento es la unidad de implementación en BentoML, uno de los artefactos más importantes para realizar un seguimiento en el flujo de trabajo de implementación de su modelo. BentoML proporciona comandos CLI y API para administrar Bentos y moverlos.

Para construir un Bento, primero cree un archivo `bentofile.yaml` en el directorio de su proyecto.

```yaml
service: "service:svc"  # El mismo que el argumento pasado a `bentoml serve`
labels:
   owner: bentoml-team
   stage: dev
include:
- "*.py"  # Un patrón para definir que archivos incluir en el Bento
python:
   packages:  # Paquetes adicionales requeridos por el servicio
   - scikit-learn
   - pandas
```

**A continuación, ejecuta el comando CLI `bentoml build` desde el mismo directorio:**

![](/attachments/cap_iris12.png)


¡Acabas de crear tu primer Bento y ahora está listo para servir en producción! Para empezar, ahora puede servirlo con el comando CLI `bentoml serve` y el tag con el que se construyó el Bento, ejemplo:

```
bentoml serve clasificador_iris:w6t6frwpgo3n75fv --production 
```

Una vez más, el modelo está siendo servido desde el puerto `localhost:3000`

![](/attachments/cap_iris13.png)



## ¿Cómo generar una imagen de Docker del Bento?

Una imagen Docker se puede generar automáticamente desde un Bento para la implementación o despliegue en producción (no hay una manera satisfactoria de traducir **deployment** al español) , a través del comando de CLI `bentoml containerize`, ejemplo:

```
bentoml containerize clasificador_iris:latest
```

**NOTA:** Para ejecutar este comando debes de tener Docker instalado y se asume que estás familiarizado con la herramienta. La instalación y explicación de Docker van más allá del scope de esta guía.

![](/attachments/cap_iris14.png)

![](/attachments/cap_iris15.png)


Con la instrucción `docker images` puedes corroborar que la imagen fue creada:

![](/attachments/cap_iris16.png)

Con el Bento Dockerizado puedes hacer un `push` en DockerHub y/o puedes hacer deployment en cualquier nube pública, ya sea Google Cloud Platform, Amazon Web Services, Azure, etc. con Kubernetes, o una IaaC como Terraform. Una vez obtenida la imagen en Docker, las opciones para hacer deployment son diversas, la decisión final dependerá de la integración con el sistema.


Posteriormente agregaré a esta guía como hacer deployment en alguna nube.

Espero que BentoML  te sirva para tus proyectos y si encuentras esta guía útil, compartela y regalale una estrella  :star2: a este humilde repositorio :pray:




## Pendientes

### Deployment del Bento Dockerizado en Google Cloud Run
### Deployment del Bento Dockerizado en Amazon Web Services
