import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

runner = bentoml.keras.get("flower_model:cuh4o3gokoxi75fv").to_runner()

svc = bentoml.Service("clasificador_iris", runners=[runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def classify(input_series: np.ndarray) -> str:
    class_ind = np.argmax(await runner.predict.async_run(input_series), axis=-1)[0]
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    return classes[class_ind]
    
    
