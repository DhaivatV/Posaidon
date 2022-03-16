from fastapi import FastAPI
import uvicorn
from fastapi import UploadFile, File
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

model_path = (r"C:\Users\dvipa\OneDrive\Desktop\Posaidon\Species_Classfier.json")

model = load_model(model_path)

@app.get("/")
def Hello_World():
    return "Hello World"

@app.post('/predict')
async def predict (file : UploadFile = File(...)):
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents))
    pil_image = pil_image.resize((200, 200))
    print("resize done")
    i = np.asarray(pil_image)
    i = preprocess_input(i)
    input_arr = np.array([i])
    print("array made")
    pred = np.argmax(model.predict(input_arr))
    pred = pred.tolist()
    return {"prediction" : pred}

if __name__ == '__main__':
    uvicorn.run(app, host= '127.0.0.1', port= 8000)

