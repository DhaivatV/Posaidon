from fastapi import FastAPI
import uvicorn
from fastapi import UploadFile, File
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from PIL import Image
import numpy as np
import io
from tensorflow.python import tf2
from pickle import load

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model_path = (r"C:\Users\dvipa\OneDrive\Desktop\Posaidon\Species_Classfier.h5")

model = load_model("Species_Classfier.h5")

class_dict = {'Anthias anthias': 0,
              'Atherinomorus lacunosus': 1,
              'Belone belone': 2,
              'Boops boops': 3,
              'Chlorophthalmus agassizi': 4,
              'Coris julis': 5,
              'Dasyatis centroura': 6,
              'Epinephelus caninus': 7,
              'Gobius niger': 8,
              'Mugil cephalus': 9,
              'Phycis phycis': 10,
              'Polyprion americanus': 11,
              'Pseudocaranx dentex': 12,
              'Rhinobatos cemiculus': 13,
              'Scomber japonicus': 14,
              'Solea solea': 15,
              'Squalus acanthias': 16,
              'Tetrapturus belone': 17,
              'Trachinus draco': 18,
              'Trigloporus lastoviza': 19}


@app.get("/")
def Hello_World():
    return "Hello World"


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
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
    key = (list(class_dict.keys())[list(class_dict.values()).index(pred)])
    return {"prediction": key }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
