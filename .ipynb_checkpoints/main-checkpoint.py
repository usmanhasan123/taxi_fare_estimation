from fastapi import FastAPI
from app.inference import predict

print("Loaded main.py")
print(__file__)

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def prediction(request: int):
    prediction = predict(request)
    return {"prediction": prediction}