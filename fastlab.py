from fastapi import FastAPI
24
app = FastAPI()
def sum_two_args(x,y):
 return x+y
# Hello World route
@app.get("/")
def read_root():
 return {"Hello": "World"}