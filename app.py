from fastapi import FastAPI
from transformers import pipeline


# create new FASTAPI instance
app = FastAPI()

# text generation pipeline
pipe = pipeline("text2text-generation", model="google/flan-t5-small")


@app.get("/")
def home():
    return {"message": "Hello World"}


@app.get("/generate")
def generate(text: str):
    ## use the pipeline to generate text from given input text
    output = pipe(text)

    ## return the generate text in Json reposne
    return {"output": output[0]["generated_text"]}
