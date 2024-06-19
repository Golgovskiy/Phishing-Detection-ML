import gradio as gr
from console_app import PhishingDetector
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

detector = PhishingDetector("models")

def gradio_interface():
    def predict(url):
        return "This website is likely phishing." if detector.predict(url) == 1 else "This website is likely legitimate."

    interface = gr.Interface(fn=predict, inputs="text", outputs="text", allow_flagging="never", title="Phishing website prediction")
    return interface

app = FastAPI()
gradio_app = gradio_interface()
gradio_app.launch(share=False)
# Mount the Gradio app as a static files app
app.mount("/gradio", StaticFiles(directory=gradio_app.flagging_dir), name="gradio")

@app.get("/predict")
def predict_api():
    return "This website is likely phishing." if detector.predict(url) == 1 else "This website is likely legitimate."
    
    
@app.get("/")
def read_root(url):
    return {"message": "Welcome to the Phishing website predictor!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)