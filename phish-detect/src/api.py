from fastapi import FastAPI
from main import PhishingDetector
import uvicorn

app = FastAPI()
detector = PhishingDetector()

@app.get("/")
async def index():
    return "Send an URL with to 'checksite?url=<url>'"

@app.get("/checksite")
async def get_predictions(url: str):
    return "This website is likely phishing." if detector.predict(url) == 1 else "This website is likely legitimate."


if __name__ == '__main__':
    try:
        uvicorn.run("api:app", host="localhost", port=8000, reload=True)
    except:
        quit()
# to run the app, use the following command:
# uvicorn app:app --host 127.0.0.1 --port 8000 --reload
