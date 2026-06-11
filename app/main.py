from fastapi import FastAPI

from app.routers.prediction import router

app=FastAPI(
    title="OutbreakIQ API",
    version="1.0"
)

app.include_router(router)

@app.get("/")
def root():

    return {
        "project":"OutbreakIQ",
        "status":"running"
    }