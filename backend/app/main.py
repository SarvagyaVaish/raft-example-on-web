import io
from typing import Union

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
import cv2
import numpy as np
from app.raft_utils import RAFTArgs, inference


app = FastAPI()

# Middleware: https://fastapi.tiangolo.com/tutorial/cors/
origins = [
    "http://localhost",
    "http://localhost:5173",  # Default webapp port
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/process")
async def process(
    first_image: UploadFile = File(...),
    second_image: UploadFile = File(...),
):
    print("Progress: /process called")

    # Read and decode images
    frame_1 = await first_image.read()
    frame_1 = cv2.imdecode(np.frombuffer(frame_1, np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite("input_frame_1.png", frame_1)

    frame_2 = await second_image.read()
    frame_2 = cv2.imdecode(np.frombuffer(frame_2, np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite("input_frame_2.png", frame_2)

    print("Progress: input images read")

    args = RAFTArgs()
    args.frame_1 = "input_frame_1.png"
    args.frame_2 = "input_frame_2.png"

    print("Progress: calling inference()")
    flow_img = inference(args)
    flow_img_bytes = cv2.imencode(".png", flow_img)[1].tobytes()

    # Return the image as a streaming response
    return StreamingResponse(io.BytesIO(flow_img_bytes), media_type="image/png")
