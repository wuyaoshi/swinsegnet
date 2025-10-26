from fastapi import FastAPI, UploadFile, File, BackgroundTasks
import os, shutil, uuid
from test import run_inference   # 导入你在 test.py 里新增的函数

app = FastAPI()



UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

tasks = {}

def process_image(job_id: str, file_path: str):
    result_path = os.path.join(RESULT_DIR, f"{job_id}_mask.png")
    run_inference(file_path, result_path)   # 调用你的模型
    tasks[job_id]["status"] = "done"
    tasks[job_id]["result_file"] = result_path

@app.post("/upload")
async def upload_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    tasks[job_id] = {"status": "queued", "result_file": None}

    background_tasks.add_task(process_image, job_id, file_path)

    return {"job_id": job_id, "status": "queued"}

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    task = tasks.get(job_id)
    if not task:
        return {"error": "job_id not found"}
    return task
