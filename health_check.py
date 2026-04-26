from fastapi import FastAPI
import uvicorn
import threading

app = FastAPI()

@app.get("/")
def health():
    return {"status": "training"}

def run_health_server():
    # Must be 0.0.0.0 and 7860 for Hugging Face to see it
    uvicorn.run(app, host="0.0.0.0", port=7860)

# Start the server in a separate thread so it doesn't block your training
threading.Thread(target=run_health_server, daemon=True).start()