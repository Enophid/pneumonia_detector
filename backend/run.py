import subprocess
import time
import webbrowser

def main():
    # Start FastAPI server
    api_process = subprocess.Popen(
        ["uvicorn", "backend.route:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd="."
    )
    
    # Wait for API server to start
    time.sleep(2)
    
    # Start React frontend
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd="./frontend"
    )
    
    try:
        api_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        api_process.terminate()
        frontend_process.terminate()

if __name__ == "__main__":
    main()