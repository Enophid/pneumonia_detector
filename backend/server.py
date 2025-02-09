import subprocess
import sys
import signal
from pathlib import Path
import socket

def find_available_port(start_port=8000, max_attempts=100):
    """Find an available port"""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except socket.error:
                continue
    raise RuntimeError("No available ports found")

def terminate_process(process):
    """Terminate process on Windows"""
    if sys.platform == 'win32':
        subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], 
                     check=False)

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\nShutting down server...")
    if 'api_process' in globals():
        terminate_process(api_process)
    sys.exit(0)

def main():
    global api_process
    
    try:
        # Setup paths
        project_root = Path(__file__).parent.parent.absolute()
        python_path = sys.executable

        # Register signal handler
        signal.signal(signal.SIGINT, signal_handler)
        
        # Find available port
        port = find_available_port()
        print(f"Starting FastAPI server on port {port}...")
        
        # Start FastAPI server
        api_process = subprocess.Popen(
            [python_path, "-m", "uvicorn", 
             "backend.route:app",
             "--host", "127.0.0.1",
             "--port", str(port),
             "--log-level", "info"],
            cwd=str(project_root)
        )
        
        print(f"Server running at http://localhost:{port}")
        print("Press Ctrl+C to stop")
        
        api_process.wait()
            
    except Exception as e:
        print(f"Error: {e}")
        if 'api_process' in globals():
            terminate_process(api_process)
        sys.exit(1)

if __name__ == "__main__":
    main()