"""Start CAPITEX locally: python run.py. Ctrl+C stops services started here."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

PROJECT_DIR=Path(__file__).resolve().parent
FASTAPI_HOST='127.0.0.1'
FASTAPI_PORT=8000
STREAMLIT_PORT=8501
FASTAPI_HEALTH_URL='http://127.0.0.1:8000/api/health'
FASTAPI_DOCS_URL='http://localhost:8000/docs'
FRONTEND_HEALTH_URL='http://127.0.0.1:8501/_stcore/health'
FRONTEND_URL='http://localhost:8501'


def is_backend_ready(url=FASTAPI_HEALTH_URL,timeout=1.):
    try:
        with urlopen(url,timeout=timeout) as response:
            if response.status!=200:return False
            if url==FASTAPI_HEALTH_URL:
                return json.loads(response.read()).get('capitex_database')=='connected'
            return response.read().strip()==b'ok'
    except (OSError,ValueError):return False


def port_available(port):
    with socket.socket() as sock:
        try:sock.bind((FASTAPI_HOST,port));return True
        except OSError:return False


def terminate_process(process,name):
    if process is None or process.poll() is not None:return
    process.terminate()
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill();process.wait(timeout=5)
    print(f'{name} stopped.',flush=True)


def wait_ready(process,url,name,timeout=60.):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if process.poll() is not None:raise RuntimeError(f'{name} exited during startup. See its output above.')
        if is_backend_ready(url):return
        time.sleep(.4)
    raise RuntimeError(f'{name} did not become ready within {timeout:.0f} seconds.')


def main():
    if sys.platform=='win32':
        for stream in (sys.stdout,sys.stderr):
            if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    backend=frontend=None
    def stop_requested(sig,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGINT,stop_requested)
    signal.signal(signal.SIGTERM,stop_requested)
    if hasattr(signal,'SIGBREAK'):signal.signal(signal.SIGBREAK,stop_requested)
    child_options={'cwd':str(PROJECT_DIR)}
    if sys.platform=='win32':child_options['creationflags']=subprocess.CREATE_NO_WINDOW
    print('CAPITEX | Wallet & AI Predictor',flush=True)
    try:
        if not is_backend_ready():
            if not port_available(FASTAPI_PORT):raise RuntimeError('Port 8000 is occupied by another or unready service. Stop it and retry.')
            backend=subprocess.Popen([sys.executable,'-m','uvicorn','main:app','--host',FASTAPI_HOST,'--port',str(FASTAPI_PORT)],**child_options)
            wait_ready(backend,FASTAPI_HEALTH_URL,'Backend')
        else:print('Using the existing CAPITEX backend.',flush=True)
        print(f'API documentation: {FASTAPI_DOCS_URL}',flush=True)
        if not is_backend_ready(FRONTEND_HEALTH_URL):
            if not port_available(STREAMLIT_PORT):raise RuntimeError('Port 8501 is occupied by another or unready service. Stop it and retry.')
            frontend=subprocess.Popen([sys.executable,'-m','streamlit','run','app.py','--server.address',FASTAPI_HOST,
                '--server.port',str(STREAMLIT_PORT),'--server.headless','true'],**child_options)
            wait_ready(frontend,FRONTEND_HEALTH_URL,'Streamlit')
        else:print('Using the existing Streamlit service.',flush=True)
        state={'frontend_url':FRONTEND_URL,'backend_url':'http://localhost:8000','workspace':str(PROJECT_DIR),
               'launcher_pid':os.getpid(),'backend_pid':backend.pid if backend else None,
               'frontend_pid':frontend.pid if frontend else None}
        (PROJECT_DIR/'data').mkdir(exist_ok=True)
        (PROJECT_DIR/'data'/'local_services.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        print(f'CAPITEX is ready: {FRONTEND_URL}',flush=True)
        print('Press Ctrl+C to stop services started by this launcher. Reused services stay running.',flush=True)
        while True:
            for process,name in ((backend,'Backend'),(frontend,'Streamlit')):
                if process is not None and process.poll() is not None:
                    raise RuntimeError(f'{name} stopped unexpectedly.')
            time.sleep(.5)
    except KeyboardInterrupt:
        print('\nStopping CAPITEX...',flush=True)
        return 0
    except (RuntimeError,OSError) as exc:
        print(f'Unable to launch CAPITEX: {exc}',file=sys.stderr,flush=True)
        return 1
    finally:
        terminate_process(frontend,'Streamlit')
        terminate_process(backend,'Backend')


if __name__=='__main__':sys.exit(main())
