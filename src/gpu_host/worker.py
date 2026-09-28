"""Background worker for executing jobs."""

import asyncio
import os
import sys
import venv
from pathlib import Path
from datetime import datetime, timezone
import traceback

from sqlalchemy.orm import Session

from contextlib import contextmanager
from gpu_core.database import get_session_factory
from gpu_core.logging_config import get_logger
from gpu_core.models import Job, JobLog, JobStatus, RuntimeMode

logger = get_logger("host.worker")

_worker_task = None
_stop_event = asyncio.Event()

@contextmanager
def db_session():
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _save_log(job_id: str, stream: str, message: str):
    """Save a log entry synchronously."""
    with db_session() as db:
        log = JobLog(job_id=job_id, stream=stream, message=message)
        db.add(log)
        db.commit()

def _update_job_status(job_id: str, status: JobStatus, pid: int = None, exit_code: int = None, error_msg: str = None):
    with db_session() as db:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = status
            if pid is not None:
                job.pid = pid
            if exit_code is not None:
                job.exit_code = exit_code
            if error_msg is not None:
                job.error_message = error_msg
            
            if status == JobStatus.RUNNING and not job.started_at:
                job.started_at = _utcnow()
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                job.completed_at = _utcnow()
                
            db.commit()

def _get_next_job():
    """Get the next QUEUED job."""
    with db_session() as db:
        job = db.query(Job).filter(Job.status == JobStatus.QUEUED).order_by(Job.created_at.asc()).first()
        if job:
            job.status = JobStatus.STARTING
            db.commit()
            # return dict with needed info to decouple from DB session
            return {
                "id": job.id,
                "working_dir": job.working_dir,
                "entrypoint": job.entrypoint,
                "runtime_mode": job.runtime_mode,
                "requirements_file": job.requirements_file,
                "gpu_device": job.gpu_device
            }
        return None

async def _read_stream(stream, job_id: str, stream_name: str):
    """Read from an async stream and save to db."""
    try:
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode('utf-8', errors='replace').rstrip('\r\n')
            await asyncio.to_thread(_save_log, job_id, stream_name, text)
    except Exception as e:
        logger.error(f"Error reading {stream_name} for job {job_id}: {e}")

async def _setup_venv(working_dir: str, requirements_file: str | None, job_id: str) -> str:
    """Sets up a venv and returns the python executable path."""
    venv_dir = Path(working_dir) / ".venv"
    
    await asyncio.to_thread(_save_log, job_id, "stdout", f"Setting up virtual environment at {venv_dir}")
    
    if not venv_dir.exists():
        await asyncio.to_thread(venv.create, venv_dir, with_pip=True)
    
    if sys.platform == "win32":
        python_exe = venv_dir / "Scripts" / "python.exe"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
    else:
        python_exe = venv_dir / "bin" / "python"
        pip_exe = venv_dir / "bin" / "pip"

    if requirements_file:
        req_path = Path(working_dir) / requirements_file
        if req_path.exists():
            await asyncio.to_thread(_save_log, job_id, "stdout", f"Installing requirements from {requirements_file}")
            proc = await asyncio.create_subprocess_exec(
                str(pip_exe), "install", "-r", str(req_path),
                cwd=working_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await asyncio.gather(
                _read_stream(proc.stdout, job_id, "stdout"),
                _read_stream(proc.stderr, job_id, "stderr")
            )
            await proc.wait()
            if proc.returncode != 0:
                raise RuntimeError("Failed to install requirements")
                
    return str(python_exe)

async def _execute_job(job_info: dict):
    job_id = job_info["id"]
    working_dir = job_info["working_dir"]
    entrypoint = job_info["entrypoint"]
    
    try:
        if job_info["runtime_mode"] == RuntimeMode.VENV:
            python_exe = await _setup_venv(working_dir, job_info.get("requirements_file"), job_id)
        else:
            python_exe = sys.executable  # System python

        env = os.environ.copy()
        if job_info.get("gpu_device") and job_info["gpu_device"] != "auto":
            env["CUDA_VISIBLE_DEVICES"] = job_info["gpu_device"]
            
        # Optional: inject code to hook into metrics / checkpoints
        # For now just run the entrypoint
        
        await asyncio.to_thread(_update_job_status, job_id, JobStatus.RUNNING)
        await asyncio.to_thread(_save_log, job_id, "stdout", f"Starting execution of {entrypoint}")
        
        proc = await asyncio.create_subprocess_exec(
            python_exe, entrypoint,
            cwd=working_dir,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await asyncio.to_thread(_update_job_status, job_id, JobStatus.RUNNING, pid=proc.pid)
        
        await asyncio.gather(
            _read_stream(proc.stdout, job_id, "stdout"),
            _read_stream(proc.stderr, job_id, "stderr")
        )
        
        exit_code = await proc.wait()
        
        if exit_code == 0:
            await asyncio.to_thread(_update_job_status, job_id, JobStatus.COMPLETED, exit_code=exit_code)
            await asyncio.to_thread(_save_log, job_id, "stdout", "Job completed successfully")
        else:
            await asyncio.to_thread(_update_job_status, job_id, JobStatus.FAILED, exit_code=exit_code, error_msg=f"Exited with code {exit_code}")
            await asyncio.to_thread(_save_log, job_id, "stderr", f"Job failed with exit code {exit_code}")
            
    except Exception as e:
        logger.error(f"Job {job_id} execution failed: {e}")
        await asyncio.to_thread(_update_job_status, job_id, JobStatus.FAILED, error_msg=str(e))
        await asyncio.to_thread(_save_log, job_id, "stderr", f"Execution error: {e}\n{traceback.format_exc()}")

async def _worker_loop():
    logger.info("Background job worker started.")
    while not _stop_event.is_set():
        try:
            job_info = await asyncio.to_thread(_get_next_job)
            if job_info:
                logger.info(f"Worker picked up job {job_info['id']}")
                await _execute_job(job_info)
            else:
                # No jobs, sleep for a bit
                await asyncio.sleep(2)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Worker loop error: {e}")
            await asyncio.sleep(5)
    logger.info("Background job worker stopped.")

def start_worker():
    global _worker_task, _stop_event
    _stop_event.clear()
    if _worker_task is None or _worker_task.done():
        loop = asyncio.get_running_loop()
        _worker_task = loop.create_task(_worker_loop())

def stop_worker():
    global _worker_task, _stop_event
    if _worker_task and not _worker_task.done():
        _stop_event.set()
        _worker_task.cancel()
