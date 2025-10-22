import csv
import psutil
import platform
import subprocess
import sys
import os
import time
import threading


def log_to_csv(data, path):
    # Check if the file exists and needs headers
    file_exists = False
    try:
        with open(path, 'r'):
            file_exists = True
    except FileNotFoundError:
        pass
    
    # Append the new dictionary as a row in the CSV
    with open(path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        
        # Write the header only if the file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(data)


def get_metadata():
    try:
        git_repo = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode('utf-8')
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    except subprocess.CalledProcessError:
        git_repo = "Not a Git repository"
        git_commit = "No commit found"
    hostname = subprocess.check_output('hostname', shell=True).strip().decode('utf-8')
    username = subprocess.check_output('whoami', shell=True).strip().decode('utf-8')
    python_version = subprocess.check_output(['python', '--version'], stderr=subprocess.STDOUT).strip().decode('utf-8')
    python_executable = subprocess.check_output(['which', 'python'], stderr=subprocess.STDOUT).strip().decode('utf-8')
    metadata = {
        'cmd': 'python ' + ' '.join(sys.argv),
        'launch_dir': os.getcwd(),
        'python_version': python_version,
        'python_executable': python_executable,
        'OS': platform.platform(),
        'hostname': hostname,
        'username': username,
        'git_repo': git_repo,
        'git_commit': git_commit,
        'cpu_count': psutil.cpu_count(logical=False),
        'logical_cpu_count': psutil.cpu_count(logical=True),
        'ram_total_gb': round(psutil.virtual_memory().total / 1e9, 2),
        'ram_available_gb': round(psutil.virtual_memory().available / 1e9, 2),
        'ssd_total_gb': round(psutil.disk_usage('/').total / 1e9, 2),
        'ssd_available_gb': round(psutil.disk_usage('/').free / 1e9, 2),
        'job_id': os.environ.get('SLURM_JOB_ID'),
        'job_name': os.environ.get('SLURM_JOB_NAME'),
        'ram_allocated_mb': os.environ.get('SLURM_MEM_PER_NODE'),
        'cpu_allocated': os.environ.get('SLURM_CPUS_PER_TASK'),
        'time_limit': os.environ.get('SLURM_TIME_LIMIT'),
    }
    return metadata


def log_memory_usage_(log_file_path, interval=10, runner=None, stop_event=None):
    """Logs memory usage of the current process and its children to a file."""
    pid = os.getpid()
    process = psutil.Process(pid)
    
    with open(log_file_path, 'a') as log_file:
        log_file.write("Time,Step,Memory Usage (MB),CPU Usage (%)\n")
        while not (stop_event and stop_event.is_set()):
            time.sleep(interval)

            try:
                # get memory usage
                mem_info = process.memory_info().rss
                mem_children = sum(child.memory_info().rss for child in process.children(recursive=True))
                total_mem = (mem_info + mem_children) / (1024 ** 2)  # convert to MB
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

                # get cpu usage
                cpu_usage = process.cpu_percent(interval=None)
                cpu_children = sum(child.cpu_percent(interval=None) for child in process.children(recursive=True))
                total_cpu = cpu_usage + cpu_children

                # get current step
                current_step = runner.current_step()
                
                # write to log file
                log_entry = f"{timestamp},{current_step},{total_mem:.2f},{total_cpu:.2f}\n"
                log_file.write(log_entry)
                log_file.flush()  # ensure the data is written to the file
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},Error: Could not access memory info,,\n")
            except Exception as e:
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},Unexpected error: {e},,\n")


def log_memory_usage_periodically(log_file_path='memory_usage.log', log_interval_s=10, runner=None):
    """Periodically calls log_memory_usage_."""
    stop_event = threading.Event()
    logging_thread = threading.Thread(target=log_memory_usage_, args=(log_file_path, log_interval_s, runner, stop_event))
    logging_thread.daemon = True
    logging_thread.start()
    return logging_thread, stop_event
