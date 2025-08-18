import subprocess
import os
import click
import time
TARGET_USER = "riscv"                # Your RISC-V board username
TARGET_IP   = "192.168.33.96"        # Your RISC-V board IP address
def transfer_folder(folder: str, target_user: str, target_ip: str, target_dir: str):
    """
    Transfers a whole folder recursively to the target using scp -r.
    
    Parameters:
        folder (str): The path to the folder to be transferred.
        target_user (str): The username on the target machine.
        target_ip (str): The IP address of the target machine.
        target_dir (str): The destination directory on the target machine.
    """
    subprocess.run(
        ["scp", "-r", folder, f"{target_user}@{target_ip}:{target_dir}"],
        check=True
    )
    print("[Host] Folder transferred successfully.")

def check_remote_file_exists(target_user: str, target_ip: str, remote_file: str) -> bool:
    """
    Uses SSH to check if a remote file exists on the device.
    
    Parameters:
        target_user (str): Username on the device.
        target_ip (str): IP address of the device.
        remote_file (str): Full path to the file on the device.
        
    Returns:
        bool: True if the file exists, False otherwise.
    """
    # The command returns "exists" if the file is found.
    command = f"ssh {target_user}@{target_ip} 'test -f {remote_file} && echo exists'"
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return "exists" in result.stdout
    except subprocess.TimeoutExpired:
        print("SSH command timed out while checking for remote file.")
        return False

def retrieve_confirmation_file(target_user: str, target_ip: str, remote_file: str, local_destination: str):
    """
    Retrieves a confirmation file from the device using scp.
    
    Parameters:
        target_user (str): Username on the device.
        target_ip (str): IP address of the device.
        remote_file (str): Full path to the confirmation file on the device.
        local_destination (str): Path on the host where the file should be saved.
    """
    try:
        subprocess.run(
            ["scp", f"{target_user}@{target_ip}:{remote_file}", local_destination],
            check=True
        )
        print(f"[Host] Confirmation file retrieved successfully to {local_destination}.")
    except subprocess.CalledProcessError as e:
        print(f"[Host] Failed to retrieve confirmation file: {e}")

def remote_run_program_send_back_result(target_dir, script_run_program, model_name, batch_size):
    ssh_cmd = " && ".join([
        f"cd {target_dir}",
        f"python3 {script_run_program} --model-name {model_name} --batch-size {batch_size}"
    ])
    proc = subprocess.Popen(
        ["ssh", f"{TARGET_USER}@{TARGET_IP}", ssh_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy()
    )
    out, err = proc.communicate(timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(f"Remote build failed:\n{err.decode()}")
    subprocess.run([
        "scp",
        f"{TARGET_USER}@{TARGET_IP}:{target_dir}/output_file_{model_name}_{batch_size}.npz",
        "."
    ], check=True)

def poll_for_confirmation(target_user: str, target_ip: str, remote_file: str, local_destination: str, timeout: int = 300, poll_interval: int = 5):
    """
    Polls the device for the confirmation file and retrieves it once available.
    
    Parameters:
        target_user (str): Username on the device.
        target_ip (str): IP address of the device.
        remote_file (str): Full path to the confirmation file on the device.
        local_destination (str): Path on the host where the file should be saved.
        timeout (int): Maximum time to wait in seconds.
        poll_interval (int): Seconds between checks.
    """
    waited = 0
    while waited < timeout:
        print(f"[Host] Checking for remote confirmation file... {waited} sec elapsed")
        if check_remote_file_exists(target_user, target_ip, remote_file):
            print("[Host] Confirmation file exists on device. Retrieving...")
            retrieve_confirmation_file(target_user, target_ip, remote_file, local_destination)
            return
        else:
            time.sleep(poll_interval)
            waited += poll_interval
    print("[Host] Confirmation file was not found on device within the timeout period.")

