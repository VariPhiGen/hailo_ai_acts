import os
import sys
import json
import asyncio
import subprocess
import hashlib
from datetime import datetime, timezone, timedelta
import requests
import websockets
import shutil
from dotenv import load_dotenv
from typing import Dict, Optional, Tuple

# -------------------------------------------------
# Continuous Model Download for OTA Updates
# -------------------------------------------------
# This module supports continuous/resumable downloads for large AI models during OTA updates.
# Key features:
# - Resume capability: Downloads can resume from interruption points
# - Progress persistence: Download progress is saved to disk and survives restarts
# - Large file support: Optimized timeouts and chunk sizes for models >100MB
# - Verification: Optional file size and SHA256 hash verification
# - Network resilience: Exponential backoff and generous timeouts for unstable networks
#
# Usage examples:
# 1. Basic download: download_file("https://example.com/model.hef", "/path/to/model.hef")
# 2. With verification: download_file(url, dst, expected_size=104857600, expected_hash="abc123...")
# 3. Large model download: download_file(url, dst, unstable_network=True, overall_timeout_hours=4)
# 4. Test resume: python edge_ota.py test <url> interrupt <expected_size>
#
# -------------------------------------------------
# Base paths (device-name agnostic)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "configuration.json")
MODEL_DIR = os.path.join(BASE_DIR, "resources/models/hailo8l")
HEF_FILE = "model.hef"
LABELS_FILE = "labels.json"

# -------------------------------------------------
# Load ENV
# -------------------------------------------------
load_dotenv(os.path.join(BASE_DIR, ".env"))
WS_SERVER_URL = os.getenv("WS_SERVER_URL")
WS_PING_INTERVAL = float(os.getenv("WS_PING_INTERVAL", "20"))
WS_PING_TIMEOUT = float(os.getenv("WS_PING_TIMEOUT", "60"))  # tolerate jitter

if not WS_SERVER_URL:
    raise RuntimeError("WS_SERVER_URL not found in .env")

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def load_config() -> Dict:
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save_config(cfg: Dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

def get_device_info() -> Dict:
    cfg = load_config()
    return {
        "sensor_id": cfg.get("sensor_id", "UNKNOWN"),
        "client_name": cfg.get("client_name", "UNKNOWN"),
        "device_name": cfg.get("device_name", "UNKNOWN")
    }

def verify_download(file_path: str, expected_size: int = None, expected_hash: str = None) -> bool:
    """Verify downloaded file integrity

    Args:
        file_path: Path to the downloaded file
        expected_size: Expected file size in bytes (optional)
        expected_hash: Expected SHA256 hash (optional)

    Returns:
        bool: True if verification passes, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"❌ File does not exist: {file_path}")
        return False

    actual_size = os.path.getsize(file_path)

    # Check size if expected
    if expected_size and actual_size != expected_size:
        print(f"❌ Size mismatch: expected {expected_size}, got {actual_size}")
        return False

    # Check hash if expected
    if expected_hash:
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        actual_hash = sha256.hexdigest()
        if actual_hash != expected_hash:
            print(f"❌ Hash mismatch: expected {expected_hash}, got {actual_hash}")
            return False

    print(f"✅ File verification passed: {file_path} ({actual_size} bytes)")
    return True


def download_file(url: str, dst: str, max_retries=3, unstable_network=False, overall_timeout_hours=2, expected_size=None, expected_hash=None):
    """Download file with network fluctuation resilience and resume capability for large models

    Args:
        url: Download URL
        dst: Destination file path
        max_retries: Maximum retry attempts
        unstable_network: If True, uses very generous timeouts for unreliable connections
        overall_timeout_hours: Maximum total time to spend on download (default: 2 hours)
        expected_size: Expected file size for verification (optional)
        expected_hash: Expected SHA256 hash for verification (optional)
    """
    import time
    from urllib3.exceptions import IncompleteRead

    overall_timeout_seconds = overall_timeout_hours * 3600
    start_time = time.time()

    # Progress file to track download state
    progress_file = dst + ".progress"
    temp_file = dst + ".tmp"

    # Try to get file size with HEAD request (may not work with all S3 presigned URLs)
    total_size = 0
    supports_range = False

    try:
        head_response = requests.head(url, timeout=30, headers={"Accept-Encoding": "identity"})
        if head_response.status_code == 200:
            total_size = int(head_response.headers.get('content-length', 0))
            supports_range = head_response.headers.get('accept-ranges') == 'bytes'
            print(f"📊 File size: {total_size / (1024*1024):.1f} MB, Range support: {supports_range}")
    except Exception as e:
        print(f"⚠️ Could not get file size via HEAD request: {e}")

    for attempt in range(max_retries):
        # Check if we've exceeded overall timeout
        elapsed_time = time.time() - start_time
        if elapsed_time > overall_timeout_seconds:
            raise TimeoutError(f"Download exceeded overall timeout of {overall_timeout_hours} hours ({elapsed_time:.1f}s elapsed)")

        try:
            # Check if we have a partial download to resume
            existing_size = 0
            if os.path.exists(temp_file):
                existing_size = os.path.getsize(temp_file)
                print(f"🔄 Resuming download from byte {existing_size}")

            # For unstable networks, be very patient - allow up to 1 hour per chunk for massive files
            if unstable_network:
                # Very generous timeouts for unreliable networks, but avoid indefinite hangs
                connection_timeout = 120  # 2 minutes to connect
                read_timeout = 300  # 5 minutes per read to prevent permanent stalls
            else:
                # Calculate adaptive timeouts based on file size for stable networks
                base_read_timeout = 120  # 2 minutes base
                if total_size > 100 * 1024 * 1024:  # > 100MB
                    # Scale timeout: allow up to 10 minutes for very large files
                    read_timeout = min(600, base_read_timeout + (total_size // (100 * 1024 * 1024)) * 60)
                else:
                    read_timeout = base_read_timeout
                connection_timeout = 30

            # Prepare headers for resume capability
            headers = {"Accept-Encoding": "identity"}
            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"

            with requests.get(
                    url,
                    stream=True,
                    timeout=(connection_timeout, read_timeout),
                    headers=headers
                ) as r:
                    r.raise_for_status()

                    # Check if server supports range requests
                    if existing_size > 0 and r.status_code != 206:
                        print("⚠️ Server doesn't support range requests, restarting download")
                        existing_size = 0
                        if os.path.exists(temp_file):
                            os.remove(temp_file)

                    # Get actual content length for this request
                    content_length = int(r.headers.get('content-length', 0))
                    if total_size == 0:
                        total_size = existing_size + content_length

                    downloaded = existing_size
                    mode = 'ab' if existing_size > 0 else 'wb'

                    with open(temp_file, mode) as f:
                        for chunk in r.iter_content(65536):  # Larger chunks for better performance
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)

                                # Save progress periodically (every 5MB)
                                if downloaded % (5 * 1024 * 1024) < 65536:
                                    with open(progress_file, 'w') as pf:
                                        pf.write(f"{downloaded}\n{total_size}\n{time.time()}")
                                    if total_size > 0:
                                        print(f"📥 Progress: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({downloaded/total_size*100:.1f}%)")
                                    else:
                                        print(f"📥 Progress: {downloaded / (1024*1024):.1f} MB (total unknown)")

                    # Verify download completion
                    if total_size > 0 and downloaded >= total_size:
                        # Move temp file to final destination
                        os.rename(temp_file, dst)
                        # Clean up progress file
                        if os.path.exists(progress_file):
                            os.remove(progress_file)

                        # Verify the downloaded file
                        if verify_download(dst, expected_size or total_size, expected_hash):
                            print(f"✅ Download completed and verified: {dst} ({downloaded / (1024*1024):.1f} MB)")
                            return  # Success, exit function
                        else:
                            print("❌ Download verification failed, will retry...")
                            if os.path.exists(dst):
                                os.remove(dst)
                            continue

                    elif total_size == 0:
                        # No size info available, assume complete
                        os.rename(temp_file, dst)
                        if os.path.exists(progress_file):
                            os.remove(progress_file)

                        # Verify the downloaded file if we have expectations
                        if (expected_size or expected_hash) and not verify_download(dst, expected_size, expected_hash):
                            print("❌ Download verification failed, will retry...")
                            if os.path.exists(dst):
                                os.remove(dst)
                            continue

                        print(f"✅ Download completed: {dst}")
                        return

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print(f"❌ Access forbidden (403) for URL: {url}")
                print("   This could be due to:")
                print("   - Expired S3 presigned URL (URLs expire after 1 hour)")
                print("   - Incorrect AWS credentials in server")
                print("   - Bucket permissions issue on S3 server")
                print("   - Wrong S3 endpoint URL configuration")
                print("   - Clock skew between server and S3")
                print("   - HEAD request not supported (trying GET instead)")
                print(f"   URL endpoint: {url.split('://')[1].split('/')[0] if '://' in url else 'unknown'}")
                if attempt == max_retries - 1:
                    raise Exception(f"S3 access forbidden (403): URL may be expired or HEAD requests not supported. Try downloading manually first.")
            elif e.response.status_code == 416:
                # Range not satisfiable - file may be complete
                if os.path.exists(temp_file):
                    actual_size = os.path.getsize(temp_file)
                    if total_size > 0 and actual_size >= total_size:
                        os.rename(temp_file, dst)
                        if os.path.exists(progress_file):
                            os.remove(progress_file)
                        print(f"✅ File already complete: {dst}")
                        return
                if attempt == max_retries - 1:
                    raise e
            else:
                if attempt == max_retries - 1:
                    raise e
        except (requests.exceptions.RequestException, OSError, IncompleteRead) as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Download failed after {max_retries} attempts: {e}")
                raise e
            # Exponential backoff: wait 2^attempt seconds, but cap at 30 seconds for large files
            wait_time = min(30, 2 ** attempt)
            print(f"Download attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

def set_nested_key(cfg: Dict, dotted_key: str, value):
    keys = dotted_key.split(".")
    ref = cfg
    for k in keys[:-1]:
        ref = ref.setdefault(k, {})
    ref[keys[-1]] = value

def nested_key_exists(cfg: Dict, dotted_key: str) -> bool:
    """Check if a dotted path exists in the config."""
    keys = dotted_key.split(".")
    ref = cfg
    for k in keys[:-1]:
        if not isinstance(ref, dict) or k not in ref:
            return False
        ref = ref[k]
    return isinstance(ref, dict) and keys[-1] in ref


def read_cpu_temp() -> Optional[float]:
    """Best-effort CPU temp in Celsius."""
    candidates = [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/hwmon/hwmon0/temp1_input",
    ]
    for path in candidates:
        try:
            with open(path, "r") as f:
                raw = f.read().strip()
                val = float(raw)
                # Some files return millidegrees
                if val > 200:
                    val = val / 1000.0
                return val
        except Exception:
            continue
    return None


def get_loadavg() -> Tuple[float, float, float]:
    try:
        return os.getloadavg()
    except Exception:
        return (0.0, 0.0, 0.0)


def ping_host(host: str, timeout_seconds: int = 1) -> bool:
    if not host:
        return False
    try:
        res = subprocess.run(
            ["ping", "-c", "1", "-W", str(timeout_seconds), host],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return res.returncode == 0
    except Exception:
        return False


def build_health_report() -> Dict:
    cfg = load_config()
    cam_ip = cfg.get("camera_details", {}).get("camera_ip")
    cpu_temp = read_cpu_temp()
    load1, load5, load15 = get_loadavg()
    camera_reachable = ping_host(cam_ip)

    return {
        "status": "success",
        "cpu_temp_c": cpu_temp,
        "load_avg": {
            "1m": load1,
            "5m": load5,
            "15m": load15,
        },
        "camera_ip": cam_ip,
        "camera_reachable": camera_reachable,
        "time": (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).isoformat(),
    }

# -------------------------------------------------
# OTA Handlers
# -------------------------------------------------
def update_models(step: Dict):
    os.makedirs(MODEL_DIR, exist_ok=True)
    hef_path = os.path.join(MODEL_DIR, HEF_FILE)
    labels_path = os.path.join(MODEL_DIR, LABELS_FILE)

    # Backup current
    if os.path.exists(hef_path):
        shutil.copy2(hef_path, hef_path + ".backup")
    if os.path.exists(labels_path):
        shutil.copy2(labels_path, labels_path + ".backup")

    # For large model downloads, use extended timeout and unstable network settings
    download_kwargs = {
        "unstable_network": step.get("unstable_network", True),  # Assume unstable for model downloads
        "overall_timeout_hours": step.get("download_timeout_hours", 4)  # 4 hours for large models
    }

    if step.get("model_hef_url"):
        hef_size = step.get("model_hef_size")
        hef_hash = step.get("model_hef_hash")
        print(f"📥 Downloading model HEF file (size: {hef_size or 'unknown'}, hash: {hef_hash or 'none'})")
        download_file(step["model_hef_url"], hef_path,
                     expected_size=hef_size, expected_hash=hef_hash, **download_kwargs)

    if step.get("labels_json_url"):
        labels_size = step.get("labels_json_size")
        labels_hash = step.get("labels_json_hash")
        print(f"📥 Downloading labels JSON file (size: {labels_size or 'unknown'}, hash: {labels_hash or 'none'})")
        download_file(step["labels_json_url"], labels_path,
                     expected_size=labels_size, expected_hash=labels_hash, **download_kwargs)

def update_configuration(step: Dict):
    if step.get("mode") == "s3" and step.get("config_s3_url"):
        # Backup only when we intend to write
        shutil.copy2(CONFIG_PATH, CONFIG_PATH + ".backup")
        download_file(step["config_s3_url"], CONFIG_PATH)
        return

    cfg = load_config()

    # Validate all nested keys exist before making any changes
    for k in step.get("nested_updates", {}).keys():
        if k == "sensor_id":
            continue
        if not nested_key_exists(cfg, k):
            raise ValueError(f"Config key '{k}' not found; rejecting OTA update.")

    # Backup after validation succeeds
    shutil.copy2(CONFIG_PATH, CONFIG_PATH + ".backup")

    for k, v in step.get("nested_updates", {}).items():
        if k != "sensor_id":
            set_nested_key(cfg, k, v)
    save_config(cfg)

def rollback_models():
    hef_path = os.path.join(MODEL_DIR, HEF_FILE)
    labels_path = os.path.join(MODEL_DIR, LABELS_FILE)
    if os.path.exists(hef_path + ".backup"):
        shutil.copy2(hef_path + ".backup", hef_path)
    if os.path.exists(labels_path + ".backup"):
        shutil.copy2(labels_path + ".backup", labels_path)

def rollback_configuration():
    if os.path.exists(CONFIG_PATH + ".backup"):
        shutil.copy2(CONFIG_PATH + ".backup", CONFIG_PATH)

def make_scripts_executable():
    """Make essential shell scripts executable"""
    scripts_to_make_executable = [
        "run_detection.sh",
        "run_ota.sh",
        "setup_ota_env.sh",
        "setup_system.sh",
        "rpi_relay_fix.sh"
    ]

    print("🔧 Making essential scripts executable...")
    for script in scripts_to_make_executable:
        script_path = os.path.join(BASE_DIR, script)
        if os.path.exists(script_path):
            subprocess.run(["chmod", "+x", script_path], check=False)
            print(f"   ✅ {script}")
        else:
            print(f"   ⚠️ {script} not found")

def restart_services():
    """Restart detection and OTA services after updates"""
    print("🔄 Restarting services after update...")

    # Ensure scripts are executable before any restart attempt
    make_scripts_executable()

    # Prefer systemd restart if available; avoid killing OTA process directly
    detection_restarted = False
    if shutil.which("systemctl"):
        result = subprocess.run(["sudo", "-n", "systemctl", "restart", "acts-detection"], check=False)
        if result.returncode == 0:
            detection_restarted = True
            print("✅ Restarted acts-detection via systemd")
            # Wait for detection service to become active (Hailo init can take time)
            for _ in range(36):  # up to 3 minutes
                status = subprocess.run(["sudo", "-n", "systemctl", "is-active", "--quiet", "acts-detection"])
                if status.returncode == 0:
                    print("✅ acts-detection is active")
                    break
                import time
                time.sleep(5)
        else:
            print("⚠️ systemd restart failed for acts-detection, falling back to script restart")

    # Fallback: stop and restart detection locally
    if not detection_restarted:
        print("🛑 Stopping existing detection processes...")
        subprocess.run(["pkill", "-f", "detection.py"], check=False)
        subprocess.run(["pkill", "-f", "python.*detection"], check=False)  # More specific pattern

    # Give processes time to stop
    import time
    time.sleep(60)

    # Restart detection service (fallback path)
    if not detection_restarted:
        detection_script = os.path.join(BASE_DIR, "run_detection.sh")
        if os.path.exists(detection_script):
            print("🚀 Restarting detection service...")
            try:
                # Start in background
                process = subprocess.Popen(
                    [detection_script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd=BASE_DIR
                )
                print(f"✅ Detection service restarted (PID: {process.pid})")
            except Exception as e:
                print(f"⚠️ Failed to restart detection service: {e}")

    print("✅ Service restart completed")

def git_update(step: Dict):
    branch = step.get("branch", "main")
    tag = step.get("tag")
    do_pull = step.get("pull", True)
    restart_after_update = step.get("restart_services", True)  # Default to True
    # Default to True so OTA restarts with new code after git updates
    restart_ota_after_update = step.get("restart_ota", True)

    print("📥 Starting git update...")
    subprocess.run(["git", "fetch", "--all"], cwd=BASE_DIR, check=True)

    if branch:
        subprocess.run(["git", "checkout", branch], cwd=BASE_DIR, check=True)
        subprocess.run(["git", "reset", "--hard", f"origin/{branch}"], cwd=BASE_DIR, check=True)
        if do_pull:
            subprocess.run(["git", "pull"], cwd=BASE_DIR, check=True)

    if tag:
        subprocess.run(["git", "fetch", "--tags"], cwd=BASE_DIR, check=True)
        subprocess.run(["git", "checkout", tag], cwd=BASE_DIR, check=True)
        print("✅ Git tag checkout completed")

    # Always restart services after git operations (regardless of changes)
    if restart_after_update:
        print("🔄 Performing post-git service restart...")
        restart_services()
    else:
        print("ℹ️ Service restart disabled in configuration")

    # Defer OTA restart until after ACK is sent
    if restart_ota_after_update:
        global RESTART_OTA_REQUESTED
        RESTART_OTA_REQUESTED = True
        print("🕒 OTA restart scheduled after ACK")

def run_shell(step: Dict):
    script = step.get("script_name")
    if not script:
        return

    path = os.path.join(BASE_DIR, script)

    # Ensure the script exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Script not found: {path}")

    # Make executable
    print(f"🔧 Making script executable: {script}")
    subprocess.run(["chmod", "+x", path], check=False)

    # Run the script
    print(f"🚀 Executing script: {script}")
    result = subprocess.run([path], check=True, cwd=BASE_DIR, capture_output=True, text=True)

    if result.stdout:
        print(f"📝 Script output: {result.stdout.strip()}")

    # Check if we should restart services after script execution
    if step.get("restart_services_after", False):
        print("🔄 Restarting services after script execution...")
        restart_services()

    print(f"✅ Script {script} completed successfully")

def post_actions(step: Dict):
    """Execute post-update actions like service restarts and reboots"""

    # Restart specific system service
    if step.get("restart_service"):
        service_name = step["service_name"]
        print(f"🔄 Restarting system service: {service_name}")
        result = subprocess.run(["sudo", "systemctl", "restart", service_name], check=False)
        if result.returncode == 0:
            print(f"✅ System service {service_name} restarted")
        else:
            print(f"⚠️ Failed to restart system service {service_name}")

    # Restart application services
    if step.get("restart_app_services", False):
        restart_services()

    # Reboot system
    if step.get("reboot"):
        print("🔄 System reboot requested...")
        print("⚠️ System will reboot in 10 seconds")
        import time
        time.sleep(10)
        subprocess.run(["sudo", "reboot"])

    # Custom post-update commands
    if step.get("custom_commands"):
        for cmd in step["custom_commands"]:
            print(f"🚀 Executing custom command: {cmd}")
            try:
                result = subprocess.run(cmd, shell=True, check=True, cwd=BASE_DIR, capture_output=True, text=True)
                if result.stdout:
                    print(f"📝 Command output: {result.stdout.strip()}")
                print("✅ Custom command completed")
            except Exception as e:
                print(f"⚠️ Custom command failed: {e}")
                # Don't fail the entire OTA for custom command errors

def handle_ota(payload: Dict):
    device_info = get_device_info()
    device_id = device_info["sensor_id"]
    targets = payload.get("targets", {}).get("device_ids", [])
    if targets and device_id not in targets:
        return

    steps = payload.get("steps", {})

    try:
        if steps.get("model_update", {}).get("enabled"):
            update_models(steps["model_update"])
        if steps.get("config_update", {}).get("enabled"):
            update_configuration(steps["config_update"])
        if steps.get("git_update", {}).get("enabled"):
            git_update(steps["git_update"])
        if steps.get("shell_exec", {}).get("enabled"):
            run_shell(steps["shell_exec"])
        if steps.get("post_actions"):
            post_actions(steps["post_actions"])

        # Send ACK success
        return {
            "sensor_id": device_id,
            "status": "success",
            "command": payload.get("command"),
            "command_id": payload.get("command_id"),
        }
    except Exception as e:
        # On any failure, attempt rollback if configured
        rollback_errors = []
        if steps.get("rollback", {}).get("config"):
            try:
                rollback_configuration()
            except Exception as re:
                rollback_errors.append(f"config rollback failed: {re}")
        if steps.get("rollback", {}).get("model"):
            try:
                rollback_models()
            except Exception as re:
                rollback_errors.append(f"model rollback failed: {re}")

        error_msg = str(e)
        if rollback_errors:
            error_msg = f"{error_msg}; " + "; ".join(rollback_errors)

        # Send ACK failure with error detail
        return {
            "sensor_id": device_id,
            "status": "failed",
            "command": payload.get("command"),
            "command_id": payload.get("command_id"),
            "error": error_msg,
        }

# -------------------------------------------------
# WebSocket Client
# -------------------------------------------------
RECONNECT_BASE_DELAY = 3  # seconds
RECONNECT_MAX_DELAY = 60  # cap backoff
RESTART_OTA_REQUESTED = False


async def send_register(ws, device_info: Dict):
    """Send register payload to announce online status."""
    register_payload = {
        "type": "register",
        "sensor_id": device_info["sensor_id"],
        "status": "online",
    }
    await ws.send(json.dumps(register_payload))


async def health_publisher(ws, device_info: Dict, interval_seconds: int = 60):
    """Periodically publish device health."""
    while True:
        health = await asyncio.to_thread(build_health_report)
        # print({
        #     "type": "health",
        #     "sensor_id": device_info["sensor_id"],
        #     "payload": health,
        # })
        await ws.send(json.dumps({
            "type": "health",
            "sensor_id": device_info["sensor_id"],
            "payload": health,
        }))
        await asyncio.sleep(interval_seconds)


async def schedule_ota_restart(delay_seconds: int = 5):
    """Restart OTA service after a short delay to allow ACK to send."""
    await asyncio.sleep(delay_seconds)
    if shutil.which("systemctl"):
        subprocess.run(["sudo", "-n", "systemctl", "restart", "edge-ota"], check=False)
    else:
        print("⚠️ systemctl not available; OTA restart skipped")


async def ws_client():
    global RESTART_OTA_REQUESTED
    device_info = get_device_info()
    retry_attempt = 0

    while True:
        health_task = None
        try:
            async with websockets.connect(
                WS_SERVER_URL,
                ping_interval=WS_PING_INTERVAL,
                ping_timeout=WS_PING_TIMEOUT,
            ) as ws:
                # Successful connection resets retry counter
                retry_attempt = 0
                await send_register(ws, device_info)

                health_task = asyncio.create_task(health_publisher(ws, device_info))

                async for msg in ws:
                    data = json.loads(msg)
                    # print(data)
                    # Return current configuration
                    if data.get("command") == "get_configuration":
                        cfg = load_config()
                        ist_time = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
                        await ws.send(json.dumps({
                            "type": "config",
                            "sensor_id": device_info["sensor_id"],
                            "command_id": data.get("command_id"),
                            "payload": {
                                "status": "success",
                                "config": cfg,
                                "time": ist_time.isoformat(),
                            }
                        }))
                        continue
                    if data.get("command") == "health_check":
                        health = await asyncio.to_thread(build_health_report)
                        await ws.send(json.dumps({
                            "type": "health",
                            "sensor_id": device_info["sensor_id"],
                            "command_id": data.get("command_id"),
                            "payload": health,
                        }))
                        continue
                    if data.get("command") != "ota_update":
                        continue
                    ack = await asyncio.to_thread(handle_ota, data)
                    if ack:
                        # print(ack)
                        await ws.send(json.dumps({
                            "type": "ack",
                            "sensor_id": device_info["sensor_id"],
                            "command_id": data.get("command_id"),
                            "payload": ack
                        }))
                        if RESTART_OTA_REQUESTED:
                            RESTART_OTA_REQUESTED = False
                            asyncio.create_task(schedule_ota_restart())

        except Exception as e:
            retry_attempt += 1
            delay = min(RECONNECT_BASE_DELAY * retry_attempt, RECONNECT_MAX_DELAY)
            # print(f"WebSocket error: {e}. Retrying in {delay}s (attempt {retry_attempt})")
            await asyncio.sleep(delay)
        finally:
            # Ensure periodic health task is cleaned up
            if health_task:
                try:
                    health_task.cancel()
                except Exception:
                    pass


def test_continuous_download(url: str, expected_size: int = None, simulate_interrupt: bool = False):
    """Test continuous download functionality with optional interruption simulation"""
    import tempfile
    import time

    print("🧪 Testing continuous download functionality...")

    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(delete=False, suffix='.test') as tmp:
        test_file = tmp.name

    try:
        if simulate_interrupt:
            print("🔄 Testing download resume capability...")

            # Start download but interrupt after a short time
            import threading
            import signal

            def interrupt_download():
                time.sleep(2)  # Let download start
                os.kill(os.getpid(), signal.SIGINT)

            # Set up interrupt
            timer = threading.Timer(2.0, interrupt_download)
            timer.start()

            try:
                download_file(url, test_file, max_retries=1, overall_timeout_hours=0.1)
            except KeyboardInterrupt:
                print("⏸️ Download interrupted as planned")
                timer.cancel()

            # Check if progress file exists
            progress_file = test_file + ".progress"
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 1:
                        resumed_bytes = int(lines[0].strip())
                        print(f"📊 Progress saved: {resumed_bytes} bytes downloaded")

                        # Resume download
                        print("🔄 Resuming download...")
                        download_file(url, test_file, expected_size=expected_size)
                    else:
                        print("⚠️ Progress file exists but is empty")
            else:
                print("⚠️ No progress file found after interrupt")
        else:
            # Normal download test
            download_file(url, test_file, expected_size=expected_size)

        # Verify final file
        if os.path.exists(test_file):
            final_size = os.path.getsize(test_file)
            print(f"✅ Test completed. Final file size: {final_size} bytes")

            if expected_size and final_size == expected_size:
                print("✅ File size matches expected value")
            elif expected_size:
                print(f"⚠️ File size mismatch: expected {expected_size}, got {final_size}")
        else:
            print("❌ Test file not found after download")

    finally:
        # Clean up
        for ext in ['', '.tmp', '.progress']:
            cleanup_file = test_file + ext
            if os.path.exists(cleanup_file):
                os.remove(cleanup_file)
                print(f"🧹 Cleaned up: {cleanup_file}")


# -------------------------------------------------
# Entry
# -------------------------------------------------
if __name__ == "__main__":
    # Check for test URL argument
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        if len(sys.argv) > 2:
            url = sys.argv[2]
            simulate_interrupt = len(sys.argv) > 3 and sys.argv[3] == "interrupt"
            expected_size = int(sys.argv[4]) if len(sys.argv) > 4 else None
            test_continuous_download(url, expected_size, simulate_interrupt)
        else:
            print("Usage: python edge_ota.py test <url> [interrupt] [expected_size]")
    else:
        asyncio.run(ws_client())
