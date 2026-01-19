import os
import sys
import json
import asyncio
import subprocess
from datetime import datetime, timezone, timedelta
import requests
import websockets
import shutil
from dotenv import load_dotenv
from typing import Dict, Optional, Tuple

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

def download_file(url: str, dst: str, max_retries=3, unstable_network=False, overall_timeout_hours=2):
    """Download file with network fluctuation resilience

    Args:
        url: Download URL
        dst: Destination file path
        max_retries: Maximum retry attempts
        unstable_network: If True, uses very generous timeouts for unreliable connections
        overall_timeout_hours: Maximum total time to spend on download (default: 2 hours)
    """
    import time
    from urllib3.exceptions import IncompleteRead

    overall_timeout_seconds = overall_timeout_hours * 3600
    start_time = time.time()

    for attempt in range(max_retries):
        # Check if we've exceeded overall timeout
        elapsed_time = time.time() - start_time
        if elapsed_time > overall_timeout_seconds:
            raise TimeoutError(f"Download exceeded overall timeout of {overall_timeout_hours} hours ({elapsed_time:.1f}s elapsed)")

        try:
            # First, get file size to calculate adaptive timeouts
            # Note: Some S3 presigned URLs may not support HEAD requests
            total_size = 0 

            # For unstable networks, be very patient - allow up to 1 hour per chunk for massive files
            if unstable_network:
                # Very generous timeouts for unreliable networks
                connection_timeout = 120  # 2 minutes to connect
                read_timeout = None  # No read timeout - wait as long as needed
            else:
                # Calculate adaptive timeouts based on file size for stable networks
                base_read_timeout = 120  # 2 minutes base
                if total_size > 100 * 1024 * 1024:  # > 100MB
                    # Scale timeout: allow up to 5 minutes for very large files
                    read_timeout = min(300, base_read_timeout + (total_size // (100 * 1024 * 1024)) * 30)
                else:
                    read_timeout = base_read_timeout
                connection_timeout = 30

            with requests.get(
                    url,
                    stream=True,
                    timeout=(connection_timeout, read_timeout),
                    headers={"Accept-Encoding": "identity"}  # important for MinIO
                ) as r:
                    r.raise_for_status()
                    downloaded = 0

                    with open(dst, "wb") as f:
                        for chunk in r.iter_content(16384):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)


                            # Log progress for large files every 10MB
                            if total_size > 10*1024*1024 and downloaded % (10*1024*1024) < 16384:
                                print(".1f")

            print(f"Download completed: {dst}")
            return  # Success, exit function

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
            else:
                if attempt == max_retries - 1:
                    raise e
        except (requests.exceptions.RequestException, OSError, IncompleteRead) as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Download failed after {max_retries} attempts: {e}")
                raise e
            # Exponential backoff: wait 2^attempt seconds
            wait_time = 2 ** attempt
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

    if step.get("model_hef_url"):
        download_file(step["model_hef_url"], hef_path)
    if step.get("labels_json_url"):
        download_file(step["labels_json_url"], labels_path)

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

def git_update(step: Dict):
    branch = step.get("branch", "main")
    tag = step.get("tag")
    do_pull = step.get("pull", True)

    subprocess.run(["git", "fetch", "--all"], cwd=BASE_DIR, check=True)

    if branch:
        subprocess.run(["git", "checkout", branch], cwd=BASE_DIR, check=True)
        subprocess.run(["git", "reset", "--hard", f"origin/{branch}"], cwd=BASE_DIR, check=True)
        if do_pull:
            subprocess.run(["git", "pull"], cwd=BASE_DIR, check=True)

    if tag:
        subprocess.run(["git", "fetch", "--tags"], cwd=BASE_DIR, check=True)
        subprocess.run(["git", "checkout", tag], cwd=BASE_DIR, check=True)

def run_shell(step: Dict):
    script = step.get("script_name")
    if not script:
        return
    path = os.path.join(BASE_DIR, script)
    subprocess.run(["chmod", "+x", path], check=False)
    subprocess.run([path], check=True)

def post_actions(step: Dict):
    if step.get("restart_service"):
        subprocess.run(["sudo", "systemctl", "restart", step["service_name"]], check=False)
    if step.get("reboot"):
        subprocess.run(["sudo", "reboot"])

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


async def ws_client():
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


# -------------------------------------------------
# Entry
# -------------------------------------------------
if __name__ == "__main__":
    # Check for test URL argument
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        if len(sys.argv) > 2:
            test_download_url(sys.argv[2])
        else:
            print("Usage: python edge_ota.py test <url>")
    else:
        asyncio.run(ws_client())
