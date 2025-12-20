import io
import json
import time
import uuid
import queue
import threading
import signal
import sys
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.client import Config
from video_clipper import VideoClipRecorder
import cv2
import numpy as np
from api_handler import APIHandler
from helper_utils import make_labelled_image

# --- Optional Kafka dependency ---
# This project can run in API-only mode (no Kafka). Import kafka-python lazily/optionally so
# environments without it can still use API + S3 functionality.
try:
    from kafka import KafkaProducer  # type: ignore
    from kafka.errors import KafkaError, NoBrokersAvailable  # type: ignore
    KAFKA_AVAILABLE = True
    _KAFKA_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover
    KafkaProducer = None  # type: ignore

    class KafkaUnavailableError(Exception):
        """Raised when Kafka functionality is requested but kafka-python is not available."""

    KafkaError = KafkaUnavailableError  # type: ignore
    NoBrokersAvailable = KafkaUnavailableError  # type: ignore
    KAFKA_AVAILABLE = False
    _KAFKA_IMPORT_ERROR = _e

# S3 Transfer configuration
S3_TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=5 * 1024 * 1024,  # 5MB
    multipart_chunksize=5 * 1024 * 1024,
    max_concurrency=4
)
class KafkaHandler:
    """Resilient Kafka + S3 handler with continuous operation and async uploads."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # NOTE: Avoid referencing KafkaProducer directly in annotations because it may be None
        # when kafka-python isn't installed (API-only mode).
        self.kafka_pipeline: Optional[Any] = None
        self.kafka_available: bool = bool(KAFKA_AVAILABLE)
        self.s3_clients: Dict[str, Any] = {}
        self.api_s3_clients: Dict[str, Any] = {}
        self.recorder: Optional[VideoClipRecorder] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._stop_event = threading.Event()
        self._shutdown_requested = False

        # Flush settings
        self.last_flush_time = time.time()
        self.messages_since_flush = 0
        self.flush_interval = 5
        self.flush_threshold = 10

        # Kafka brokers
        self.brokers = self._get_broker_list()
        self.current_broker_index = 0
        self.broker_health = {b: True for b in self.brokers}

        # S3 configs - separate normal and API configs
        self.s3_configs = self._get_s3_configs()
        self.api_s3_configs = self._get_api_s3_configs()
        self.current_s3_index = 0
        self.current_api_s3_index = 0
        self.s3_health = {name: True for name in self.s3_configs.keys()}
        self.api_s3_health = {name: True for name in self.api_s3_configs.keys()}

        # Health check
        self.health_check_interval = int(self.config.get("kafka_variables", {}).get("health_check_interval", 15))
        self._health_thread = None

        #API Integration

        # Setup AWS S3 and Video Recorder
        self._setup_aws_s3()
        self._setup_video_recorder()
        self._start_health_monitor()
        self._setup_signal_handlers()

        if not self.kafka_available:
            print(
                "DEBUG: kafka-python not available; Kafka mode disabled. "
                f"API-only can still run. Import error: {_KAFKA_IMPORT_ERROR}"
            )
        
        # Initialize API Handler (after setup is complete)
        self.api_handler = APIHandler(
            executor=self.executor,
            upload_to_s3_safe=self.upload_to_s3_safe,
            config=self.config
        )

    # ------------------------- Setup helpers -------------------------
    def _extract_category_subcategory(self, message: Dict[str, Any]) -> Tuple[str, str]:
        """Extract event category and subcategory from message."""
        try:
            subcategory_full = message["absolute_bbox"][0]["subcategory"]
            parts = subcategory_full.split("-", 1)
            event_category = parts[0] if len(parts) > 0 else subcategory_full
            event_subcategory = parts[1] if len(parts) > 1 else ""
            return event_category, event_subcategory
        except (KeyError, IndexError, AttributeError) as e:
            print(f"DEBUG: Error extracting category/subcategory: {e}")
            return "Unknown", ""
        
    def _get_broker_list(self) -> List[str]:
        kafka_config = self.config.get("kafka_variables", {})
        bootstrap_servers = kafka_config.get("bootstrap_servers")
        if isinstance(bootstrap_servers, list) and bootstrap_servers:
            return bootstrap_servers
        brokers = []
        primary = kafka_config.get("primary_broker")
        secondary = kafka_config.get("secondary_broker")
        if primary:
            brokers.append(primary)
        if secondary:
            brokers.append(secondary)
        return brokers if brokers else ["localhost:9092"]

    def _get_s3_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get normal S3 configs (primary, secondary) for Kafka uploads"""
        aws_config = self.config.get("kafka_variables", {}).get("AWS_S3", {})
        s3_configs: Dict[str, Dict[str, Any]] = {}
        if "primary" in aws_config:
            s3_configs["primary"] = aws_config["primary"]
        if "secondary" in aws_config:
            s3_configs["secondary"] = aws_config["secondary"]
        return s3_configs

    def _get_api_s3_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get API S3 configs (api_primary, api_secondary) for API uploads"""
        aws_config = self.config.get("kafka_variables", {}).get("AWS_S3", {})
        api_s3_configs: Dict[str, Dict[str, Any]] = {}
        if "api_primary" in aws_config:
            api_s3_configs["api_primary"] = aws_config["api_primary"]
        if "api_secondary" in aws_config:
            api_s3_configs["api_secondary"] = aws_config["api_secondary"]
        return api_s3_configs
        

    def _setup_aws_s3(self):
        # Setup normal S3 clients
        for name, config in self.s3_configs.items():
            try:
                client = boto3.client(
                    "s3",
                    aws_access_key_id=config.get("aws_access_key_id"),
                    aws_secret_access_key=config.get("aws_secret_access_key"),
                    region_name=config.get("region_name"),
                    endpoint_url=f"{config.get('end_point_url')}" if config.get("end_point_url") else None,
                    config=Config(signature_version="s3v4")
                )
                self.s3_clients[name] = client
                print(f"DEBUG: Initialized S3 client for {name}: {config.get('BUCKET_NAME')}")
            except Exception as e:
                print(f"DEBUG: Failed to init S3 client {name}: {e}")
                self.s3_health[name] = False
        
        # Setup API S3 clients
        for name, config in self.api_s3_configs.items():
            try:
                client = boto3.client(
                    "s3",
                    aws_access_key_id=config.get("aws_access_key_id"),
                    aws_secret_access_key=config.get("aws_secret_access_key"),
                    region_name=config.get("region_name"),
                    endpoint_url=f"{config.get('end_point_url')}" if config.get("end_point_url") else None,
                    config=Config(signature_version="s3v4")
                )
                self.api_s3_clients[name] = client
                print(f"DEBUG: Initialized API S3 client for {name}: {config.get('BUCKET_NAME')}")
            except Exception as e:
                print(f"DEBUG: Failed to init API S3 client {name}: {e}")
                self.api_s3_health[name] = False

    def _setup_video_recorder(self):
        try:
            self.recorder = VideoClipRecorder(maxlen=300, fps=15, prefix="clips")
        except Exception as e:
            print(f"DEBUG: VideoClipRecorder init failed: {e}")
            self.recorder = None

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"DEBUG: Received signal {signum}, initiating graceful shutdown...")
            self._shutdown_requested = True
            self._stop_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    # ------------------------- Health monitoring -------------------------
    def _test_s3_connectivity(self, s3_name: str) -> bool:
        try:
            client = self.s3_clients.get(s3_name)
            if not client:
                return False
            bucket_name = self.s3_configs[s3_name].get("BUCKET_NAME")
            client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
            return True
        except Exception:
            return False

    def _test_api_s3_connectivity(self, s3_name: str) -> bool:
        try:
            client = self.api_s3_clients.get(s3_name)
            if not client:
                return False
            bucket_name = self.api_s3_configs[s3_name].get("BUCKET_NAME")
            client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
            return True
        except Exception:
            return False

    def _test_broker_connectivity(self, broker: str) -> bool:
        if not self.kafka_available or KafkaProducer is None:
            return False
        try:
            test_producer = KafkaProducer(bootstrap_servers=str(broker), request_timeout_ms=3000, max_block_ms=2000)
            test_producer.close()
            return True
        except Exception:
            return False

    def _start_health_monitor(self):
        def monitor():
            while not self._stop_event.is_set():
                try:
                    if self.kafka_available:
                        for broker in self.brokers:
                            if not self.broker_health.get(broker, False):
                                healthy = self._test_broker_connectivity(broker)
                                self.broker_health[broker] = healthy
                                if healthy:
                                    print(f"DEBUG: Broker recovered: {broker}")
                    
                    for s3_name in self.s3_configs.keys():
                        if not self.s3_health.get(s3_name, False):
                            healthy = self._test_s3_connectivity(s3_name)
                            self.s3_health[s3_name] = healthy
                            if healthy:
                                print(f"DEBUG: S3 recovered: {s3_name}")
                    
                    for s3_name in self.api_s3_configs.keys():
                        if not self.api_s3_health.get(s3_name, False):
                            healthy = self._test_api_s3_connectivity(s3_name)
                            self.api_s3_health[s3_name] = healthy
                            if healthy:
                                print(f"DEBUG: API S3 recovered: {s3_name}")
                except Exception as e:
                    print(f"DEBUG: Health monitor error: {e}")
                time.sleep(self.health_check_interval)
        
        self._health_thread = threading.Thread(target=monitor, daemon=True)
        self._health_thread.start()

    # ------------------------- Kafka helpers -------------------------
    def _get_next_healthy_broker(self) -> Optional[str]:
        healthy = [b for b, ok in self.broker_health.items() if ok]
        if not healthy:
            return None
        broker = healthy[self.current_broker_index % len(healthy)]
        self.current_broker_index += 1
        return broker

    def _create_kafka_producer(self, max_attempts=3) -> Optional[Any]:
        if not self.kafka_available or KafkaProducer is None:
            return None
        attempt = 0
        backoff = 1
        while attempt < max_attempts and not self._stop_event.is_set():
            broker = self._get_next_healthy_broker()
            if not broker and self.brokers:
                broker = self.brokers[self.current_broker_index % len(self.brokers)]
            if not broker:
                return None
            try:
                kafka_config = self.config.get("kafka_variables", {})
                producer = KafkaProducer(
                    bootstrap_servers=str(broker),
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    acks="all",
                    retries=5,
                    retry_backoff_ms=500,
                    compression_type="gzip",
                    batch_size=int(kafka_config.get("batch_size", 512 * 1024)),
                    buffer_memory=67108864,
                    max_request_size=10 * 1024 * 1024,
                    request_timeout_ms=10000,
                    max_block_ms=5000,
                    linger_ms=int(kafka_config.get("linger_ms", 50))
                )
                _ = producer.metrics()
                print(f"DEBUG: Connected to broker: {broker}")
                return producer
            except Exception as e:
                print(f"DEBUG: Failed to create Kafka producer with broker {broker}: {e}")
                self.broker_health[broker] = False
                time.sleep(backoff)
                backoff = min(backoff * 2, 10)
                attempt += 1
        return None

    def _ensure_kafka_pipeline(self) -> bool:
        if not self.kafka_available:
            return False
        if self.kafka_pipeline:
            return True
        producer = self._create_kafka_producer(max_attempts=1)
        if producer:
            self.kafka_pipeline = producer
            return True
        return False

    def _handle_broker_failure(self):
        if self.kafka_pipeline:
            try:
                self.kafka_pipeline.close()
            except Exception:
                pass
            finally:
                self.kafka_pipeline = None
        if self.kafka_available:
            self.kafka_pipeline = self._create_kafka_producer()

    # ------------------------- S3 upload helpers -------------------------
    def _get_next_healthy_s3(self, is_api: bool = False) -> Optional[str]:
        """Get next healthy S3 config name. If is_api=True, returns API S3 configs only."""
        if is_api:
            healthy = [name for name, ok in self.api_s3_health.items() if ok]
            if not healthy:
                return None
            name = healthy[self.current_api_s3_index % len(healthy)]
            self.current_api_s3_index += 1
        else:
            healthy = [name for name, ok in self.s3_health.items() if ok]
            if not healthy:
                return None
            name = healthy[self.current_s3_index % len(healthy)]
            self.current_s3_index += 1
        return name

    def _build_s3_url(self, config: Dict[str, Any], filename: str, file_type: str = "image") -> Optional[str]:
        """Construct full URL for an uploaded file using the provided config."""
        if not filename or not config:
            return None
        end_point = config.get('end_point_url', '')
        bucket = config.get('BUCKET_NAME', '')
        if file_type == "video":
            key_prefix = config.get('video_fn', '')
        elif file_type == "image":
            key_prefix = config.get('org_img_fn', '')
        else:
            key_prefix = config.get('cgi_fn', '')
        return f"{end_point}/{bucket}/{key_prefix}{filename}"

    def _get_s3_url(self, filename: str, file_type: str = "image") -> Optional[str]:
        """Build URL for normal (Kafka) S3 using the first available config."""
        if not filename or not self.s3_configs:
            return None
        first = next(iter(self.s3_configs.values()), None)
        return self._build_s3_url(first, filename, file_type)

    def upload_to_s3(self, file_bytes: bytes, file_type: str = "image", is_api: bool = False) -> Optional[Dict[str, str]]:
        """Upload file to S3 with retries and failover.
        
        Args:
            file_bytes: The file data to upload
            file_type: Type of file ("image", "video", "snapshot")
            is_api: If True, uses API S3 configs (api_primary, api_secondary), 
                   otherwise uses normal S3 configs (primary, secondary)
        """
        if not file_bytes:
            return None

        upload_retries = int(self.config.get("kafka_variables", {}).get("AWS_S3", {}).get("upload_retries", 3))
        tried = set()

        # Select appropriate configs and clients based on is_api flag.
        if is_api:
            if not self.api_s3_configs:
                print("DEBUG: API upload requested but no API S3 configs; skipping upload.")
                return None
            s3_configs = self.api_s3_configs
            s3_clients = self.api_s3_clients
            s3_health = self.api_s3_health
        else:
            s3_configs = self.s3_configs
            s3_clients = self.s3_clients
            s3_health = self.s3_health

        # Try healthy S3 buckets first
        for _ in range(max(1, len(s3_configs))):
            s3_name = self._get_next_healthy_s3(is_api=is_api)
            if not s3_name or s3_name in tried:
                break
            tried.add(s3_name)
            
            client = s3_clients.get(s3_name)
            config = s3_configs.get(s3_name)
            if not client or not config:
                print(f"DEBUG: S3 client/config missing for {s3_name} (client={bool(client)}, config={bool(config)})")
                s3_health[s3_name] = False
                continue

            unique_filename = (f"clips{uuid.uuid4()}.mp4" if file_type == "video" else f"{uuid.uuid4()}.jpg")
            content_type = ("video/mp4" if file_type == "video" else "image/jpg")
            bucket = config.get("BUCKET_NAME")
            endpoint = config.get("end_point_url")
            key_prefix = config.get('org_img_fn', '') if file_type == "image" else config.get('cgi_fn', '')
            video_prefix = config.get('video_fn', '')

            for attempt in range(upload_retries):
                try:
                    if file_type == "video":
                        client.upload_fileobj(
                            io.BytesIO(file_bytes),
                            Bucket=config.get("BUCKET_NAME"),
                            Key=f"{video_prefix}{unique_filename}",
                            ExtraArgs={"ContentType": content_type},
                            Config=S3_TRANSFER_CONFIG,
                        )
                        url = self._build_s3_url(config, unique_filename, "video")
                        return {"filename": unique_filename, "url": url}
                    else:
                        client.put_object(
                            Bucket=bucket,
                            Key=f"{key_prefix}{unique_filename}",
                            Body=file_bytes,
                            ContentType=content_type,
                        )
                        url = self._build_s3_url(config, unique_filename, file_type)
                        return {"filename": unique_filename, "url": url}
                except Exception as e:
                    print(
                        f"DEBUG: S3 {file_type} upload attempt {attempt + 1} to {s3_name} "
                        f"(endpoint={endpoint}, bucket={bucket}, key_prefix={video_prefix if file_type=='video' else key_prefix}) failed: {e}"
                    )
                    time.sleep(0.5 * (attempt + 1))
            
            # Mark as unhealthy after retries
            s3_health[s3_name] = False

        return None

    def upload_to_s3_safe(self, file_bytes: bytes, file_type: str = "image", is_api: bool = False) -> Optional[Dict[str, str]]:
        """Non-blocking S3 upload with retries."""
        try:
            if file_bytes:
                return self.upload_to_s3(file_bytes, file_type, is_api=is_api)
        except Exception as e:
            print(f"DEBUG: S3 upload failed for {file_type}: {e}")
        return None

    def _get_api_s3_url(self, filename: str, file_type: str = "image") -> Optional[str]:
        """Construct full URL for an uploaded file using API S3 configs."""
        if not filename or not self.api_s3_configs:
            return None
        # Use the first available API S3 config
        config_name = next(iter(self.api_s3_configs.keys()), None)
        if not config_name:
            return None
        config = self.api_s3_configs[config_name]
        end_point = config.get('end_point_url', '')
        bucket = config.get('BUCKET_NAME', '')
        
        if file_type == "video":
            key_prefix = config.get('video_fn', '')
        elif file_type == "image":
            key_prefix = config.get('org_img_fn', '')
        else:
            key_prefix = config.get('cgi_fn', '')
        
        return f"{end_point}/{bucket}/{key_prefix}{filename}"

    # ------------------------- Flush helpers -------------------------
    def _smart_flush(self):
        if not self.kafka_pipeline:
            return
        self.messages_since_flush += 1
        current_time = time.time()
        if self.messages_since_flush >= self.flush_threshold or (current_time - self.last_flush_time) >= self.flush_interval:
            try:
                self.kafka_pipeline.flush(timeout=5)
            except Exception as e:
                print(f"DEBUG: Smart flush failed: {e}")
            finally:
                self.last_flush_time = time.time()
                self.messages_since_flush = 0

    def _force_flush(self):
        if self.kafka_pipeline:
            try:
                self.kafka_pipeline.flush()
            except Exception as e:
                print(f"DEBUG: Force flush failed: {e}")
            finally:
                self.last_flush_time = time.time()
                self.messages_since_flush = 0

    # ------------------------- Queue processing -------------------------
    def process_events_queue(self, events_queue: queue.Queue, topic: str,api_m,kafka_m) -> bool:
        try:
            message = events_queue.get(timeout=1)
            if message is None or topic == "None":
                return True

                
            if message["org_img"] is not None:
                absolute_bbox = message.get("absolute_bbox") or []
                # Support both list and single-dict shapes
                if isinstance(absolute_bbox, dict):
                    confidence_values = [absolute_bbox.get("confidence", 0)]
                else:
                    confidence_values = [
                        bbox.get("confidence", 0)
                        for bbox in absolute_bbox
                        if isinstance(bbox, dict)
                    ]
                max_confidence = max(confidence_values) if confidence_values else 0

                if api_m:
                    if max_confidence >= 0.85:
                        self.api_handler.process_and_submit(message, topic)
                    else:
                        print(f"DEBUG: Skipping API submission, confidence too low ({max_confidence:.2f})")
                
                if kafka_m:
                    if not (0.7 <= max_confidence < 0.85):
                        print(f"DEBUG: Skipping Kafka submission, confidence {max_confidence:.2f} outside 0.70-0.90")
                        return True
                    image_bytes = message.get("org_img")
                    snap_shot_bytes = message.get("snap_shot")
                    video_bytes = message.get("video")
                    # Extract category and subcategory
                    event_category, event_subcategory = self._extract_category_subcategory(message)

                    if not image_bytes:
                        print("DEBUG: No image bytes in message, skipping API submission")
                        return False

                    # Create labelled image first (needed before parallel uploads)
                    labelled_image_bytes = make_labelled_image(message,event_category,event_subcategory)
                    uploads = {}
                    futures = {
                        self.executor.submit(self.upload_to_s3_safe, labelled_image_bytes, "image", False): "org_img",
                        self.executor.submit(self.upload_to_s3_safe, snap_shot_bytes, "snapshot", False): "snap_shot",
                        self.executor.submit(self.upload_to_s3_safe, video_bytes, "video", False): "video"
                    }

                    for fut in as_completed(futures):
                        key = futures[fut]
                        try:
                            uploads[key] = fut.result()
                        except Exception as e:
                            print(f"DEBUG: Upload task failed for {key}: {e}")
                            uploads[key] = None
                    # Build final message payload just before send
                    org_img_info = uploads.get("org_img") or {}
                    snap_shot_info = uploads.get("snap_shot") or {}
                    video_info = uploads.get("video") or {}

                    message_out = {
                        "sensor_id": message.get("sensor_id"),
                        "org_img": org_img_info.get("url"),
                        "org_img_local": org_img_info.get("filename"),
                        "video_url": video_info.get("filename"),
                        "topic": "arresto_event_produce2",
                        "absolute_bbox": message.get("absolute_bbox"),
                        "datetimestamp_trackerid": message.get("datetimestamp"),
                        "imgsz": message.get("imgsz"),
                        "severity": "Medium",
                        "count": len(message.get("absolute_bbox") or []),
                    }


                    topic_str = str(topic) if topic is not None else ""
                    if not topic_str:
                        print("DEBUG: Kafka topic is empty or invalid, skipping message")
                        return False

                    if not self._ensure_kafka_pipeline():
                        print("DEBUG: Kafka pipeline unavailable, skipping message")
                        return False
                    try:
                        future = self.kafka_pipeline.send(topic_str, message_out)
                        record_metadata = future.get(timeout=10)
                        self._smart_flush()
                        #print(f"DEBUG: Message sent to partition {record_metadata.partition} offset {record_metadata.offset}")
                        return True
                    except (KafkaError, NoBrokersAvailable):
                        self._handle_broker_failure()
                        return False
                    except Exception as e:
                        print(f"DEBUG: Kafka send failed: {e}")
                        return False
                return True
            else:
                print("DEBUG: Insufficient uploads, message skipped")
                return False

        except queue.Empty:
            return True
        except Exception as e:
            print(f"DEBUG: Events queue processing error: {e}")
            return False

    # ------------------------- Main loop -------------------------
    def run_kafka_loop(self, events_queue: queue.Queue, analytics_queue: queue.Queue = None, api_mode = False, kafka_mode = False ):
        """Main Kafka processing loop - runs indefinitely until shutdown is requested"""
        kafka_config = self.config.get("kafka_variables", {})
        send_events_pipeline = kafka_config.get("send_events_pipeline")
        queues_and_topics = [(events_queue, send_events_pipeline)]
        api_m = True if api_mode == 1 else False
        kafka_m = True if kafka_mode == 1 else False

        # If Kafka was requested but kafka-python isn't available, gracefully degrade to API-only.
        if kafka_m and not self.kafka_available:
            print("DEBUG: kafka_mode requested but kafka-python is unavailable; running with kafka_mode=0 (API-only).")
            kafka_m = False

        consecutive_empty_cycles = 0
        retry_sleep = 1

        print(f"DEBUG: Starting continuous Kafka loop with brokers: {self.brokers}")
        print(f"DEBUG: Normal S3 buckets: {[config.get('BUCKET_NAME') for config in self.s3_configs.values()]}")
        print(f"DEBUG: API S3 buckets: {[config.get('BUCKET_NAME') for config in self.api_s3_configs.values()]}")

        while not self._stop_event.is_set():
            try:
                # Important: only touch Kafka pipeline if kafka_m is True.
                # Also: do NOT block API processing just because Kafka is down. If Kafka can't be
                # ensured, we temporarily disable Kafka publishing for this cycle.
                kafka_enabled = kafka_m
                if kafka_m:
                    if not self._ensure_kafka_pipeline():
                        kafka_enabled = False
                        retry_sleep = min(retry_sleep * 2, 10)
                    else:
                        retry_sleep = 1

                messages_processed = 0
                for queue_obj, topic in queues_and_topics:
                    if self.process_events_queue(queue_obj, topic, api_m, kafka_enabled):
                        messages_processed += 1

                if messages_processed == 0:
                    consecutive_empty_cycles += 1
                    time.sleep(min(1 * consecutive_empty_cycles, 5))
                else:
                    consecutive_empty_cycles = 0
                    time.sleep(0.01)
            except (KafkaError, NoBrokersAvailable) as e:
                print(f"DEBUG: Kafka connection error in main loop: {e}")
                self._handle_broker_failure()
                time.sleep(1)
            except Exception as e:
                print(f"DEBUG: Unexpected error in Kafka loop: {e}")
                time.sleep(1)

        print("DEBUG: Kafka loop stopped due to shutdown request")

    # ------------------------- Close handler -------------------------
    def close(self):
        """Gracefully shutdown the Kafka handler"""
        print("DEBUG: Closing KafkaHandler...")
        self._stop_event.set()
        
        try:
            # Wait for health monitor thread to finish
            if self._health_thread and self._health_thread.is_alive():
                self._health_thread.join(timeout=5)
                print("DEBUG: Health monitor thread stopped")

            # Shutdown thread pool executor properly - NO TIMEOUT
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
                print("DEBUG: ThreadPoolExecutor closed")

            # Close Kafka producer
            if self.kafka_pipeline:
                try:
                    self._force_flush()
                    print("DEBUG: Kafka producer flushed")
                except Exception as e:
                    print(f"DEBUG: Error flushing Kafka producer: {e}")
                try:
                    self.kafka_pipeline.close()
                    print("DEBUG: Kafka producer closed")
                except Exception as e:
                    print(f"DEBUG: Error closing Kafka producer: {e}")
                finally:
                    self.kafka_pipeline = None

            # S3 clients - NO client.close() calls removed
            print("DEBUG: S3 clients left open for reuse")

            print("DEBUG: KafkaHandler closed successfully")
        except Exception as e:
            print(f"DEBUG: Error during close: {e}")

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested"""
        return self._shutdown_requested or self._stop_event.is_set()

    def request_shutdown(self):
        """Request graceful shutdown"""
        print("DEBUG: Shutdown requested")
        self._shutdown_requested = True
        self._stop_event.set()