import io
import json
import time
import uuid
import queue
import threading
import signal
import sys
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.client import Config
from video_clipper import VideoClipRecorder

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
        self.kafka_pipeline: Optional[KafkaProducer] = None
        self.s3_clients: Dict[str, Any] = {}
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

        # S3 configs
        self.s3_configs = self._get_s3_configs()
        self.current_s3_index = 0
        self.s3_health = {name: True for name in self.s3_configs.keys()}

        # Health check
        self.health_check_interval = int(self.config.get("kafka_variables", {}).get("health_check_interval", 15))
        self._health_thread = None

        # Setup AWS S3 and Video Recorder
        self._setup_aws_s3()
        self._setup_video_recorder()
        self._start_health_monitor()
        self._setup_signal_handlers()

    # ------------------------- Setup helpers -------------------------
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
        aws_config = self.config.get("kafka_variables", {}).get("AWS_S3", {})
        s3_configs: Dict[str, Dict[str, Any]] = {}
        if "primary" in aws_config:
            s3_configs["primary"] = aws_config["primary"]
        if "secondary" in aws_config:
            s3_configs["secondary"] = aws_config["secondary"]
        if not s3_configs and "BUCKET_NAME" in aws_config:
            s3_configs["primary"] = aws_config
        return s3_configs

    def _setup_aws_s3(self):
        for name, config in self.s3_configs.items():
            try:
                client = boto3.client(
                    "s3",
                    aws_access_key_id=config.get("aws_access_key_id"),
                    aws_secret_access_key=config.get("aws_secret_access_key"),
                    region_name=config.get("region_name"),
                    endpoint_url=f"http://{config.get('end_point_url')}" if config.get("end_point_url") else None,
                    config=Config(signature_version="s3v4")
                )
                self.s3_clients[name] = client
                print(f"DEBUG: Initialized S3 client for {name}: {config.get('BUCKET_NAME')}")
            except Exception as e:
                print(f"DEBUG: Failed to init S3 client {name}: {e}")
                self.s3_health[name] = False

    def _setup_video_recorder(self):
        try:
            self.recorder = VideoClipRecorder(maxlen=60, fps=20, prefix="clips")
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

    def _test_broker_connectivity(self, broker: str) -> bool:
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

    def _create_kafka_producer(self, max_attempts=3) -> Optional[KafkaProducer]:
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
        self.kafka_pipeline = self._create_kafka_producer()

    # ------------------------- S3 upload helpers -------------------------
    def _get_next_healthy_s3(self) -> Optional[str]:
        healthy = [name for name, ok in self.s3_health.items() if ok]
        if not healthy:
            return None
        name = healthy[self.current_s3_index % len(healthy)]
        self.current_s3_index += 1
        return name

    def upload_to_s3(self, file_bytes: bytes, file_type: str = "image") -> Optional[str]:
        """Upload file to S3 with retries and failover"""
        if not file_bytes:
            return None

        upload_retries = int(self.config.get("kafka_variables", {}).get("AWS_S3", {}).get("upload_retries", 3))
        tried = set()

        # Try healthy S3 buckets first
        for _ in range(max(1, len(self.s3_configs))):
            s3_name = self._get_next_healthy_s3()
            if not s3_name or s3_name in tried:
                break
            tried.add(s3_name)
            
            client = self.s3_clients.get(s3_name)
            config = self.s3_configs.get(s3_name)
            if not client or not config:
                self.s3_health[s3_name] = False
                continue

            unique_filename = (f"clips{uuid.uuid4()}.mp4" if file_type == "video" else f"{uuid.uuid4()}.jpg")
            content_type = ("video/mp4" if file_type == "video" else "image/jpg")

            for attempt in range(upload_retries):
                try:
                    if file_type == "video":
                        client.upload_fileobj(
                            io.BytesIO(file_bytes),
                            Bucket=config.get("BUCKET_NAME"),
                            Key=f"{config.get('video_fn', '')}{unique_filename}",
                            ExtraArgs={"ContentType": content_type},
                            Config=S3_TRANSFER_CONFIG,
                        )
                        url = f"http://{config.get('end_point_url')}/{config.get('BUCKET_NAME')}/{config.get('video_fn', '')}{unique_filename}"
                        return url
                    else:
                        key_prefix = config.get('org_img_fn', '') if file_type == "image" else config.get('cgi_fn', '')
                        client.put_object(
                            Bucket=config.get("BUCKET_NAME"),
                            Key=f"{key_prefix}{unique_filename}",
                            Body=file_bytes,
                            ContentType=content_type,
                        )
                        url = f"http://{config.get('end_point_url')}/{config.get('BUCKET_NAME')}/{key_prefix}{unique_filename}"
                        return url
                except Exception as e:
                    print(f"DEBUG: S3 {file_type} upload attempt {attempt + 1} to {s3_name} failed: {e}")
                    time.sleep(0.5 * (attempt + 1))
            
            # Mark as unhealthy after retries
            self.s3_health[s3_name] = False

        # Last-ditch attempt with any S3 bucket
        for name, config in self.s3_configs.items():
            if name in tried:
                continue
            client = self.s3_clients.get(name)
            try:
                unique_filename = (f"clips{uuid.uuid4()}.mp4" if file_type == "video" else f"{uuid.uuid4()}.jpg")
                content_type = ("video/mp4" if file_type == "video" else "image/jpg")
                
                if file_type == "video":
                    client.upload_fileobj(
                        io.BytesIO(file_bytes),
                        Bucket=config.get("BUCKET_NAME"),
                        Key=f"{config.get('video_fn', '')}{unique_filename}",
                        ExtraArgs={"ContentType": content_type},
                        Config=S3_TRANSFER_CONFIG
                    )
                    return f"http://{config.get('end_point_url')}/{config.get('BUCKET_NAME')}/{config.get('video_fn', '')}{unique_filename}"
                else:
                    key_prefix = config.get('org_img_fn', '') if file_type == "image" else config.get('cgi_fn', '')
                    client.put_object(
                        Bucket=config.get("BUCKET_NAME"),
                        Key=f"{key_prefix}{unique_filename}",
                        Body=file_bytes,
                        ContentType=content_type
                    )
                    return f"http://{config.get('end_point_url')}/{config.get('BUCKET_NAME')}/{key_prefix}{unique_filename}"
            except Exception as e:
                print(f"DEBUG: Last-ditch upload to {name} failed: {e}")
                continue

        return None

    def upload_to_s3_safe(self, file_bytes: bytes, file_type: str = "image") -> Optional[str]:
        """Non-blocking S3 upload with retries."""
        try:
            if file_bytes:
                return self.upload_to_s3(file_bytes, file_type)
        except Exception as e:
            print(f"DEBUG: S3 upload failed for {file_type}: {e}")
        return None

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
    def process_events_queue(self, events_queue: queue.Queue, topic: str) -> bool:
        try:
            message = events_queue.get(timeout=1)
            if message is None or topic == "None":
                return True

            image_bytes = message.get("org_img")
            snap_shot_bytes = message.get("snap_shot")
            video_bytes = message.get("video")

            uploads = {}
            futures = {
                self.executor.submit(self.upload_to_s3_safe, image_bytes, "image"): "org_img",
                self.executor.submit(self.upload_to_s3_safe, snap_shot_bytes, "snapshot"): "snap_shot",
                self.executor.submit(self.upload_to_s3_safe, video_bytes, "video"): "video"
            }

            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    uploads[key] = fut.result()
                except Exception as e:
                    print(f"DEBUG: Upload task failed for {key}: {e}")
                    uploads[key] = None

            message["org_img"] = uploads.get("org_img")
            message["snap_shot"] = uploads.get("snap_shot")
            message["video"] = uploads.get("video")

            if message["org_img"] is not None:
                if not self._ensure_kafka_pipeline():
                    print("DEBUG: Kafka pipeline unavailable, skipping message")
                    return False
                try:
                    future = self.kafka_pipeline.send(topic, message)
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
            else:
                print("DEBUG: Insufficient uploads, message skipped")
                return False

        except queue.Empty:
            return True
        except Exception as e:
            print(f"DEBUG: Events queue processing error: {e}")
            return False

    # ------------------------- Main loop -------------------------
    def run_kafka_loop(self, events_queue: queue.Queue, analytics_queue: queue.Queue = None):
        """Main Kafka processing loop - runs indefinitely until shutdown is requested"""
        kafka_config = self.config.get("kafka_variables", {})
        send_events_pipeline = kafka_config.get("send_events_pipeline")
        queues_and_topics = [(events_queue, send_events_pipeline)]
        consecutive_empty_cycles = 0
        retry_sleep = 1

        print(f"DEBUG: Starting continuous Kafka loop with brokers: {self.brokers}")
        print(f"DEBUG: S3 buckets: {[config.get('BUCKET_NAME') for config in self.s3_configs.values()]}")

        while not self._stop_event.is_set():
            try:
                if not self._ensure_kafka_pipeline():
                    time.sleep(retry_sleep)
                    retry_sleep = min(retry_sleep * 2, 10)
                    continue
                retry_sleep = 1

                messages_processed = 0
                for queue_obj, topic in queues_and_topics:
                    if self.process_events_queue(queue_obj, topic):
                        messages_processed += 1

                if messages_processed == 0:
                    consecutive_empty_cycles += 1
                    time.sleep(min(0.5 * consecutive_empty_cycles, 5))
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