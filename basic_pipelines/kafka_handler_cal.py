import io
import json
import time
import uuid
import queue
import boto3
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from video_clipper import VideoClipRecorder
from boto3.s3.transfer import TransferConfig
from botocore.client import Config 

# S3 Transfer configuration for multipart uploads (used for video files)
S3_TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=5*1024*1024,  # 5MB
    multipart_chunksize=5*1024*1024,
    max_concurrency=4
)


class KafkaHandler:
    """Handles Kafka message production and S3 uploads with dual broker and dual S3 redundancy."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Kafka handler with configuration.
        
        Args:
            config: Configuration dictionary containing Kafka and AWS settings
        """
        self.config = config
        self.kafka_pipeline = None
        self.s3_clients = {}
        self.recorder = None
        self.last_error_time = 0
        self.error_interval = 300
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Flush tracking for smart flushing
        self.last_flush_time = time.time()
        self.messages_since_flush = 0
        self.flush_interval = 20  # Flush every 5 seconds
        self.flush_threshold = 10  # Or after 10 messages
        
        # Dual broker redundancy settings
        self.brokers = self._get_broker_list()
        self.current_broker_index = 0
        self.broker_failover_timeout = config.get("kafka_variables", {}).get("broker_failover_timeout", 30)
        self.last_broker_failure = 0
        self.broker_health = {broker: True for broker in self.brokers}
        
        # Dual S3 redundancy settings
        self.s3_configs = self._get_s3_configs()
        self.current_s3_index = 0
        self.s3_failover_timeout = config.get("kafka_variables", {}).get("AWS_S3", {}).get("s3_failover_timeout", 30)
        self.last_s3_failure = 0
        self.s3_health = {name: True for name in self.s3_configs.keys()}
        
        print(f"DEBUG: Initialized with {len(self.brokers)} brokers and {len(self.s3_configs)} S3 buckets")
        
        self._setup_aws_s3()
        self._setup_video_recorder()
        
    def _get_broker_list(self) -> List[str]:
        """Get list of brokers from configuration."""
        kafka_config = self.config.get("kafka_variables", {})
        
        bootstrap_servers = kafka_config.get("bootstrap_servers")
        if isinstance(bootstrap_servers, list):
            print("List of Bootstrap_servers",bootstrap_servers)
            return bootstrap_servers
        
        primary = kafka_config.get("primary_broker")
        secondary = kafka_config.get("secondary_broker")
        
        brokers = []
        if primary:
            brokers.append(primary)
        if secondary:
            brokers.append(secondary)
        
        return brokers if brokers else ["localhost:9092"]
        
    def _get_s3_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get S3 configurations from config."""
        aws_config = self.config.get("kafka_variables", {}).get("AWS_S3", {})
        
        s3_configs = {}
        
        # Check for new dual S3 format
        if "primary" in aws_config:
            s3_configs["primary"] = aws_config["primary"]
        if "secondary" in aws_config:
            s3_configs["secondary"] = aws_config["secondary"]
        
        # Fallback to old single S3 format
        if not s3_configs and "BUCKET_NAME" in aws_config:
            s3_configs["primary"] = aws_config
        
        return s3_configs
        
    def _setup_aws_s3(self):
        """Initialize AWS S3 clients for all configured buckets."""
        for name, config in self.s3_configs.items():
            try:
                client = boto3.client(
                    "s3",
                    aws_access_key_id=config.get("aws_access_key_id"),
                    endpoint_url=f"http://{config.get('end_point_url')}",
                    config=Config(signature_version="s3v4"),
                    aws_secret_access_key=config.get("aws_secret_access_key"),
                    region_name=config.get("region_name")
                )
                self.s3_clients[name] = client
                print(f"DEBUG: Initialized S3 client for {name}: {config.get('BUCKET_NAME')}")
            except Exception as e:
                print(f"DEBUG: Failed to initialize S3 client for {name}: {e}")
                self.s3_health[name] = False
        
    def _setup_video_recorder(self):
        """Initialize video recorder for frame buffering."""
        self.recorder = VideoClipRecorder(
            maxlen=60,
            fps=20,
            prefix="clips"
        )
        
    def _test_s3_connectivity(self, s3_name: str) -> bool:
        """Test if an S3 bucket is reachable."""
        try:
            client = self.s3_clients.get(s3_name)
            if not client:
                return False
            
            config = self.s3_configs[s3_name]
            bucket_name = config.get("BUCKET_NAME")
            
            # Test by listing objects (limited to 1)
            client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
            return True
        except Exception:
            return False
            
    def _get_next_healthy_s3(self) -> Optional[str]:
        """Get the next healthy S3 bucket in round-robin fashion."""
        current_time = time.time()
        
        # Check if enough time has passed to retry failed S3 buckets
        if current_time - self.last_s3_failure > self.s3_failover_timeout:
            for s3_name in self.s3_configs.keys():
                if not self.s3_health[s3_name]:
                    self.s3_health[s3_name] = self._test_s3_connectivity(s3_name)
        
        # Find healthy S3 buckets
        healthy_s3 = [name for name in self.s3_configs.keys() if self.s3_health[name]]
        
        if not healthy_s3:
            return None
            
        # Round-robin selection
        s3_name = healthy_s3[self.current_s3_index % len(healthy_s3)]
        self.current_s3_index += 1
        return s3_name
        
    def _test_broker_connectivity(self, broker: str) -> bool:
        """Test if a broker is reachable."""
        try:
            test_producer = KafkaProducer(
                bootstrap_servers=str(broker),
                request_timeout_ms=5000,
                max_block_ms=2000
            )
            test_producer.close()
            return True
        except Exception:
            return False
            
    def _get_next_healthy_broker(self) -> Optional[str]:
        """Get the next healthy broker in round-robin fashion."""
        current_time = time.time()
        
        if current_time - self.last_broker_failure > self.broker_failover_timeout:
            for broker in self.brokers:
                if not self.broker_health[broker]:
                    self.broker_health[broker] = self._test_broker_connectivity(broker)
        
        healthy_brokers = [broker for broker in self.brokers if self.broker_health[broker]]
        
        if not healthy_brokers:
            return None
            
        broker = healthy_brokers[self.current_broker_index % len(healthy_brokers)]
        
        self.current_broker_index += 1
        return broker
        
    def _create_kafka_producer(self) -> Optional[KafkaProducer]:
        """Create and configure Kafka producer with dual broker redundancy."""
        broker = self._get_next_healthy_broker()
        if not broker:
            print("DEBUG: No healthy brokers available")
            return None
            
        try:
            kafka_config = self.config.get("kafka_variables", {})
            linger = int(kafka_config.get("linger_ms", 50))
            batch = int(kafka_config.get("batch_size", 512*1024))
            
            producer = KafkaProducer(
                bootstrap_servers=str(broker),
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3,
                retry_backoff_ms=500,
                compression_type='gzip',
                batch_size=batch,
                buffer_memory=67108864,
                max_request_size=1048576,
                request_timeout_ms=5000,
                max_block_ms=1000,
                linger_ms=linger
            )
            
            producer.metrics()
            print(f"DEBUG: Connected to broker: {broker}")
            return producer
            
        except Exception as e:
            print(f"DEBUG: Failed to create Kafka producer with broker {broker}: {e}")
            self.broker_health[broker] = False
            self.last_broker_failure = time.time()
            return None
            
    def _handle_broker_failure(self):
        """Handle broker failure and switch to next available broker."""
        if self.kafka_pipeline:
            try:
                self.kafka_pipeline.close()
            except:
                pass
            finally:
                self.kafka_pipeline = None
                
        self.kafka_pipeline = self._create_kafka_producer()
        
    def _smart_flush(self):
        """Smart flush that balances performance and reliability."""
        if not self.kafka_pipeline:
            return
            
        current_time = time.time()
        self.messages_since_flush += 1
        
        # Flush if:
        # 1. Time interval exceeded, OR
        # 2. Message threshold reached, OR
        # 3. This is a critical message (you can add logic here)
        should_flush = (
            (current_time - self.last_flush_time) >= self.flush_interval or
            self.messages_since_flush >= self.flush_threshold
        )
        
        if should_flush:
            try:
                self.kafka_pipeline.flush(timeout=5)
                self.last_flush_time = current_time
                self.messages_since_flush = 0
                #print(f"DEBUG: Smart flush completed - {self.messages_since_flush} messages in {current_time - self.last_flush_time:.2f}s")
            except Exception as e:
                print(f"DEBUG: Smart flush failed: {e}")
        
    def _force_flush(self):
        """Force immediate flush for critical messages."""
        if self.kafka_pipeline:
            try:
                self.kafka_pipeline.flush(timeout=10)
                self.last_flush_time = time.time()
                self.messages_since_flush = 0
                #print("DEBUG: Force flush completed")
            except Exception as e:
                print(f"DEBUG: Force flush failed: {e}")
        
    def upload_to_s3(self, file_bytes: bytes, file_type: str = "image", retries: int = 2, delay: int = 1) -> Optional[str]:
        """Upload file bytes to S3 with dual bucket redundancy."""
        upload_retries = self.config.get("kafka_variables", {}).get("AWS_S3", {}).get("upload_retries", 3)
        
        # Set content type and filename based on file type
        if file_type == "video":
            unique_filename = f"clips{uuid.uuid4()}.mp4"
            content_type = "video/mp4"
        else:
            unique_filename = f"{uuid.uuid4()}.jpg"
            content_type = "image/jpg"
        
        # Try each S3 bucket with retries
        for s3_name in self.s3_configs.keys():
            if not self.s3_health[s3_name]:
                continue
                
            client = self.s3_clients.get(s3_name)
            config = self.s3_configs[s3_name]
            
            if not client or not config:
                continue
            
            for attempt in range(upload_retries):
                try:
                    if file_type == "video":
                        # Use upload_fileobj for video files
                        client.upload_fileobj(
                            io.BytesIO(file_bytes),
                            Bucket=config.get("BUCKET_NAME"),
                            Key=f"{config.get('video_fn')}{unique_filename}",
                            ExtraArgs={"ContentType": content_type},
                            Config=S3_TRANSFER_CONFIG
                        )
                        
                        minio_url = f"http://{config.get('end_point_url')}/{config.get('BUCKET_NAME')}/{config.get('video_fn')}{unique_filename}"
                        #print(f"DEBUG: Successfully uploaded video to {s3_name}: {minio_url}")
                        return minio_url
                    elif file_type == "image":
                        # Use put_object for images
                        client.put_object(
                            Bucket=config.get("BUCKET_NAME"),
                            Key=f"{config.get('org_img_fn')}{unique_filename}",
                            Body=file_bytes,
                            ContentType=content_type
                        )
                        minio_url = f"http://{config.get('end_point_url')}/{config.get('BUCKET_NAME')}/{config.get('org_img_fn')}{unique_filename}"
                        #print(f"DEBUG: Successfully uploaded video to {s3_name}: {minio_url}")
                        return minio_url
                    
                    elif file_type == "snapshot":
                        client.put_object(
                            Bucket=config.get("BUCKET_NAME"),
                            Key=f"{config.get('cgi_fn')}{unique_filename}",
                            Body=file_bytes,
                            ContentType=content_type
                        )
                        minio_url = f"http://{config.get('end_point_url')}/{config.get('BUCKET_NAME')}/{config.get('cgi_fn')}{unique_filename}"
                        #print(f"DEBUG: Successfully uploaded video to {s3_name}: {minio_url}")
                        return minio_url
                        
                except Exception as e:
                    print(f"DEBUG: S3 {file_type} upload attempt {attempt + 1} to {s3_name} failed: {e}")
                    if attempt < upload_retries - 1:
                        time.sleep(delay)
            
            # Mark S3 as unhealthy if all attempts failed
            self.s3_health[s3_name] = False
            self.last_s3_failure = time.time()
                    
        return None
    
    def _upload_image(self, image_bytes: bytes) -> tuple[str, Optional[str]]:
        """Upload image to S3."""
        return ("org_img", self.upload_to_s3(image_bytes, "image") if image_bytes else None)
    
    def _upload_snapshot(self, snap_shot_bytes: bytes) -> tuple[str, Optional[str]]:
        """Upload snapshot to S3 or return DISABLED if None."""
        if snap_shot_bytes is None:
            return ("snap_shot", "DISABLED")
        return ("snap_shot", self.upload_to_s3(snap_shot_bytes, "snapshot") if snap_shot_bytes else None)
    
    def _upload_video(self, video_bytes: bytes) -> tuple[str, Optional[str]]:
        """Upload video to S3."""
        return ("video", self.upload_to_s3(video_bytes, "video") if video_bytes else None)
    
    def process_events_queue(self, events_queue: queue.Queue, topic: str) -> bool:
        #print("DEBUG","Running the Process Events Queue")
        """Process events from queue and send to Kafka with dual broker and S3 redundancy."""
        try:
            message = events_queue.get(timeout=1)  # waits up to 1 seconds
            if message is None or topic == "None":
                return True
                
            # # Upload files to S3 in parallel and allow partial success
            # image_bytes = message.get("org_img")
            # snap_shot_bytes = message.get("snap_shot")
            # video_bytes = message.get("video")

            # uploads = {}
            # futures = [
            #     self.executor.submit(self._upload_image, image_bytes),
            #     self.executor.submit(self._upload_snapshot, snap_shot_bytes),
            #     self.executor.submit(self._upload_video, video_bytes)
            # ]
            # for fut in as_completed(futures):
            #     k, v = fut.result()
            #     uploads[k] = v


            # success = False
            # # Check if all required uploads succeeded (treat "DISABLED" as success for snap_shot)
            # uploads_successful = (
            #     uploads.get("org_img") is not None and 
            #     uploads.get("video") is not None and
            #     (uploads.get("snap_shot") is not None or uploads.get("snap_shot") == "DISABLED")
            # )

            # # Update message with available S3 URLs
            # message["org_img"] = uploads.get("org_img")
            # # Set snap_shot to None if it was disabled, otherwise use the uploaded URL
            # message["snap_shot"] = None if uploads.get("snap_shot") == "DISABLED" else uploads.get("snap_shot")
            # message["video"] = uploads.get("video")

            # # Send to Kafka only if ALL uploads succeeded
            # if uploads_successful and self.kafka_pipeline:
            #     try:
            #         #print("DEBUG",topic,message,self.kafka_pipeline)
            #         future = self.kafka_pipeline.send(topic, message)
                    
            #         # Wait for the message to be sent (with timeout)
            #         record_metadata = future.get(timeout=10)
            #         print(f"DEBUG: Message sent successfully to partition {record_metadata.partition} at offset {record_metadata.offset}")
                    
            #         # Smart flush - only when needed
            #         self._smart_flush()
            #         success = True
            #         print("DEBUG: Data Sent in Kafka Successfully")
                    
            #     except (KafkaError, NoBrokersAvailable) as e:
            #         print(f"DEBUG: Kafka send error: {e}")
            #         self._handle_broker_failure()
            #         if self.kafka_pipeline:
            #             try:
            #                 future = self.kafka_pipeline.send(topic, message)
            #                 record_metadata = future.get(timeout=10)
            #                 self._smart_flush()
            #                 success = True
            #                 print("DEBUG: Message sent successfully after broker failover")
            #             except Exception as retry_e:
            #                 print(f"DEBUG: Retry send failed: {retry_e}")
            #                 pass
				
            return None
                
        except queue.Empty:
            return True
        except Exception as e:
            print(f"DEBUG: Events queue processing error: {e}")
            return False
            
    def process_analytics_queue(self, analytics_queue: queue.Queue, topic: str) -> bool:
        """Process analytics from queue and send to Kafka with dual broker redundancy."""
        try:
            message = analytics_queue.get_nowait()
            if message is None or topic == "None":
                return True
                
            if self.kafka_pipeline:
                try:
                    future = self.kafka_pipeline.send(topic, message)
                    record_metadata = future.get(timeout=10)
                    self._smart_flush()
                    print(f"DEBUG: Analytics message sent successfully to partition {record_metadata.partition}")
                    return True
                except (KafkaError, NoBrokersAvailable) as e:
                    print(f"DEBUG: Kafka analytics send failed: {e}")
                    self._handle_broker_failure()
                    if self.kafka_pipeline:
                        try:
                            future = self.kafka_pipeline.send(topic, message)
                            record_metadata = future.get(timeout=10)
                            self._smart_flush()
                            print("DEBUG: Analytics message sent successfully after broker failover")
                            return True
                        except Exception as retry_e:
                            print(f"DEBUG: Analytics retry send failed: {retry_e}")
                            pass
            
            return False
                
        except queue.Empty:
            return True
        except Exception as e:
            print(f"DEBUG: Analytics queue processing error: {e}")
            return False
            
    def send_error_log(self, error_message: str, error_details: str = None, sensor_id: str = None):
        """Send error log to Kafka with dual broker redundancy."""
        current_time = time.time()
        
        if current_time - self.last_error_time < self.error_interval:
            return
            
        try:
            if self.kafka_pipeline is None:
                return
                
            log_message = {
                "timestamp": datetime.now().isoformat(),
                "level": "ERROR",
                "message": error_message,
                "sensor_id": sensor_id,
                "details": error_details,
                "rate_limited": True
            }
            
            log_topic = self.config.get("kafka_variables", {}).get("log_topic", "log_topic")
            
            try:
                self.kafka_pipeline.send(log_topic, log_message)
                self.last_error_time = current_time
            except (KafkaError, NoBrokersAvailable):
                self._handle_broker_failure()
                if self.kafka_pipeline:
                    try:
                        self.kafka_pipeline.send(log_topic, log_message)
                        self.last_error_time = current_time
                    except:
                        pass
                        
        except Exception as e:
            print(f"DEBUG: Kafka error logging failed: {e}")
            
    def run_kafka_loop(self, events_queue: queue.Queue, analytics_queue: queue.Queue):
        """Main Kafka processing loop with dual broker and S3 redundancy."""
        kafka_config = self.config.get("kafka_variables", {})
        send_events_pipeline = kafka_config.get("send_events_pipeline")
        send_analytics_pipeline = kafka_config.get("send_analytics_pipeline")
        
        queues_and_topics = [
            (events_queue, send_events_pipeline)
        ]
        
        consecutive_empty_cycles = 0
        
        print(f"DEBUG: Starting Kafka loop with dual brokers: {self.brokers}")
        print(f"DEBUG: S3 buckets: {[config.get('BUCKET_NAME') for config in self.s3_configs.values()]}")
        
        while True:
            try:
                # print("DEBUG","Yes Kafka Loop running")
                if self.kafka_pipeline is None:
                    self.kafka_pipeline = self._create_kafka_producer()
                    if self.kafka_pipeline is None:
                        print("DEBUG: No healthy brokers available, retrying in 10 seconds")
                        time.sleep(10)
                        continue
                    time.sleep(5)
                
                messages_processed = 0
                for queue_obj, topic in queues_and_topics:
                    if topic == send_events_pipeline:
                        if self.process_events_queue(queue_obj, topic):
                            messages_processed += 1
                    else:
                        if self.process_analytics_queue(queue_obj, topic):
                            messages_processed += 1
                
                if messages_processed == 0:
                    consecutive_empty_cycles += 1
                    sleep_time = min(2 * consecutive_empty_cycles, 10)
                    time.sleep(sleep_time)
                else:
                    consecutive_empty_cycles = 0
                    time.sleep(0.1)
                    
            except (KafkaError, NoBrokersAvailable) as e:
                print(f"DEBUG: Kafka connection error: {e}")
                self._handle_broker_failure()
                time.sleep(5)
            except Exception as e:
                print(f"DEBUG: Unexpected error in Kafka loop: {e}")
                time.sleep(5)
    
    def close(self):
        """Gracefully close the Kafka handler and cleanup resources."""
        print("DEBUG: Closing Kafka handler...")
        
        try:
            # Close the executor
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
                print("DEBUG: ThreadPoolExecutor closed")
            
            # Flush and close Kafka producer
            if self.kafka_pipeline:
                try:
                    self._force_flush()  # Force flush all pending messages
                    print("DEBUG: Kafka producer flushed")
                except Exception as e:
                    print(f"DEBUG: Error flushing Kafka producer: {e}")
                
                try:
                    self.kafka_pipeline.close(timeout=10)
                    print("DEBUG: Kafka producer closed")
                except Exception as e:
                    print(f"DEBUG: Error closing Kafka producer: {e}")
                finally:
                    self.kafka_pipeline = None
            
            # Close S3 clients
            for name, client in self.s3_clients.items():
                try:
                    client.close()
                    print(f"DEBUG: S3 client {name} closed")
                except Exception as e:
                    print(f"DEBUG: Error closing S3 client {name}: {e}")
            
            print("DEBUG: Kafka handler closed successfully")
            
        except Exception as e:
            print(f"DEBUG: Error during Kafka handler close: {e}")
