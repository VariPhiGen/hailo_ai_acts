# Variphi SVDS System

A comprehensive AI pipeline system built for Hailo AI hardware, featuring computer vision applications, real-time processing, and cloud integration. The Variphi SVDS (Smart Video Detection System) provides robust, scalable solutions for edge AI applications.

## 🚀 Features

- **Advanced AI Pipelines**: Detection, depth estimation, face recognition, and more
- **Dual Redundancy Architecture**: Kafka brokers and S3 storage with automatic failover
- **Optimized Video Processing**: Frame buffering and video generation with efficient uploads
- **Community Projects**: Ready-to-use applications and examples
- **Easy Installation**: Automated setup scripts for Raspberry Pi and x86 systems
- **Real-time Processing**: Low-latency inference on Hailo AI hardware
- **Cloud Integration**: Seamless AWS S3 and Kafka integration

## 📁 Project Structure

```
Variphi-SVDS/
├── basic_pipelines/          # Core AI processing pipelines
│   ├── kafka_handler.py      # Dual Kafka/S3 redundancy handler
│   ├── video_clipper.py      # Video processing and frame buffering
│   ├── detection_simple.py   # Object detection pipeline
│   ├── depth.py             # Depth estimation pipeline
│   ├── face_recognition.py   # Face recognition pipeline
│   └── radar_handler.py     # Radar data processing
├── community_projects/       # Community-contributed applications
│   ├── traffic_sign_detection/
│   ├── fruit_ninja/
│   ├── Navigator/
│   ├── RoboChess/
│   └── TEMPO/
├── doc/                     # Documentation and guides
├── tests/                   # Test suites and validation
├── config.yaml              # Main configuration file
├── install.sh               # Main installation script
├── requirements.txt         # Python dependencies
└── detection.py             # Main detection application
```

## 🛠️ Installation

### Prerequisites
- Hailo AI hardware (Hailo-8 or Hailo-8L)
- Ubuntu 20.04+ or Raspberry Pi OS
- Python 3.8+
- GStreamer 1.0

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/VariPhiGen/SVDS.git
cd SVDS


# Setup system environment
source setup_system.sh
# Run installation
./install.sh


```

### Custom Installation
```bash
# Install with custom Hailo packages
./install.sh --pyhailort /path/to/custom/hailort.whl

# Skip installation, just setup
./install.sh --no-installation

# Install all resources
./install.sh --all
```

### System Environment Setup
After installation, set up the system environment:
```bash
# Setup system environment variables and paths
source setup_system.sh

# Or run the setup script directly
./setup_system.sh
```

**Note**: The `setup_system.sh` script configures:
- Environment variables for Hailo hardware
- Python path configurations
- GStreamer plugin paths
- System dependencies and permissions

## ⚙️ Configuration

Edit `config.yaml` to customize:
- Hailo hardware settings (`hailo_arch`, `host_arch`)
- Model versions and paths (`hailort_version`, `tappas_version`)
- Kafka and S3 configurations
- Virtual environment settings (`virtual_env_name`)
- Resource paths and storage directories

### Example Configuration
```yaml
# Hailo Hardware Configuration
hailo_arch: "hailo8l"  # or "hailo8"
host_arch: "rpi"       # or "x86"

# Model Versions
hailort_version: "auto"
tappas_version: "auto"
model_zoo_version: "v2.14.0"

# Storage and Resources
resources_path: "resources"
storage_dir: "hailo_temp_resources"
```

## 🎯 Usage

### Activate Environment
```bash
source setup_env.sh
```

### Run Main Detection Application
```bash
python detection.py
```

### Run Basic Detection Pipeline
```python
from basic_pipelines.detection_simple import DetectionPipeline
pipeline = DetectionPipeline()
pipeline.run()
```

### Use Kafka Handler with Dual Redundancy
```python
from basic_pipelines.kafka_handler import KafkaHandler
handler = KafkaHandler(config)
handler.run_kafka_loop(events_queue, analytics_queue)
```

### Video Processing with Frame Buffering
```python
from basic_pipelines.video_clipper import VideoClipRecorder
recorder = VideoClipRecorder(maxlen=100, fps=20)
recorder.add_frame(frame)
video_bytes = recorder.generate_video_bytes()
```

## 🔧 Advanced Features

### Dual S3 Redundancy
The system supports multiple S3 buckets with automatic failover:
```yaml
AWS_S3:
  primary:
    BUCKET_NAME: "primary-bucket"
    aws_access_key_id: "..."
    aws_secret_access_key: "..."
    region_name: "us-east-1"
  secondary:
    BUCKET_NAME: "backup-bucket"
    aws_access_key_id: "..."
    aws_secret_access_key: "..."
    region_name: "us-west-2"
```

### Dual Kafka Broker Support
Multiple Kafka brokers with health monitoring and failover:
```yaml
kafka_variables:
  bootstrap_servers: ["broker1:9092", "broker2:9092"]
  primary_broker: "broker1:9092"
  secondary_broker: "broker2:9092"
```

### Community Projects
Explore ready-to-use applications:
- **Traffic Sign Detection**: Real-time traffic sign recognition
- **Fruit Ninja**: AI-powered fruit slicing game
- **Navigator**: Autonomous navigation system
- **RoboChess**: AI chess playing robot
- **TEMPO**: Heart rate monitoring system

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
./run_tests.sh

# Run specific test
pytest tests/test_hailo_rpi5_examples.py

# Run with coverage
pytest --cov=basic_pipelines tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Write tests for new features
- Update documentation as needed

## 📊 Performance

### Benchmarks
- **Detection Latency**: < 50ms on Hailo-8L
- **Throughput**: 30+ FPS for 1080p video
- **Memory Usage**: Optimized for edge devices
- **Power Efficiency**: Designed for low-power operation

### Supported Models
- YOLOv5 (various sizes)
- SCRFD (face detection)
- ArcFace (face recognition)
- Depth estimation models
- Custom Hailo-optimized models

## 🆘 Support

### Documentation
- Check the [documentation](doc/) for detailed guides
- Review [installation guide](doc/install-raspberry-pi5.md)
- Explore [community projects](community_projects/)

### Troubleshooting
- Check [Hailo documentation](https://hailo.ai/developer-zone/)
- Review system requirements and dependencies
- Verify hardware compatibility
- Check logs in `hailort.log`

### Community
- Open an issue for bug reports
- Join our community discussions
- Share your projects and improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏗️ Built With

- [Hailo AI](https://hailo.ai/) - AI acceleration hardware
- [OpenCV](https://opencv.org/) - Computer vision library
- [GStreamer](https://gstreamer.freedesktop.org/) - Multimedia framework
- [Kafka](https://kafka.apache.org/) - Streaming platform
- [AWS S3](https://aws.amazon.com/s3/) - Cloud storage
- [PyGObject](https://pygobject.readthedocs.io/) - GStreamer Python bindings

## 🙏 Acknowledgments

- Hailo AI team for hardware and software support
- Open source community for various libraries and tools
- Contributors and users of the Variphi SVDS system

---

**Variphi SVDS System** - Empowering edge AI with intelligent video processing and real-time analytics. 