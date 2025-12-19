## Quick Start

Clone the repo and work from its root directory:
```
git clone https://github.com/VariPhiGen/hailo_ai_acts.git && cd hailo_ai_acts
```

Run these commands in order from the project root:

1) Base system setup  
```
bash setup_system.sh
sudo reboot
```

2) Install project dependencies  
```
bash install.sh
```

3) Download required resources (models, etc.)  
```
bash download_resources.sh or bash download_hailo8l.sh
```

4) Load environment variables  
```
source setup_env.sh
```

5) Smoke test the simple detector  
```
python basic_pipelines/detection_simple.py
```

If you hit common errors, run the helper:
```
bash Common_Error.sh
```
6) AI environment setup  
```
bash setup_ai_env.sh
```

7) Write the default configuration file  
```
bash write_configuration.sh
```



If any script needs a custom path or args, pass them as needed (defaults assume project root). For `.env`, ensure it exists before sourcing `setup_env.sh`.  
#Variphi Acts on Hailo