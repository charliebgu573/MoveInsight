# MoveInsight Server

## Setup

### Create Conda Environment

```bash
conda create -n moveinsight python=3.9
conda activate moveinsight
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start Server

```bash
python analysis_server.py
```

## Connecting iOS App

When the MoveInsight iOS app prompts for a server address:

1. Visit [whatismyipaddress.com](https://whatismyipaddress.com/) on your computer
2. Copy the IPv4 address displayed
3. Enter this IP address in the iOS app server settings

### Troubleshooting Connection Issues

If you're having connectivity problems:
- Try turning on mobile data on your phone instead of using WiFi
