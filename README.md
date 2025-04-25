
# Fall Detection System using YOLOv8

This project is a fall detection system that uses computer vision powered by a custom-trained **YOLOv8** model. It detects when a person falls and sends alerts via WhatsApp along with the user's approximate geolocation.

## 🚀 Features

- Real-time person detection using YOLOv8
- Fall detection with time-based tracking (10 seconds threshold)
- Automatic alerts via WhatsApp
- Location fetching via IP geolocation
- Modular codebase with separate scripts for detection and alerting

## 🧠 Model

This project includes a **custom-trained YOLOv8 model** (`.pt` file) to accurately detect persons and assist in fall detection scenarios.

## 📁 File Overview

| File            | Description |
|------------------|-------------|
| `main2.py`       | Main detection + alert system script |
| `10secfall.py`   | Monitors for fall events lasting 10 seconds |
| `alert1.py`      | Sends WhatsApp alerts and gets location |
| `classes.txt`    | YOLO class labels |
| `requirements.txt` | Dependencies to run the project |

## 🛠 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Make sure to also have:
- A working webcam or video input
- WhatsApp Web logged in on your system (for `pywhatkit`)
- Internet connection (for geolocation and messaging)

## 📦 Running the Code

```bash
python main2.py
```

## 📬 Alerts

Alerts are sent through WhatsApp using `pywhatkit` and the user’s approximate location is fetched using `geocoder`.

## 🤖 Model Training (Optional)

If you'd like to train your own YOLOv8 model, check out [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/).

---

### 🔒 Disclaimer

WhatsApp automation via `pywhatkit` may not work properly on all systems and can be impacted by WhatsApp Web updates. Use responsibly.

