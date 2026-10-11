import os
import urllib.request

YOLO_DIR = os.path.dirname(os.path.abspath(__file__))

WEIGHTS_URLS = [
    "https://raw.githubusercontent.com/AlexeyAB/darknet/master/yolov3-tiny.weights",
    "https://pjreddie.com/media/files/yolov3-tiny.weights"
]
CFG_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
NAMES_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

WEIGHTS_FILE = os.path.join(YOLO_DIR, "yolov3-tiny.weights")
CFG_FILE = os.path.join(YOLO_DIR, "yolov3-tiny.cfg")
NAMES_FILE = os.path.join(YOLO_DIR, "coco.names")

def download_file(urls, target_path):
    if isinstance(urls, str):
        urls = [urls]

    if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
        print(f"Downloading {os.path.basename(target_path)}...")
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
                with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"Downloaded: {target_path}")
                return True
            except Exception as e:
                print(f"Attempt failed for {url}: {e}")
        return False
    else:
        print(f"Found existing file: {target_path}")
        return True

def setup_yolo_files():
    print("Checking YOLOv3 model files...")
    download_file(CFG_URL, CFG_FILE)
    download_file(NAMES_URL, NAMES_FILE)
    download_file(WEIGHTS_URLS, WEIGHTS_FILE)

if __name__ == "__main__":
    setup_yolo_files()
