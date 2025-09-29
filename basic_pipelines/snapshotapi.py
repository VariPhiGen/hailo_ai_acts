import time
import requests
from requests.auth import HTTPDigestAuth
import datetime

class Snapshot:
    def __init__(self, camera_ip, user, pwd, ch=1, timeout=2, retries=3, backoff=0.5):
        self.url = f"http://{camera_ip}/cgi-bin/snapshot.cgi?chn={ch}"
        self.auth = HTTPDigestAuth(user, pwd)
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def capture(self, prefix="snap"):
        ts = time.time()
        fname = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')}_{prefix}.jpg"
        for attempt in range(1, self.retries + 1):
            try:
                r = requests.get(self.url, auth=self.auth, timeout=self.timeout)
                r.raise_for_status()
                #with open(fname, "wb") as f:
                 # f.write(r.content)
                return fname,r.content
            except Exception as e:
                print(f"⚠️ Attempt {attempt} failed:", e)
                time.sleep(self.backoff * attempt)
        print("❌ All attempts failed.")
        print(time.time() - ts)
        return None,None
