"""Download creditcard.csv (Kaggle ULB credit card fraud dataset, mirrored on GitHub)
into data/. If this fails, download it manually from Kaggle (see README)."""
import sys
import urllib.request
from pathlib import Path

URL = ("https://raw.githubusercontent.com/nsethi31/"
       "Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv")
dest = Path(__file__).parent / "data" / "creditcard.csv"
dest.parent.mkdir(exist_ok=True)
if dest.exists():
    print(f"{dest} already exists - nothing to do.")
    sys.exit(0)
print("Downloading (~100 MB)...")
urllib.request.urlretrieve(URL, dest)
print(f"Saved to {dest}")
