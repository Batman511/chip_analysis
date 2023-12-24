## Description
This project is a tool for detecting anomalies on chips. Anomalies may include malfunctions, overheating, or other unusual events that may cause the chips to malfunction or malfunction.

## Pipeline anomaly detection

- **Canny for borders detection**
- **DBSCAN for detect main microcircuit**
- **Finding rectangle of minimum area**
- **Gaussian blur**
- **Find differences by SSIM**

## Example predictions

<img src="https://github.com/Fruha/ASE_CHIPS/blob/master/git_images/example.png" width="60%">

## Installation
### WINDOWS
```bash
git clone https://github.com/Fruha/ASE_CHIPS
cd ASE_CHIPS
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
```
## Usage/Visualization

open main.ipynb
