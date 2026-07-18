import urllib.request
import zipfile
import io
import os

print('Downloading radar.zip...')
url = 'https://d3ehgyu1hepsur.cloudfront.net/TruckDrive/scene_28_1/radar.zip'
response = urllib.request.urlopen(url)
print('Extracting radar.zip...')
with zipfile.ZipFile(io.BytesIO(response.read())) as z:
    z.extractall('data/TruckDrive/scene_28_1')
print('Done!')
