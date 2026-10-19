#!/usr/bin/env bash
# Download EuroSAT_MS (all 13 Sentinel-2 bands, 2.1 GB) from Zenodo,
# check the file against its published MD5 checksum and unpack it into data/.
set -euo pipefail

URL="https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1"
MD5="091174add3c8e680a49244acf185b9f0"

mkdir -p data
cd data
wget -c -O EuroSAT_MS.zip "$URL"
echo "$MD5  EuroSAT_MS.zip" | md5sum -c -
unzip -q -n EuroSAT_MS.zip
rm EuroSAT_MS.zip
echo "GeoTIFF tiles found: $(find . -name '*.tif' | wc -l)"
