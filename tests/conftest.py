import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


@pytest.fixture
def fake_eurosat(tmp_path):
    """A tiny EuroSAT look-alike: 3 class folders with 10 GeoTIFFs of 13 bands each."""
    rng = np.random.default_rng(0)
    for c, name in enumerate(["Forest", "Highway", "River"]):
        folder = tmp_path / name
        folder.mkdir()
        for i in range(10):
            image = rng.normal(1000 + 500 * c, 100, size=(13, 64, 64)).clip(0).astype("uint16")
            with rasterio.open(folder / f"{name}_{i + 1}.tif", "w", driver="GTiff",
                               width=64, height=64, count=13, dtype="uint16",
                               crs="EPSG:32632", transform=from_origin(500000, 5500000, 10, 10)) as dst:
                dst.write(image)
    return tmp_path
