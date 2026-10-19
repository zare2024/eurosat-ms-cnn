# Test results: all (13 bands)

Test tiles: 4050 | Overall accuracy: 98.0%

| Class | Precision | Recall | F1 | Test tiles |
|---|---|---|---|---|
| AnnualCrop | 0.965 | 0.991 | 0.978 | 450 |
| Forest | 0.985 | 0.998 | 0.991 | 450 |
| HerbaceousVegetation | 0.963 | 0.980 | 0.971 | 450 |
| Highway | 0.978 | 0.949 | 0.963 | 375 |
| Industrial | 0.976 | 0.973 | 0.975 | 375 |
| Pasture | 0.986 | 0.963 | 0.975 | 300 |
| PermanentCrop | 0.983 | 0.952 | 0.967 | 375 |
| Residential | 0.980 | 0.991 | 0.986 | 450 |
| River | 0.989 | 0.984 | 0.987 | 375 |
| SeaLake | 0.996 | 1.000 | 0.998 | 450 |

## Most frequent mistakes

| True class | Predicted as | Tiles | Share of true class |
|---|---|---|---|
| PermanentCrop | AnnualCrop | 9 | 2.4% |
| Highway | Industrial | 8 | 2.1% |
| PermanentCrop | HerbaceousVegetation | 7 | 1.9% |
| Pasture | HerbaceousVegetation | 5 | 1.7% |
| Highway | Residential | 5 | 1.3% |

## Forest

- Recall: 99.8% of Forest tiles found
- Forest tiles predicted as something else: HerbaceousVegetation 1
- Other tiles predicted as Forest: HerbaceousVegetation 4, Pasture 2, River 1
