# Test results: rgb (3 bands)

Test tiles: 4050 | Overall accuracy: 96.2%

| Class | Precision | Recall | F1 | Test tiles |
|---|---|---|---|---|
| AnnualCrop | 0.935 | 0.956 | 0.945 | 450 |
| Forest | 0.987 | 0.993 | 0.990 | 450 |
| HerbaceousVegetation | 0.919 | 0.956 | 0.937 | 450 |
| Highway | 0.981 | 0.957 | 0.969 | 375 |
| Industrial | 0.976 | 0.976 | 0.976 | 375 |
| Pasture | 0.959 | 0.947 | 0.953 | 300 |
| PermanentCrop | 0.932 | 0.917 | 0.925 | 375 |
| Residential | 0.996 | 0.984 | 0.990 | 450 |
| River | 0.950 | 0.920 | 0.935 | 375 |
| SeaLake | 0.985 | 0.996 | 0.990 | 450 |

## Most frequent mistakes

| True class | Predicted as | Tiles | Share of true class |
|---|---|---|---|
| PermanentCrop | HerbaceousVegetation | 19 | 5.1% |
| HerbaceousVegetation | PermanentCrop | 11 | 2.4% |
| River | AnnualCrop | 10 | 2.7% |
| PermanentCrop | AnnualCrop | 10 | 2.7% |
| AnnualCrop | Pasture | 7 | 1.6% |

## Forest

- Recall: 99.3% of Forest tiles found
- Forest tiles predicted as something else: AnnualCrop 1, HerbaceousVegetation 2
- Other tiles predicted as Forest: HerbaceousVegetation 2, Pasture 2, River 2
