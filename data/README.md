# Dataset layout

The official assignment requires the public Kaggle Cats and Dogs binary-classification dataset. This project defaults to `tongpython/cat-and-dog`, configured in `params.yaml`.

`dvc repro` creates and versions these generated directories:

```text
data/
  raw/                         # Kaggle download (DVC output)
  processed/
    train/{cats,dogs}/         # 80%
    validation/{cats,dogs}/    # 10%
    test/{cats,dogs}/          # 10%
    manifest.csv               # source, split, label, SHA-256
```

Raw and processed images are intentionally excluded from Git and the submission ZIP. Their content hashes are captured by DVC once the pipeline runs. Do not upload Kaggle data to a public Git repository unless its licence permits redistribution.

