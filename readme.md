# Movie Recommendation Engine

A recommendation system (RecSys) that models user preferences from rating history and generates personalized movie suggestions. It is built on the MovieLens 1M dataset with Spark MLlib (ALS), scikit-learn, and a Streamlit demo.

**Live demo:** https://movierecommengine-hue2es7fjhfwhjb2mikjfa.streamlit.app/

## What it does

- Takes a MovieLens user ID and returns a ranked list of top-N movie recommendations
- Uses Spark ALS (matrix factorization) blended with a popularity score for users with enough rating history
- Falls back to popularity-based recommendations for cold-start users (50 or fewer ratings) and for unknown user IDs
- Shows which strategy produced each result set, with adjustable list size and popularity-blend weight

## Dataset

[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/): 1,000,209 ratings from 6,040 users on 3,706 movies (95.5% sparse). Evaluation uses a time-based 80/20 split per user, so no future behavior leaks into training.

## Results

Rating prediction (lower is better):

| Model | RMSE | MAE |
|---|---|---|
| Global mean baseline | 1.148 | 0.945 |
| Movie average baseline | 0.989 | 0.787 |
| Item-item collaborative filtering | 0.994 | 0.769 |
| SVD (k=50) | 1.029 | 0.808 |
| **Spark ALS (rank=10, regParam=0.05)** | **0.875** | **0.690** |

Ranking quality at K=10 (300-user evaluation sample):

| Model | Precision | Recall | NDCG | Hit Rate |
|---|---|---|---|---|
| Popularity | 0.090 | 0.048 | 0.233 | 0.367 |
| Spark ALS (raw scores) | 0.021 | 0.010 | 0.062 | 0.153 |

ALS wins clearly on rating prediction, but plain popularity beat raw ALS scores on top-N ranking. Rating accuracy and ranking quality are different objectives. This finding is why the final recommender blends ALS scores with a popularity boost and uses a popularity fallback for cold-start users, whose ranking quality was far weaker than that of regular users.

## Project structure

```
Movie_Recomm_Engine/
  data/
    raw/             MovieLens files (gitignored, downloaded by script)
    processed/       train/test splits and encoders (gitignored)
  notebooks/         02_eda ... 08_recommendation_generation
  src/
    download_data.py Downloads MovieLens 1M
    load_data.py     Loads data into Spark and runs sanity checks
    app.py           Streamlit demo
  models/            Saved model artifacts (gitignored, except app artifacts)
  reports/           Baseline/evaluation results and the final project report
  requirements.txt
```

## Run it locally

1. Create and activate a virtual environment:
```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
```
2. Install dependencies:
```
   pip install -r requirements.txt
```
3. PySpark needs Java 8, 11, 17, or 21 on your PATH (check with `java -version`).
4. Download the data and run the notebooks in order (02 to 08) to regenerate the processed data and models:
```
   python src\download_data.py
```
5. Launch the demo:
```
   streamlit run src\app.py
```

The two files the demo needs (`models/recommendation_artifacts.pkl` and `data/processed/movies_final.parquet`) are committed to the repo so the app can be deployed without rerunning the pipeline.

## Big data concepts used

- Spark DataFrames for data loading, with lazy evaluation and in-memory caching
- Spark MLlib ALS, a distributed matrix factorization algorithm (run here in local mode on one machine)
- Sparse matrix storage (about 3 MB instead of about 171 MB dense)
- Shuffle partition tuning

## Limitations

- Spark ran in local mode on a single machine, not on a multi-node cluster
- Ranking evaluation used a 300-user sample rather than the full test set
- Cold-start users are handled by popularity fallback, not a dedicated hybrid model
- On Windows, Spark's native model `.save()` fails without winutils, so ALS factors are saved as parquet instead

## Future improvements

- Learned hybrid of ALS and content-based (genre) scores
- Implicit-feedback ALS or a ranking-specific loss to improve top-N quality
- Genre-picker mode for users with no rating history
- Full test-set evaluation with distributed evaluation

## Tech stack

Python, Apache Spark (PySpark), scikit-learn, pandas, scipy, Streamlit. Everything is free and open-source.