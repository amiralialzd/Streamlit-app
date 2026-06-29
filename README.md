# 🚀 SpaceX Falcon 9 Landing Prediction Dashboard

An interactive Streamlit dashboard that explores SpaceX Falcon 9 first-stage landing outcomes and compares two tuned classification models. Users can filter by launch site and payload mass to see success rates, and switch between a Decision Tree and a K-Nearest Neighbors classifier.

## Live Demo


 http://192.168.8.141:8501


## Features

- **Interactive filters** — select launch site (CCAFS SLC 40, KSC LC 39A, VAFB SLC 4E, or All) and a payload mass range
- **Live visualizations** — success-rate pie chart and payload-vs-outcome scatter plot that update with your selection (built with Plotly)
- **Model comparison** — choose between a Decision Tree and a KNN classifier, each tuned with `GridSearchCV` (10-fold)
- **Confusion matrix** — visualized as an annotated heatmap for the selected model

## Dataset

90 SpaceX Falcon 9 launches, from the IBM Data Science Professional Certificate capstone dataset (`spacex_dataset_part_2.csv`). Each record includes the payload mass, orbit, launch site, booster details, and the landing outcome (`Class`: 1 = successful landing, 0 = failed).

Note: **84% of launches in this dataset were successful**, so a model that always predicts "success" scores 0.84 — the trained models should be read against that baseline.

## How the Model Works

The classifier is trained on the **full 90-launch dataset** (features standardized, 80/20 train/test split). The charts respond to the sidebar filters, but the model evaluation is computed on the full data — with only 90 launches, retraining on a filtered subset would leave too few rows for reliable results. This is noted directly in the app.

## Tech Stack

Python, Streamlit, Pandas, Plotly, Scikit-Learn

## Run Locally

```bash
pip install -r requirements.txt
streamlit run main.py
```

The app opens in your browser at `http://localhost:8501`.

## Possible Improvements

- The dataset is small (90 rows); more launch data would allow filtered-subset modeling.
- Add model accuracy/precision/recall metrics alongside the confusion matrix.
- Cache the trained models so GridSearch doesn't refit on every interaction.
