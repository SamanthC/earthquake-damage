import datetime
from pathlib import Path

# third-party imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# local imports
from cleaning import DataSet
from models import Model

pd.set_option("display.max_columns", 20)
pd.set_option('display.width', 120)


categorical_columns = {
    "geo_level_1_id": "level_1",
    "land_surface_condition": "land_surface",
    "foundation_type": "foundation",
    "roof_type": "roof",
    "ground_floor_type": "ground_floor",
    "other_floor_type": "other_floor",
    "position": "position",
    "plan_configuration": "plan",
    "legal_ownership_status": "legal_owner"
    }


to_normalize = [
    "count_floors_pre_eq",
    "age", "area_percentage",
    "height_percentage", "count_families"
]

to_drop = [
    "geo_level_2_id", "geo_level_3_id",
    "land_surface_t", "roof_q",
    "position_t", "count_floors_pre_eq"
]


# Cleaning train and test sets

train = DataSet("train_values.csv", "train_labels.csv")
train.categorical_to_binary(categorical_columns)
train.normalize(to_normalize)
train.drop(to_drop)

test = DataSet("test_values.csv", "submission_format.csv")
test.categorical_to_binary(categorical_columns)
test.normalize(to_normalize)
test.drop(to_drop)

# Defining models

RANDOM_FOREST = Model(
    name="Random Forest",
    estimator=RandomForestClassifier(),
    hyperparameters={
        "n_estimators": [50],
        "min_samples_leaf": [2],
    }
)

# Train Model

RANDOM_FOREST.choose_best_param(train)

# Make prediction

prediction = RANDOM_FOREST.estimator.predict(test.values_df)

Submission=pd.DataFrame(
    data=prediction,
    index=test.labels_series["building_id"],
    columns=["damage_grade"])

# Submission file

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{timestamp}_test_labels_series.csv"
filepath = Path("data", "submissions", filename)
filepath.parent.mkdir(exist_ok=True)
Submission.to_csv(filepath)