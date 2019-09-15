# local imports
from cleaning import DataSet

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
train = DataSet("train_values.csv", "train_labels.csv")

train.categorical_to_binary(categorical_columns)
corr = train.normalize(to_normalize).corr().abs().unstack().sort_values(ascending=False)

print(corr[(corr > 0.70) & (corr < 1.)])