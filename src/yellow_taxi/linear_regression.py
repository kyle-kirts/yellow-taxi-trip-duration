import math

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import functions as F


FEATURES = (
    "zone_distance_mi", "pickup_hour_sin", "pickup_hour_cos", "is_weekend",
    "is_rush_hour", "same_zone", "is_airport_trip",
)
LABEL = "trip_duration_min"
PREDICTION = "prediction"
VECTOR = "_linear_features"
VALUE_COLUMNS = tuple(f"_linear_value_{name}" for name in FEATURES)
MISSING_COLUMNS = tuple(f"_linear_missing_{name}" for name in FEATURES)
VECTOR_FEATURES = (*FEATURES, *(f"{name}_missing" for name in FEATURES))


def _vectorize(df):
    expressions = []
    for name, value_col, missing_col in zip(FEATURES, VALUE_COLUMNS, MISSING_COLUMNS):
        value = F.col(name).try_cast("double")
        valid = value.isNotNull() & ~F.isnan(value) & (F.abs(value) < F.lit(math.inf))
        expressions.extend([
            F.when(valid, value).otherwise(0.0).alias(value_col),
            F.when(valid, 0.0).otherwise(1.0).alias(missing_col),
        ])
    filled = df.select("*", *expressions)
    assembler = VectorAssembler(
        inputCols=[*VALUE_COLUMNS, *MISSING_COLUMNS], outputCol=VECTOR,
    )
    return assembler.transform(filled)


def fit_linear_model(train, reg_param=0.0):
    if not math.isfinite(reg_param) or reg_param < 0:
        raise ValueError("reg_param must be finite and nonnegative")
    estimator = LinearRegression(
        featuresCol=VECTOR, labelCol=LABEL, predictionCol=PREDICTION,
        regParam=reg_param, elasticNetParam=0.0,
        fitIntercept=True, standardization=True,
        solver="l-bfgs", maxIter=100, tol=1e-6,
    )
    return estimator.fit(_vectorize(train))


def predict(model, trips):
    return model.transform(_vectorize(trips)).select(*trips.columns, PREDICTION)
