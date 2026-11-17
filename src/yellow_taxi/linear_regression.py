import math

from pyspark.ml.feature import OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import functions as F

from yellow_taxi.weather_features import WEATHER_FEATURES


FEATURES = (
    "zone_distance_mi", "pickup_hour_sin", "pickup_hour_cos", "is_weekend",
    "is_rush_hour", "same_zone", "is_airport_trip",
)
LABEL = "trip_duration_min"
PREDICTION = "prediction"
VECTOR = "_linear_features"
VECTOR_FEATURES = (*FEATURES, *(f"{name}_missing" for name in FEATURES))

# 0 is unknown; the last valid category is the reference.
CATEGORY_LIMITS = {"PULocationID": 263, "DOLocationID": 263, "pickup_dow": 7}
MONTH_TERMS = ("pickup_month_sin", "pickup_month_cos", "pickup_month_missing")
WEATHER_INPUTS = (*WEATHER_FEATURES, "weather_missing")


def model_feature_inputs(weather=True):
    return (*FEATURES, *(WEATHER_INPUTS if weather else ()), *CATEGORY_LIMITS, "pickup_month")


def _missing_flags(numeric):
    # The prepared weather_missing flag already identifies missing temperature.
    return tuple(name for name in numeric if name not in ("temp_c", "weather_missing"))


def model_feature_names(weather=True):
    numeric = (*FEATURES, *(WEATHER_INPUTS if weather else ()))
    names = [*numeric, *(f"{name}_missing" for name in _missing_flags(numeric)), *MONTH_TERMS]
    for name, limit in CATEGORY_LIMITS.items():
        names += [f"{name}=unknown", *(f"{name}={i}" for i in range(1, limit))]
    return tuple(names)


def _finite(value):
    return value.isNotNull() & ~F.isnan(value) & (F.abs(value) < F.lit(math.inf))


def _assemble(df, numeric, categories, derived=None):
    derived = derived or {}
    expressions = dict(derived)
    value_cols = []
    flag_cols = []
    for name in numeric:
        value = F.col(name).try_cast("double")
        valid = _finite(value)
        col = f"_value_{name}"
        value_cols.append(col)
        if name == "weather_missing":
            valid = valid & _finite(F.col("temp_c").try_cast("double"))
            expressions[col] = F.when(valid, value).otherwise(1.0)
        else:
            expressions[col] = F.when(valid, value).otherwise(0.0)
        if name in _missing_flags(numeric):
            flag = f"_missing_{name}"
            flag_cols.append(flag)
            expressions[flag] = F.when(valid, 0.0).otherwise(1.0)

    index_cols = []
    encoded_cols = []
    for name, limit in categories.items():
        value = F.col(name).try_cast("double")
        valid = value.between(1, limit) & (value == F.floor(value))
        index = f"_index_{name}"
        index_cols.append(index)
        encoded_cols.append(f"_encoded_{name}")
        expressions[index] = F.when(valid, value).otherwise(0.0)

    indexed = df.withColumns(expressions)
    if categories:
        # Fit once on the fixed TLC/calendar ranges, independent of the trips.
        limits = df.sparkSession.range(1, numPartitions=1).select(*[
            F.lit(float(limit)).alias(index) for index, limit in zip(index_cols, categories.values())
        ])
        encoder = OneHotEncoder(inputCols=index_cols, outputCols=encoded_cols, dropLast=True)
        indexed = encoder.fit(limits).transform(indexed)
    assembler = VectorAssembler(inputCols=[*value_cols, *flag_cols, *derived, *encoded_cols], outputCol=VECTOR)
    return assembler.transform(indexed).select(*df.columns, VECTOR)


def model_vectors(df, weather=True):
    numeric = (*FEATURES, *(WEATHER_INPUTS if weather else ()))
    month = F.col("pickup_month").try_cast("double")
    valid = month.between(1, 12) & (month == F.floor(month))
    angle = 2 * math.pi * (month - 1) / 12
    month_terms = {
        "_pickup_month_sin": F.when(valid, F.sin(angle)).otherwise(0.0),
        "_pickup_month_cos": F.when(valid, F.cos(angle)).otherwise(0.0),
        "_pickup_month_missing": F.when(valid, 0.0).otherwise(1.0),
    }
    return _assemble(df, numeric, CATEGORY_LIMITS, month_terms)


def linear_estimator(reg_param):
    if not math.isfinite(reg_param) or reg_param < 0:
        raise ValueError("reg_param must be finite and nonnegative")
    return LinearRegression(
        featuresCol=VECTOR, labelCol=LABEL, predictionCol=PREDICTION,
        regParam=reg_param, elasticNetParam=0.0,
        fitIntercept=True, standardization=True,
        solver="l-bfgs", maxIter=100, tol=1e-6,
    )


def fit_linear_model(train, reg_param=0.0):
    return linear_estimator(reg_param).fit(_assemble(train, FEATURES, {}))


def predict(model, trips):
    return model.transform(_assemble(trips, FEATURES, {})).select(*trips.columns, PREDICTION)
