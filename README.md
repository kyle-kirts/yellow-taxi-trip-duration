# yellow-taxi-trip-duration

## Expected HDFS Layout

Preparing training data is hardcoded to expect the following Hadoop directory layout.

```text
/yellow-taxi
└── raw
    ├── 2024
    │   ├── yellow_tripdata_2024-01.parquet
    │   └── ...
    └── 2025
        ├── yellow_tripdata_2025-01.parquet
        └── ...
```


Prepared training data will be found at:

```text
/yellow-taxi
└── prepared
    └── train
        ├── _SUCCESS
        ├── fold=0
        │   └── part-00000-<uuid>.c000.snappy.parquet
        ├── fold=1
        │   └── ...
        └── ...
```

Where each fold=* directory has parquet files containing the samples for that fold of the preprocessed data.


# Running Spark Jobs

run.sh contains a bash script for simplifying our terminal inputs instead of having to write out the long winded spark-submit command, 
you can simply run './run.sh local script.py' to run an application on your local computer or with './run.sh cluster script.py' to run the application on your school cluster. If any scripts require further arguments, you can simply add them after the script.py argument.

EXAMPLE

Running a cluster prepare job on raw training data:  ./run.sh cluster prepare.py





