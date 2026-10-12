---
title: "Real-Time Traffic Congestion Detection and Prediction Using Apache Kafka and Apache Spark"
---

# Cover Page

**Project title:** Real-Time Traffic Congestion Detection and Prediction Using Apache Kafka and Apache Spark

**Course / Laboratory:** [fill in]

**Institution:** [fill in]

**Team members (name, roll number):**

1. [fill in]
2. [fill in]
3. [fill in]
4. [fill in]

**Faculty / Instructor:** [fill in]

**Date of submission:** 2026-10-07

**Laboratory concepts used:** Apache Kafka, Apache Spark (PySpark, Structured Streaming)

\newpage

# 1. Abstract

Road congestion must be detected while it is developing, not after it has dissolved. This project builds a streaming pipeline that analyses freeway sensor data to identify congested road segments, peak traffic periods and traffic statistics, raises congestion alerts, and predicts congestion 30 minutes ahead. The real-world dataset is UCI **PEMS-SF** (Caltrans PeMS): 963 detectors in the San Francisco Bay Area, lane occupancy sampled every 10 minutes. Because the data is historical, real recorded observations are **replayed** through **Apache Kafka** at a controlled speed; **Spark Structured Streaming** consumes the topic, computes an occupancy-based congestion index, windowed aggregates and de-duplicated alerts. Peak periods are derived from the data, a paired t-test confirms that peak occupancy differs from off-peak occupancy, and a Random Forest predicts the future congestion level using a chronological train/test split. A pandas baseline is compared with Spark on up to 37,025,424 records. Inspection of the supplied files revealed data-quality issues (only 267 distinct days out of 440 supplied, no timestamps, occupancy as the only variable) which are documented and handled rather than hidden.

# 2. Problem Statement

Develop a system to analyse vehicle-movement information and identify congested road segments and peak traffic periods, producing congested locations, peak traffic periods, traffic statistics and congestion alerts, using parallel/distributed technologies from the laboratory and a real public dataset.

# 3. Objectives

1. Use a real, publicly available traffic dataset and document its source.
2. Stream the observations through Kafka and process them with Spark Structured Streaming.
3. Detect congested sensors with a documented, configurable congestion index.
4. Identify peak traffic periods from the data, not from assumption.
5. Generate de-duplicated congestion alerts.
6. Predict upcoming congestion with a machine-learning model, evaluated without data leakage.
7. Apply a statistical test (t-test) to the peak/off-peak difference.
8. Compare Spark with a serial baseline and analyse speedup and scalability honestly.
9. Provide a live dashboard and a reproducible Windows demonstration.

# 4. Dataset Description and Source

**Dataset:** PEMS-SF, UCI Machine Learning Repository, donor Marco Cuturi.
**URL:** https://archive.ics.uci.edu/dataset/204/pems-sf (licence CC BY 4.0).
**Original data owner:** California Department of Transportation (Caltrans), Performance Measurement System (PeMS), http://pems.dot.ca.gov.
**Citation:** M. Cuturi, "Fast global alignment kernels", Proceedings of ICML 2011.

**Content:** each of the supplied days is a 963 x 144 matrix: 963 detectors, 144 ten-minute slots, value = lane occupancy rate in [0, 1]. The files are `PEMS_train`, `PEMS_test` (days), `PEMS_trainlabels`, `PEMS_testlabels` (weekday labels), `randperm` (the shuffle applied by the donor) and `stations_list` (963 sensor IDs).

**Column mapping used by the project** (the raw files have no column names):

| Project field | Derived from | Status |
|---|---|---|
| `sensor_id` | `stations_list` (row i = sensor i) | order assumed from UCI documentation |
| `day_index` | inverse of `randperm` (chronological order) | verified (weekday-chain score 0.30 in stored order, 1.00 after restoring) |
| `dow` (weekday) | label files, encoding inferred from data | verified from data: label 1 = Sunday |
| `slot` / time of day | column position x 10 min | slot 0 = 00:00 inferred from the daily profile |
| `occupancy` | matrix values | the only measurement |

**Findings from inspecting the supplied files** (all measured, none assumed):

1. **Only occupancy exists.** There is no speed, flow or vehicle count. The congestion index is therefore occupancy-based, and the project shows no speed or flow figures.
2. **No timestamps.** Chronological order was recovered from `randperm` and verified.
3. **Only 267 of the 440 supplied days are distinct.** 173 days are exact copies (days *i* and *i*+267 are identical in chronological order). They were removed; keeping them would put the same day in both the training and the test set.
4. **Weekday encoding:** the data shows labels 1 and 7 have the lowest occupancy, and the removed-day gaps coincide with Monday holidays, so label 1 = Sunday and 2 = Monday (the UCI text states Monday first).
5. **Dates are approximate.** The first day is anchored at 2008-01-02 and the last distinct day falls near 2008-09-30; only the first six removed-day gaps match known holidays. `day_index`, weekday and time of day are the reliable time axes.
6. **No coordinates** are provided, so no map is drawn.
7. Data quality after cleaning: no NaN values and no out-of-range values; 36,694 exact zeros (mostly night hours) were kept as genuine observations.

After de-duplication the working dataset has **267 days x 963 sensors x 144 slots = 37,025,424 sensor observations** (187 training days, 80 test days, split chronologically).

# 5. System Architecture

![System architecture](output/results/figures/architecture.png)

**Data flow.** Preprocessing converts the raw files into a chronological cube and daily Parquet files. The Kafka producer replays real observations (one tick = one 10-minute slot = 963 messages) to topic `traffic_data` (3 partitions, key = sensor id, so each sensor's readings stay ordered). Spark Structured Streaming reads the topic, parses JSON, drops malformed records, computes the congestion index and level, and runs two queries: (1) a micro-batch handler that stores records, per-tick network summaries, de-duplicated alerts and throughput metrics; (2) an event-time windowed aggregation (30 minutes per sensor, 10-minute watermark, parquet sink, checkpoint). The dashboard reads these outputs, the historical results and the saved prediction model. A separate benchmark pipeline compares a pandas serial implementation with Spark on the same Parquet files.

**Why Kafka.** Traffic sensors produce a continuous stream of small ordered events. Kafka decouples the producers from the processors, buffers bursts, stores events durably so they can be replayed, supports several independent consumers, and scales by adding partitions. Reading a CSV file would be batch processing and would not exercise any of this.

**Why Spark.** Structured Streaming provides event-time windows, watermarks, checkpointed fault-tolerant state and the DataFrame API, and the same code scales from a laptop to a cluster. The aggregation per sensor and window is naturally data-parallel.

**Why not MPI/OpenMP/CUDA.** The problem is stream ingestion, windowed aggregation and fault tolerance, which Kafka and Spark provide. MPI/OpenMP target tightly coupled numerical kernels and CUDA needs GPUs; none provides topics, offsets, windows or recovery.

# 6. Methodology

## 6.1 Preprocessing

Raw files are never modified. Steps: parse the matrices; restore chronology with the inverse of `randperm` and verify it with the weekday chain; remove exact duplicate days; infer the weekday encoding from the data; validate values (invalid values would be repaired by interpolation, none were found); derive hour, minute, weekday and weekend features; write the cube, metadata and one Parquet file per day.

## 6.2 Congestion index and levels

With occupancy as the only available variable the index is

$$CI = \mathrm{clip}\left(\frac{\text{occupancy}}{\text{reference}}, 0, 1\right)$$

where the reference occupancy is the 99th percentile of **training-day** occupancy (0.2552). Levels use the 90th, 95th and 99th percentiles of training occupancy, converted to CI units:

| Level | Condition (CI) | Occupancy equivalent |
|---|---|---|
| LOW | CI < 0.375 | below 0.0956 |
| MODERATE | 0.375 <= CI < 0.469 | 0.0956 to 0.1198 |
| HIGH | 0.469 <= CI < 1 | 0.1198 to 0.2552 |
| SEVERE | CI = 1 | at or above 0.2552 |

The weights of the suggested speed/occupancy/flow formula do not apply because speed and flow do not exist in this dataset. **Rationale and limitation:** the thresholds are relative (percentile-based), computed on training days only; they are not absolute engineering limits, and by construction roughly 10% of observations lie above LOW. All values are configurable (`config/config.yaml`, including a manual override).

## 6.3 Peak traffic detection

For each (weekday, hour) cell the mean network CI is computed over all days. A cell is a **peak** if it lies above the 80th percentile of the 168 cells (34 peak cells). Consecutive peak hours are merged into peak periods.

## 6.4 Alerts

An alert is raised when a sensor reaches HIGH or SEVERE. To avoid duplicates: one alert per congestion episode per sensor, one ESCALATION alert if the level rises, an episode ends after 3 consecutive ticks below HIGH, and a cooldown of 6 ticks suppresses flapping.

## 6.5 Prediction

*Target:* the congestion level of the same sensor 30 minutes (3 slots) later. *Features:* current CI, three lags, rolling means (30 and 60 minutes), two trend differences, slot, weekday, weekend flag and the sensor's training-period mean CI. *Model:* Random Forest (60 trees, depth 18, minimum leaf 20). *Split:* chronological by day, the first 187 days train and the last 80 test; 300,000 training and 150,000 test samples drawn at random inside their respective periods. *Leakage control:* features only use slots up to t; calibration and the sensor baseline use training days only; duplicates removed; hyper-parameters chosen on a validation split inside the training days. *Class imbalance:* measured (train: LOW 268,281, MODERATE 15,717, HIGH 12,763, SEVERE 3,239), reported with macro and per-class metrics; class weighting was evaluated on validation data and rejected because it lowered accuracy and macro F1 (Table in Section 9.5). *Baselines:* persistence (future level = current level) and always-LOW.

## 6.6 Statistical analysis

H0: mean occupancy in peak periods equals mean occupancy in off-peak periods; H1: they differ; alpha = 0.05. The unit of analysis is one **day** (neighbouring 10-minute values are strongly autocorrelated, so treating them as independent would inflate the t-statistic): each weekday contributes one peak mean and one off-peak mean of the network-average occupancy, and because both come from the same day the primary test is the **paired t-test** (Welch two-sample reported too). Peaks are selected for being high, so testing them on the same data would be circular; the primary analysis defines peaks on **training days** and tests them on the **held-out test days**.

## 6.7 Performance methodology

The workload (congestion index and level for every record, weekday x hour profile, per-sensor statistics) is implemented twice on the same Parquet files: a serial pandas version that processes day files one after another and merges partial sums, and a Spark version. Sizes are whole days of real data. Each configuration is run 3 times and the median is reported; Spark session start-up and the first (warm-up) job are measured separately; results of both engines are compared for equality.

# 7. Technologies Used

Python; Apache Kafka (KRaft, kafka-python client); Apache Spark / PySpark 3.5 with Structured Streaming; Pandas and NumPy; scikit-learn (Random Forest); SciPy (t-tests); Matplotlib and Plotly; Streamlit; PyArrow/Parquet; PyYAML; pytest.

# 8. Implementation Details

| Module | Role |
|---|---|
| `preprocessing/preprocess.py` | parsing, chronology recovery, de-duplication, validation, Parquet output |
| `common/congestion.py`, `common/alerts.py` | calibration, congestion index/levels, alert manager |
| `ingestion/producer.py`, `ingestion/kafka_utils.py` | real-data replay to Kafka, topic management, error handling, throughput statistics |
| `spark_jobs/spark_streaming.py` | Kafka source, JSON parsing, `foreachBatch` handler, windowed aggregation, checkpoints |
| `spark_jobs/congestion_detection.py`, `spark_jobs/aggregation.py` | Spark expressions for CI/levels, window and batch aggregations |
| `analysis/historical.py`, `stats/t_test.py` | peaks, ranking, descriptive statistics, t-tests |
| `prediction/*.py` | feature engineering, validation-based tuning, training, prediction |
| `performance/*.py` | serial baseline, Spark benchmark, scalability experiment |
| `dashboard/app.py` | Streamlit dashboard (live sections read Spark outputs) |
| `tests/` | 20 automated tests |

Configuration lives in `config/config.yaml` (replay delay, thresholds, horizon, window, watermark, alert rules, benchmark sizes). Logging is used throughout; the producer and Spark job handle connection failures and malformed records.

**Execution status (honest).** Preprocessing, analysis, t-test, model training, the benchmark, the tests and the Spark Structured Streaming job (through its offline file-source test mode) were executed in the development environment. **The live Kafka broker, the producer writing to Kafka and Spark's Kafka source were not executed there** (broker and connector downloads were not available); they must be run on the demonstration computer with the commands in the README.

**Individual contribution (TEMPLATE - edit to reflect who actually did what):**

| Member | Suggested area |
|---|---|
| Member 1 | Dataset study and preprocessing |
| Member 2 | Kafka producer and streaming ingestion |
| Member 3 | Spark streaming, congestion detection, peak analysis |
| Member 4 | ML prediction, statistics, dashboard, performance analysis |

*Replace names and adjust the table so that it states only work each member really performed.*

# 9. Results and Screenshots

## 9.1 Congestion levels and congested locations

Level shares over all 267 days: LOW 90.18%, MODERATE 4.93%, HIGH 3.92%, SEVERE 0.97%.

![Congestion level distribution](output/results/figures/level_distribution.png)

Ranking of the most congested sensors (share of observations at HIGH or SEVERE over all days); 337 sensors are at HIGH or above for more than 5% of all observations. The top three sensors are 401434, 400696, 401356; the worst one is at HIGH or above 44.5% of the time.

| Rank | Sensor | Mean occ. | % HIGH+ | % SEVERE | Peak-period CI |
|---|---|---|---|---|---|
| 1.0000 | 401434.0000 | 0.1234 | 44.4783 | 10.6586 | 0.7152 |
| 2.0000 | 400696.0000 | 0.1054 | 41.2635 | 2.1666 | 0.6245 |
| 3.0000 | 401356.0000 | 0.1406 | 40.9436 | 14.3779 | 0.7233 |
| 4.0000 | 400436.0000 | 0.1008 | 30.1810 | 6.7078 | 0.6519 |
| 5.0000 | 401466.0000 | 0.0983 | 25.0858 | 8.6246 | 0.6339 |
| 6.0000 | 401009.0000 | 0.0887 | 24.5214 | 2.5047 | 0.5646 |
| 7.0000 | 401660.0000 | 0.0772 | 24.3784 | 1.0040 | 0.5139 |
| 8.0000 | 400682.0000 | 0.0819 | 22.2118 | 3.6439 | 0.5461 |
| 9.0000 | 401110.0000 | 0.0815 | 21.8633 | 1.7894 | 0.5106 |
| 10.0000 | 400546.0000 | 0.0744 | 21.8243 | 0.6502 | 0.4648 |

![Most congested sensors](output/results/figures/top_congested_sensors.png)

## 9.2 Peak traffic periods

| Day | From | To | Mean occupancy | Mean CI | % HIGH+SEVERE |
|---|---|---|---|---|---|
| Monday | 07:00 | 10:00 | 0.0899 | 0.3437 | 15.4220 |
| Monday | 15:00 | 18:00 | 0.0902 | 0.3465 | 13.9505 |
| Tuesday | 07:00 | 10:00 | 0.0951 | 0.3618 | 18.3564 |
| Tuesday | 15:00 | 19:00 | 0.0935 | 0.3571 | 16.9065 |
| Wednesday | 07:00 | 10:00 | 0.0933 | 0.3560 | 17.5977 |
| Wednesday | 15:00 | 19:00 | 0.0959 | 0.3661 | 18.3290 |
| Thursday | 07:00 | 10:00 | 0.0934 | 0.3565 | 17.5637 |
| Thursday | 15:00 | 19:00 | 0.1011 | 0.3828 | 21.1775 |
| Friday | 07:00 | 09:00 | 0.0859 | 0.3317 | 12.7341 |
| Friday | 14:00 | 19:00 | 0.0986 | 0.3731 | 18.5996 |

Weekday peaks are the morning and afternoon rush hours (the afternoon peak starts earlier and is longer on Fridays); no weekend cell reaches the peak criterion.

![Mean occupancy by weekday and hour](output/results/figures/heatmap_weekday_hour.png)

![Weekday versus weekend profile](output/results/figures/daily_profile_weekday_weekend.png)

## 9.3 Statistical analysis (t-test)

| Quantity | Value |
|---|---|
| H0 / H1 | peak mean = off-peak mean / means differ |
| Sample | 56 held-out weekdays (peaks defined on training days) |
| Peak mean occupancy | 0.0930 |
| Off-peak mean occupancy | 0.0424 |
| Mean difference (95% CI) | 0.0506 (0.0492 to 0.0519) |
| Paired t-statistic (df = 55) | 73.69 |
| p-value | 1.16e-56 |
| Welch t-statistic / p-value | 69.17 / 4.51e-63 |
| Effect size (Cohen's d, paired) | 9.8 |
| alpha | 0.05 |
| Decision | **REJECT H0** |

**Interpretation.** The p-value is far below 0.05, so the observed difference is not explained by chance: occupancy during data-derived peak periods is higher than off-peak by about 0.0506 (a large effect). It does not by itself say anything about causes. Comparison: using all days and peaks defined on the same data gives t = 140.71, p = 2.93e-193 (n = 190); this in-sample version is optimistic because peaks were chosen for being high. Weekday versus weekend daily mean occupancy: 0.0572 vs 0.0416 (Welch t = 35.02, p = 5.85e-62).

![t-test figure](output/results/figures/t_test_peak_vs_offpeak.png)

## 9.4 Streaming results and alerts

The Spark Structured Streaming job was verified on one replayed weekday. **Source: Spark Structured Streaming via OFFLINE file source (no Kafka broker), day 189** (same transformation code as the Kafka run; only the input source differs). Results: 138,672 records in 144 ticks processed, maximum occupancy difference to the source data 2.9e-08, Spark and NumPy congestion levels agree for 100.00% of records, 24 micro-batches, 45,261 window-aggregate rows, level counts LOW 124,576, MODERATE 8,355, HIGH 5,066, SEVERE 675. **Alerts:** 5,741 raw HIGH-or-SEVERE sensor observations were reduced to 977 alerts (860 new, 117 escalations, 518 sensors), a 5.9-fold reduction through episode de-duplication and cooldown. Because the thresholds are relative (95th percentile), many sensors alert during one weekday's rush hours.

Example alert format produced by the system:

```
CONGESTION ALERT [NEW] Sensor: <id> | Day#<n> <HH:MM> | Occupancy: <value> | Congestion Index: <value> | Level: HIGH
```

(Speed and traffic flow are not available in this dataset, so the alert reports occupancy and the congestion index.) Real examples are in `output/results/alerts_filesource_test.csv`.

Kafka throughput: NOT YET EXECUTED - requires a Kafka broker. Run `run_project.bat bench` (`python -m ingestion.producer --max-speed --num-days 5`) and fill in `output\results\producer_stats_maxspeed.json`, then re-run `python docs\build_report.py`.

## 9.5 Congestion prediction (30 minutes ahead)

Evaluation on 150,000 samples from the 80 held-out days (test classes: LOW 135,157, MODERATE 7,526, HIGH 5,854, SEVERE 1,463).

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | HIGH+ precision | HIGH+ recall | HIGH+ F1 |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.9377 | 0.6921 | 0.6312 | 0.6579 | 0.7307 | 0.6918 | 0.7107 |
| Persistence baseline | 0.9290 | 0.6514 | 0.6475 | 0.6494 | 0.6954 | 0.7003 | 0.6979 |
| Always LOW baseline | 0.9010 | 0.2253 | 0.2500 | 0.2370 | 0.0000 | 0.0000 | 0.0000 |

The Random Forest is above the persistence baseline on accuracy, macro F1, HIGH+ F1 (accuracy 0.938 vs 0.929; macro F1 0.658 vs 0.649; HIGH+ F1 0.711 vs 0.698). The margin is modest because the congestion state is strongly persistent over 30 minutes; the always-LOW baseline shows that accuracy alone is misleading for this imbalanced problem.

Per-class metrics of the Random Forest:

| Level | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| LOW | 0.9689 | 0.9842 | 0.9765 | 135157 |
| MODERATE | 0.5802 | 0.4457 | 0.5041 | 7526 |
| HIGH | 0.6209 | 0.6088 | 0.6148 | 5854 |
| SEVERE | 0.5985 | 0.4860 | 0.5364 | 1463 |

Confusion matrix (counts):

|  | pred LOW | pred MODERATE | pred HIGH | pred SEVERE |
|---|---|---|---|---|
| actual LOW | 133019 | 1284 | 720 | 134 |
| actual MODERATE | 3160 | 3354 | 967 | 45 |
| actual HIGH | 935 | 1057 | 3564 | 298 |
| actual SEVERE | 177 | 86 | 489 | 711 |

Most important features: ci_t (0.25), roll_mean_3 (0.22), ci_lag1 (0.20), ci_lag2 (0.08).

![Confusion matrix](output/results/figures/confusion_matrix.png)

![Model versus baselines](output/results/figures/model_vs_baselines.png)

Model selection (validation days inside the training period, never the test days):

| Candidate (validation days) | Accuracy | Macro F1 | HIGH+ F1 |
|---|---|---|---|
| RF balanced depth14 leaf50 | 0.8799 | 0.6140 | 0.6898 |
| RF unweighted depth14 leaf50 | 0.9350 | 0.6498 | 0.6963 |
| RF unweighted depth18 leaf20 | 0.9354 | 0.6539 | 0.7008 |
| RF balanced depth10 leaf200 | 0.8681 | 0.6030 | 0.6856 |
| Persistence baseline | 0.9263 | 0.6408 | 0.6832 |

## 9.6 Performance analysis

Measured on: 1 CPU core(s), 3.9 GB RAM, Python 3.12.3, PySpark 3.5.9, pandas 3.0.2, Spark master local[*]. Median of 3 runs; Spark session start-up (19.94 s) and the first warm-up job (12.38 s) are excluded and reported separately. Results of both engines were checked for equality on every size.

| Days | Records | Serial pandas (s) | Spark (s) | Speedup | pandas rec/s | Spark rec/s |
|---|---|---|---|---|---|---|
| 1.00 | 138,672.00 | 0.19 | 1.10 | 0.17 | 747,048.03 | 125,948.30 |
| 4.00 | 554,688.00 | 0.21 | 1.04 | 0.20 | 2,628,392.76 | 532,395.03 |
| 7.00 | 970,704.00 | 0.34 | 1.04 | 0.33 | 2,856,555.33 | 930,278.36 |
| 15.00 | 2,080,080.00 | 0.59 | 1.15 | 0.52 | 3,505,728.83 | 1,814,892.45 |
| 36.00 | 4,992,192.00 | 1.10 | 2.18 | 0.50 | 4,549,880.85 | 2,285,924.17 |
| 72.00 | 9,984,384.00 | 2.29 | 2.67 | 0.85 | 4,368,877.55 | 3,732,751.65 |
| 267.00 | 37,025,424.00 | 9.29 | 7.79 | 1.19 | 3,985,283.03 | 4,754,999.20 |

**Speedup = serial time / Spark time.** Spark was slower than the serial baseline for datasets up to 9,984,384 records and faster from 37,025,424 records (1.19x). The crossover therefore lies between 9,984,384 and 37,025,424 records on this machine. The best measured speedup was 1.19x.

**Explanation.** Spark has fixed costs (JVM start-up, query planning, task scheduling and a shuffle) that dominate on small inputs, while the serial baseline is a vectorised pandas pass over files that fit comfortably in memory. As the number of records grows the fixed cost is amortised and Spark's parallel read and aggregation pay off. This benchmark was run on a machine with a single core, so it shows the amortisation of overhead but not multi-core scaling; with several cores or machines Spark's advantage is expected to appear earlier and grow. Scalability over cores: NOT YET EXECUTED - the development machine had a single core. Run `python -m performance.benchmark --scaling` on a multi-core computer. Kafka throughput and the Spark micro-batch rate in the Kafka run must be measured on the demonstration computer.

![Execution time](output/results/figures/benchmark_time.png)

![Speedup](output/results/figures/benchmark_speedup.png)

![Throughput](output/results/figures/benchmark_throughput.png)

## 9.7 Dashboard screenshots (to be inserted by the team)

*Screenshots must be taken from the live demonstration on the team's computer.*

1. [Insert screenshot: dashboard header with KPI cards while the stream is running]
2. [Insert screenshot: Live Traffic Status tab]
3. [Insert screenshot: Congested Locations tab]
4. [Insert screenshot: Peak Traffic Periods heat-map]
5. [Insert screenshot: Congestion Alerts tab with recent alerts]
6. [Insert screenshot: Prediction tab (predicted levels and live check)]
7. [Insert screenshot: Statistical Analysis tab]
8. [Insert screenshot: Performance Analysis tab]
9. [Insert screenshot: terminals showing Kafka, producer and Spark running]

# 10. Conclusion

A complete Kafka -> Spark Structured Streaming pipeline was built on a real public dataset. It detects congested sensors with a documented index, derives peak periods from data, produces de-duplicated alerts, predicts congestion 30 minutes ahead slightly better than a strong persistence baseline, and confirms statistically that peak-period occupancy differs from off-peak occupancy. The benchmark honestly shows that Spark is slower than a vectorised serial baseline on small data and overtakes it only on the largest dataset on a single core. Limitations: occupancy is the only measured variable, congestion levels are relative, the live stream is a replay of historical data, dates are approximate, Kafka and Spark ran in single-node mode, and the live Kafka path still has to be demonstrated on the team's computer.

# 11. Future Enhancements

Use full PeMS data with speed and flow and absolute congestion thresholds; add sensor coordinates and a map; multi-node Kafka and Spark; gradient boosting or sequence models; spatial features from neighbouring sensors; persistent alert state; scoring the model inside the stream (Spark MLlib); incident and weather data.

# 12. References

1. M. Cuturi, "Fast global alignment kernels", Proceedings of the 28th International Conference on Machine Learning (ICML), 2011. Dataset: PEMS-SF, UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/204/pems-sf.
2. California Department of Transportation, Performance Measurement System (PeMS), http://pems.dot.ca.gov.
3. Apache Kafka documentation, https://kafka.apache.org/documentation/.
4. Apache Spark documentation, Structured Streaming Programming Guide and Structured Streaming + Kafka Integration Guide, https://spark.apache.org/docs/latest/.
5. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python", Journal of Machine Learning Research 12, 2011.
6. P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python", Nature Methods 17, 2020.
7. L. Breiman, "Random Forests", Machine Learning 45(1), 2001.
8. Streamlit documentation, https://docs.streamlit.io/.
