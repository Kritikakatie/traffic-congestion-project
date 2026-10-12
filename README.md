# Real-Time Traffic Congestion Detection and Prediction using Apache Kafka and Apache Spark

Real traffic data (UCI **PEMS-SF**, Caltrans PeMS) is replayed through **Kafka**, processed by **Spark Structured
Streaming**, turned into a congestion index, alerts and peak-period statistics, and used to **predict congestion
30 minutes ahead** with a Random Forest. A Streamlit dashboard shows everything live. A separate benchmark compares
Spark with a serial pandas baseline.

```
PEMS-SF files -> preprocessing -> Kafka producer -> topic traffic_data -> Spark Structured Streaming
      -> congestion index / level -> alerts, tick summary, window aggregates -> Streamlit dashboard
      historical analysis (peaks, ranking, t-test) + Random Forest prediction -> dashboard
      benchmark: pandas (serial)  vs  Spark  on the same Parquet files
```

## 1. Dataset (source, URL, what it really contains)

| | |
|---|---|
| Name | PEMS-SF (San Francisco Bay Area freeway occupancy) |
| Donor / source | Marco Cuturi, UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/204/pems-sf (CC BY 4.0) |
| Original data | California Department of Transportation (Caltrans) PeMS: http://pems.dot.ca.gov |
| Citation | M. Cuturi, "Fast global alignment kernels", ICML 2011 |
| Content | 963 detectors x 144 ten-minute slots per day; value = **lane occupancy rate (0..1)** |

### Dataset findings (measured on the supplied files, they differ from the dataset description)

1. **Only occupancy is available.** No speed, no flow, no vehicle count, so the congestion index is occupancy-based
   and the dashboard shows no speed/flow. Nothing was invented.
2. **No timestamps in the files.** Chronological order is recovered with the inverse of `randperm`; the weekday
   chain check score is 0.30 in stored order and 1.00 after restoring it (see `output/results/preprocessing_report.json`).
3. **Only 267 of the 440 supplied days are distinct** (173 are exact copies). They are removed; otherwise the same
   day would sit in both train and test.
4. **Weekday labels: 1 = Sunday ... 7 = Saturday** (inferred from the data: labels 1 and 7 have the lowest
   occupancy and the skipped days line up with Monday holidays). The UCI text says Monday-first.
5. **Dates are approximate.** The first day is anchored at 2008-01-02; only the first gaps match known holidays.
   `day_index`, weekday and time of day are the reliable time axes.
6. **No coordinates**, hence no map.

The Kafka stream is a **replay of these real recorded observations** at a controlled speed. No traffic values are generated.

## 2. Project structure

Folders were renamed from the template to avoid shadowing the real `kafka` and `statistics` Python packages:
`kafka/ -> ingestion/`, `statistics/ -> stats/`, `spark/ -> spark_jobs/`.

```
config/config.yaml            all parameters (thresholds, horizon, delay, Kafka, Spark, alerts, benchmark)
common/                       config, congestion index, alert manager, Spark session, data loading
preprocessing/preprocess.py   raw files -> cube, metadata, daily Parquet (raw files are never modified)
ingestion/                    kafka_utils.py (topic, producer), producer.py (real-data replay)
spark_jobs/                   spark_streaming.py, congestion_detection.py, aggregation.py
analysis/historical.py        calibration, peak periods, location ranking, figures
stats/t_test.py               peak vs off-peak t-tests
prediction/                   feature_engineering.py, train_model.py, tune_validation.py, predict.py
performance/                  pandas_baseline.py, spark_benchmark.py, benchmark.py
dashboard/app.py              Streamlit dashboard
tests/                        pytest tests
output/results|models|alerts|stream
docs/build_report.py          rebuilds REPORT.md numbers from output/results
run_project.bat               Windows launcher
```

## 3. Windows setup (exact commands)

**Prerequisites (install once)**

| Tool | Version | Notes |
|---|---|---|
| Python | 3.10 - 3.12 | tick "Add to PATH" |
| Java JDK | **17** | `setx JAVA_HOME "C:\Program Files\Eclipse Adoptium\jdk-17..."` ; check `java -version` |
| Apache Kafka | 3.x or 4.x binary (KRaft, no ZooKeeper) | https://kafka.apache.org/downloads - unzip to a **short path**, e.g. `C:\kafka`; `setx KAFKA_HOME C:\kafka` |
| winutils (Hadoop for Windows) | for Hadoop 3.x | put `winutils.exe` (and `hadoop.dll`) in `C:\hadoop\bin`; `setx HADOOP_HOME C:\hadoop`. Spark checkpointing on Windows usually fails without it. |

Open a **new** terminal after `setx`. If Kafka misbehaves on Windows, run the broker in WSL2 instead; the project still connects to `localhost:9092`.

```bat
cd traffic-congestion-project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
(or `run_project.bat setup`)

**Data:** copy the six files of `pems_sf.zip` into `data\raw\` (or copy the zip next to `run_project.bat`).

## 4. Run: offline stages (results, figures, model, benchmark)

```bat
python -m preprocessing.preprocess --zip pems_sf.zip   :: or without --zip if data\raw is filled
python -m analysis.historical          :: calibration, peak periods, ranking, figures
python -m stats.t_test                 :: t-test
python -m prediction.tune_validation   :: validation-only model selection (optional, ~1 min)
python -m prediction.train_model       :: Random Forest + baselines
python -m performance.benchmark        :: pandas vs Spark (several minutes)
python -m performance.benchmark --scaling   :: Spark local[k] scalability (needs >= 2 cores)
```
Everything is also available as `run_project.bat analysis`. Outputs: `output\results\*.csv|json`, `output\results\figures\*.png`, `output\models\`.

Report and slides are generated from the real result files (re-run after repeating the benchmark on your own computer):

```bat
python -m docs.make_architecture_figure
python docs\build_report.py          :: writes REPORT.md and PRESENTATION_CONTENT.md
python docs\make_docx.py             :: REPORT.docx for printing (needs pandoc and python-docx)
python -m analysis.verify_stream --label "Kafka demo run"   :: verifies a streaming run against the real data
```

## 5. Run: live demo (exact order)

1. **Start Kafka** (own window, keep open): `run_project.bat kafka`
   (manual: `kafka-storage.bat random-uuid`, then `kafka-storage.bat format -t <uuid> -c %KAFKA_HOME%\config\kraft\server.properties`,
   then `%KAFKA_HOME%\bin\windows\kafka-server-start.bat %KAFKA_HOME%\config\kraft\server.properties`; for Kafka 4.x the config is `config\server.properties`).
2. **Start everything else:** `run_project.bat demo` opens three windows in this order:
   1. topic `traffic_data` is created fresh (`python -m ingestion.producer --topic-only --recreate-topic`)
   2. **Spark Structured Streaming** (`python -m spark_jobs.spark_streaming --reset`); the first run downloads the Kafka connector from Maven (internet needed)
   3. **Streamlit** (`streamlit run dashboard\app.py`, http://localhost:8501)
   4. **Producer** (`python -m ingestion.producer`) replays the first weekday of the held-out test period at `stream_delay_seconds` per 10-minute tick (0.5 s: one day ~ 72 s)
3. In the dashboard show: live KPIs and trend -> congested sensors -> **alerts** (morning rush starts ~07:00) -> **prediction** (live check compares predictions made 30 min ago with reality) -> Statistical Analysis -> Performance Analysis.

Manual fallback (four terminals, venv activated): `python -m ingestion.producer --topic-only --recreate-topic`, then
`python -m spark_jobs.spark_streaming --reset`, then `streamlit run dashboard\app.py`, then `python -m ingestion.producer`.
Useful options: `--delay 0.2`, `--day 200 --num-days 2`, env var `STREAM_DELAY_SECONDS=0.1`.
Kafka throughput: `run_project.bat bench` (`--max-speed`, result in `output\results\producer_stats_maxspeed.json`).

If Kafka goes down: the producer reports the connection error and exits; Spark keeps its checkpointed offsets and resumes when you restart it without `--reset`.

## 6. Tests

```bat
python -m pytest -v                 :: all tests (Spark test starts a JVM, ~30 s)
python -m pytest -v -m "not slow"   :: skip the Spark test
```
Covered: raw parsing and chronology recovery, duplicate removal, weekday-encoding inference, value cleaning, congestion index and levels, alert de-duplication/cooldown, Spark-vs-NumPy level equality, feature generation and no-leakage, real processed data checks.

## 7. Method summary

* **Congestion index** `CI = clip(occupancy / ref, 0, 1)`; `ref` = 99th percentile of TRAIN occupancy. Levels from the 90th / 95th / 99th percentiles of TRAIN occupancy: LOW < MODERATE < HIGH < SEVERE. These are *relative* (percentile-based) levels, not absolute engineering limits; all values in `config.yaml`.
* **Peak periods**: (weekday, hour) cells whose mean network CI is above the 80th percentile of the 168 cells.
* **Alerts**: HIGH or above; one alert per sensor episode (+ escalation), cooldown 6 ticks.
* **Prediction**: Random Forest on lags/rolling means/trends/time/sensor baseline, target = level 3 ticks (30 min) ahead, chronological split by day (first 70% of days train), compared with a persistence baseline.
* **t-test**: paired t-test on daily network means, peaks defined on training days and tested on later days.
* **Benchmark**: same workload (CI + weekday x hour profile + per-sensor stats) on identical Parquet files; results checked equal.

## 8. Honest execution status

Executed in the development sandbox (1 vCPU, 4 GB): preprocessing, historical analysis, t-test, model tuning/training, the full pandas-vs-Spark benchmark, the unit tests, the Spark Structured Streaming job **through its offline file-source test mode**, and the dashboard (headless test).
**NOT YET EXECUTED here: a real Kafka broker, the producer writing to Kafka, and Spark's Kafka source** (broker and connector downloads are blocked in the sandbox). The Kafka code paths follow the documented kafka-python and Spark APIs; run the demo above on your machine and record the screenshots and Kafka throughput yourself.
Benchmark numbers in the report come from the sandbox; **re-run `python -m performance.benchmark` on your own computer** and rebuild the report with `python docs\build_report.py`.

## 9. Limitations

Occupancy only; percentile-based levels; the replayed stream is historical; alert state lives in the Spark driver (lost on restart); the last window of a replay is emitted only when later data arrives (watermark); approximate dates; single-node Kafka and Spark local mode.

## 10. Troubleshooting (Windows)

* `JAVA_HOME is not set` / `Unsupported class file major version`: use Java 17.
* `winutils.exe` / `HADOOP_HOME` errors at checkpointing: install winutils as above.
* `NoBrokersAvailable`: the broker is not running or the port 9092 is blocked.
* First Spark run is slow: it downloads `spark-sql-kafka-0-10_2.12` once.
* Dashboard says "Waiting for the stream": start Spark and the producer; it reads `output\stream`.
#   t r a f f i c - c o n g e s t i o n - p r o j e c t  
 