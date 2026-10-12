# Presentation Content - Real-Time Traffic Congestion Detection and Prediction

*Speaker assignments below are a TEMPLATE: change them to match who really worked on each part and who will actually speak.*
*Numbers come from `output/results/` (rebuilt with `python docs\build_report.py`). Have the dashboard and a terminal ready.*

---

## Slide 1 - Problem and Objective

- Detect congested road segments and peak periods from vehicle-sensor data, raise alerts, predict congestion
- Decisions are only useful if made while congestion is developing, so: streaming
- Objectives: real dataset, Kafka + Spark pipeline, congestion index, peak periods, alerts, prediction (+30 min), t-test, serial-vs-Spark performance

**Speaker notes (Member 1):** "Traffic jams build up within minutes. We want a system that reads sensor data continuously, flags congested locations and warns early. Our lab concepts are Kafka for streaming transport and Spark for distributed stream processing."

## Slide 2 - Dataset / Input

- UCI **PEMS-SF** (Caltrans PeMS): https://archive.ics.uci.edu/dataset/204/pems-sf
- 963 freeway detectors, lane **occupancy** every 10 minutes, one matrix per day
- After cleaning: 267 distinct days, 37,025,424 observations
- Findings: only occupancy (no speed/flow); no timestamps (order recovered with `randperm`); 173 of 440 supplied days were exact duplicates (removed); no coordinates
- Kafka replays these REAL records: nothing is generated

**Speaker notes (Member 1):** "The files have no column names or dates, so we reconstructed the chronological order and verified it with the weekday sequence. We also found that 173 days were duplicated; keeping them would leak test data into training."

## Slide 3 - Architecture / Workflow

- Dataset -> preprocessing -> **Kafka producer** -> topic `traffic_data` -> **Spark Structured Streaming** -> congestion index -> alerts / windows -> **dashboard**
- Side branches: historical analysis (peaks, ranking, t-test), Random Forest prediction, benchmark (pandas vs Spark)
- Kafka: ordered per sensor (key = sensor id), 3 partitions. Spark: 30-min event-time windows, watermark, checkpoint

**Speaker notes (Member 2):** "The producer sends 963 messages per 10-minute tick, keyed by sensor so each sensor's readings stay in order. Spark is the consumer. Why both: Kafka transports and stores the stream, Spark computes on it."

## Slide 4 - Methodology / Algorithm

- Congestion index: CI = clip(occupancy / 0.2552, 0, 1); levels at the 90/95/99th percentile of training occupancy
- Peak = (weekday, hour) cells above the 80th percentile of mean CI
- Alerts: one per sensor episode (+ escalation), cooldown
- Prediction: Random Forest, lags and rolling means, target = level in 30 min, chronological split
- t-test: paired, daily means, peaks defined on training days, tested on later days

**Speaker notes (Member 3 for index/peaks/alerts, Member 4 for prediction and t-test):** "Only occupancy exists, so the index uses occupancy. Thresholds are relative percentiles from training days only. For the t-test the unit is one day because neighbouring readings are correlated."

## Slide 5 - Implementation / Code

- Modules: `ingestion/producer.py`, `spark_jobs/spark_streaming.py`, `common/congestion.py`, `analysis/historical.py`, `prediction/train_model.py`, `performance/benchmark.py`, `dashboard/app.py`
- All parameters in `config/config.yaml`; 20 automated tests
- Spark: `readStream` (Kafka) -> `from_json` -> CI/level -> `foreachBatch` + windowed `groupBy(window, sensor)`

**Speaker notes (Member 2 and 3):** show the producer loop and the `spark_streaming.py` query definitions; explain trigger (3 s), watermark (10 min) and checkpoint.

## Slide 6 - Live Execution

1. Start Kafka (`run_project.bat kafka`)
2. `run_project.bat demo` -> topic, Spark streaming, dashboard, producer
3. Show: incoming ticks and KPIs -> congested sensors -> alert -> prediction live check -> statistics -> performance
- Replay speed: 0.5 s per tick (one day = about 72 s)

**Speaker notes (all, rehearsed order):** Member 2 starts Kafka and the producer; Member 3 explains the Spark window and alerts; Member 4 walks through prediction, statistics and performance tabs.

## Slide 7 - Output / Results

- Peak periods: weekday mornings and afternoons (see peak table); no weekend peaks
- Most congested sensors: 401434, 400696, 401356 (top three)
- t-test (held-out days): peak 0.0930 vs off-peak 0.0424, t = 73.69, p = 1.16e-56 -> REJECT H0
- Prediction (+30 min): Random Forest accuracy 0.938, macro F1 0.658 (persistence baseline: 0.929, 0.649)
- Streaming check (Spark Structured Streaming via OFFLINE file source (no Kafka broker), day 189): 138,672 records processed, levels agree with NumPy for 100.00%, 5,741 raw HIGH+ observations reduced to 977 alerts

**Speaker notes (Member 3 and 4):** "The model is modestly better than the 'nothing changes' baseline; we report that honestly. The alert logic cuts duplicate alerts by a factor of 5.9."

## Slide 8 - Performance Analysis

| Days | Records | Serial pandas (s) | Spark (s) | Speedup | pandas rec/s | Spark rec/s |
|---|---|---|---|---|---|---|
| 1.00 | 138,672.00 | 0.19 | 1.10 | 0.17 | 747,048.03 | 125,948.30 |
| 4.00 | 554,688.00 | 0.21 | 1.04 | 0.20 | 2,628,392.76 | 532,395.03 |
| 7.00 | 970,704.00 | 0.34 | 1.04 | 0.33 | 2,856,555.33 | 930,278.36 |
| 15.00 | 2,080,080.00 | 0.59 | 1.15 | 0.52 | 3,505,728.83 | 1,814,892.45 |
| 36.00 | 4,992,192.00 | 1.10 | 2.18 | 0.50 | 4,549,880.85 | 2,285,924.17 |
| 72.00 | 9,984,384.00 | 2.29 | 2.67 | 0.85 | 4,368,877.55 | 3,732,751.65 |
| 267.00 | 37,025,424.00 | 9.29 | 7.79 | 1.19 | 3,985,283.03 | 4,754,999.20 |

- Measured on: 1 CPU core(s), 3.9 GB RAM, Python 3.12.3, PySpark 3.5.9, pandas 3.0.2, Spark master local[*]
- Spark was slower than the serial baseline for datasets up to 9,984,384 records and faster from 37,025,424 records (1.19x). The crossover therefore lies between 9,984,384 and 37,025,424 records on this machine.
- Why: Spark's fixed overhead (JVM, planning, scheduling) dominates small inputs; it is amortised on large data; more cores/machines would shift the crossover earlier
- Kafka throughput: NOT YET EXECUTED - requires a Kafka broker. Run `run_project.bat bench` (`python -m ingestion.producer --max-speed --num-days 5`) and fill in `output\results\producer_stats_maxspeed.json`, then re-run `python docs\build_report.py`.

**Speaker notes (Member 4):** "We do not hide that Spark is slower on small data. Speedup is serial time divided by Spark time."

## Slide 9 - Conclusion

- Complete Kafka -> Spark pipeline on a real dataset: detection, peaks, alerts, prediction, statistics
- Honest findings: occupancy-only data, duplicates removed, modest gain over persistence, Spark pays off only at scale
- Future: speed/flow data, coordinates and map, multi-node cluster, better models

**Speaker notes (Member 1):** summarise in 30 seconds and invite questions.

## Slide 10 - Individual Contribution (TEMPLATE - edit before use)

| Member | Contribution |
|---|---|
| Member 1 | Dataset study, preprocessing |
| Member 2 | Kafka producer, streaming ingestion |
| Member 3 | Spark streaming, congestion detection, peak analysis |
| Member 4 | Prediction, statistics, dashboard, performance analysis |

*State only what each member actually did; every member must be able to explain the whole pipeline.*

## Slide 11 - Viva / Questions

- Why Kafka? Why Spark? Kafka vs Spark?
- What is a topic, producer, consumer, DataFrame, window, watermark?
- How is congestion computed and why these thresholds?
- How were peaks found? How was leakage avoided? What does the p-value mean?
- Why was Spark slower on small data? What if Kafka goes down?
- See `VIVA_QUESTIONS.md` (45 questions with answers)
