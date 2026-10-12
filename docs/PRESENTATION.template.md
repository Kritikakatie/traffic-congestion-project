# Presentation Content - Real-Time Traffic Congestion Detection and Prediction

*Speaker assignments below are a TEMPLATE: change them to match who really worked on each part and who will actually speak.*
*Numbers come from `output/results/` (rebuilt with `python docs\build_report.py`). Have the dashboard and a terminal ready.*

---

## Slide 1 - Problem and Objective

- Detect congested road segments and peak periods from vehicle-sensor data, raise alerts, predict congestion
- Decisions are only useful if made while congestion is developing, so: streaming
- Objectives: real dataset, Kafka + Spark pipeline, congestion index, peak periods, alerts, prediction (+{{horizon}} min), t-test, serial-vs-Spark performance

**Speaker notes (Member 1):** "Traffic jams build up within minutes. We want a system that reads sensor data continuously, flags congested locations and warns early. Our lab concepts are Kafka for streaming transport and Spark for distributed stream processing."

## Slide 2 - Dataset / Input

- UCI **PEMS-SF** (Caltrans PeMS): https://archive.ics.uci.edu/dataset/204/pems-sf
- 963 freeway detectors, lane **occupancy** every 10 minutes, one matrix per day
- After cleaning: {{n_distinct}} distinct days, {{n_records}} observations
- Findings: only occupancy (no speed/flow); no timestamps (order recovered with `randperm`); {{n_dups}} of {{n_stored}} supplied days were exact duplicates (removed); no coordinates
- Kafka replays these REAL records: nothing is generated

**Speaker notes (Member 1):** "The files have no column names or dates, so we reconstructed the chronological order and verified it with the weekday sequence. We also found that 173 days were duplicated; keeping them would leak test data into training."

## Slide 3 - Architecture / Workflow

- Dataset -> preprocessing -> **Kafka producer** -> topic `traffic_data` -> **Spark Structured Streaming** -> congestion index -> alerts / windows -> **dashboard**
- Side branches: historical analysis (peaks, ranking, t-test), Random Forest prediction, benchmark (pandas vs Spark)
- Kafka: ordered per sensor (key = sensor id), 3 partitions. Spark: 30-min event-time windows, watermark, checkpoint

**Speaker notes (Member 2):** "The producer sends 963 messages per 10-minute tick, keyed by sensor so each sensor's readings stay in order. Spark is the consumer. Why both: Kafka transports and stores the stream, Spark computes on it."

## Slide 4 - Methodology / Algorithm

- Congestion index: CI = clip(occupancy / {{ref}}, 0, 1); levels at the 90/95/99th percentile of training occupancy
- Peak = (weekday, hour) cells above the {{peak_pct}}th percentile of mean CI
- Alerts: one per sensor episode (+ escalation), cooldown
- Prediction: Random Forest, lags and rolling means, target = level in {{horizon}} min, chronological split
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
- Most congested sensors: {{top3_ids}} (top three)
- t-test (held-out days): peak {{peak_mean}} vs off-peak {{off_mean}}, t = {{t_stat}}, p = {{t_p}} -> {{t_verdict}}
- Prediction (+{{horizon}} min): Random Forest accuracy {{rf_acc}}, macro F1 {{rf_mf1}} (persistence baseline: {{pe_acc}}, {{pe_mf1}})
- Streaming check ({{sv_source}}): {{sv_records}} records processed, levels agree with NumPy for {{sv_agree}}%, {{sv_raw}} raw HIGH+ observations reduced to {{sv_alerts}} alerts

**Speaker notes (Member 3 and 4):** "The model is modestly better than the 'nothing changes' baseline; we report that honestly. The alert logic cuts duplicate alerts by a factor of {{sv_red}}."

## Slide 8 - Performance Analysis

{{bench_table}}

- Measured on: {{bench_env}}
- {{crossover}}
- Why: Spark's fixed overhead (JVM, planning, scheduling) dominates small inputs; it is amortised on large data; more cores/machines would shift the crossover earlier
- {{kafka_text}}

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
