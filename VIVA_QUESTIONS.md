# Viva Questions and Answers

Answers are written for undergraduate level and match what THIS project actually does. Numbers you may be asked for
(accuracy, t-statistic, speedup ...) must be read from `output/results/` or the dashboard, not memorised from here.

## A. Problem, data and design

**1. What problem does the project solve?**
It finds congested road locations and peak periods from vehicle-sensor data, raises congestion alerts, predicts congestion 30 minutes ahead and shows it all live, using Kafka and Spark.

**2. Which dataset did you use and where is it from?**
UCI PEMS-SF (donor Marco Cuturi), originally Caltrans PeMS, https://archive.ics.uci.edu/dataset/204/pems-sf. 963 freeway detectors in the San Francisco Bay Area, one lane-occupancy value (0 to 1) every 10 minutes per day.

**3. What does "occupancy" mean and why can it show congestion?**
Occupancy is the fraction of time a detector is covered by vehicles. When traffic is dense and slow, vehicles stay over the detector longer, so occupancy rises. It is a standard congestion indicator.

**4. Why do you not show speed or vehicle count?**
The dataset contains only occupancy. Inventing speed or flow would be fabrication, so the congestion index uses occupancy only and we say so openly.

**5. What problems did you find in the dataset?**
(a) only occupancy, (b) no timestamps, (c) only 267 distinct days out of 440 supplied (173 exact duplicates), (d) weekday labels contradict the documentation (1 = Sunday according to the data), (e) no coordinates, (f) approximate dates only.

**6. How did you recover the time order without timestamps?**
The dataset came with `randperm`, the shuffle UCI applied. Applying its inverse restores chronological order. We verified it: consecutive days then advance the weekday by one (or two when a holiday was removed) almost always. The check score is about 0.30 in stored order and 1.00 after restoring.

**7. Why did you remove duplicate days?**
Exact copies carry no new information, and if one copy fell in the training set and another in the test set the model would be tested on data it had already seen (data leakage), inflating accuracy.

**8. Why is traffic data suitable for streaming?**
Sensors produce a continuous, ordered flow of small events, and decisions (alerts, rerouting) are only useful if they are made within seconds or minutes.

**9. Is your stream real-time?**
It is a *replay* of real recorded observations at a controlled speed (default 0.5 s per 10-minute tick). The observations are real; only their arrival pace is simulated. The project says this explicitly.

## B. Kafka

**10. What is Apache Kafka?**
A distributed, durable publish-subscribe log. Producers append messages to topics; consumers read them at their own pace and offsets.

**11. What is a topic? Which topic do you use?**
A named stream of messages, split into partitions. We use `traffic_data` with 3 partitions.

**12. What is a producer and what does yours do?**
A client that publishes messages. `ingestion/producer.py` reads the processed real data, builds one JSON message per sensor per 10-minute tick (963 messages), sends them to Kafka keyed by sensor id and sleeps `stream_delay_seconds`.

**13. What is a consumer? Who consumes?**
A client that reads from a topic. Spark Structured Streaming is the consumer (its Kafka source).

**14. Why key messages by sensor id?**
Messages with the same key go to the same partition, so each sensor's observations stay in order.

**15. Why use Kafka at all instead of Spark reading the CSV?**
Kafka decouples data producers from processors, buffers bursts, keeps data for replay, allows several independent consumers and scales by partitions. Reading a file would be batch processing, not streaming.

**16. What happens if Kafka goes down?**
The producer reports `NoBrokersAvailable`/send errors and stops cleanly. Messages already in Kafka are on disk. Spark stores its read offsets in its checkpoint, so after restarting it continues where it stopped (do not use `--reset`).

## C. Spark

**17. What is Apache Spark / PySpark?**
A distributed data-processing engine; PySpark is its Python API. Work is split into tasks that run in parallel on many cores or machines.

**18. What is a Spark DataFrame?**
A distributed table with named, typed columns. Operations are lazy: Spark builds a plan and optimises it before running.

**19. What is distributed processing?**
Splitting data and computation over several workers that run in parallel and combine their partial results.

**20. What is Spark Structured Streaming?**
A streaming engine where a stream is treated as an unbounded table that grows. You write normal DataFrame code; Spark runs it incrementally in micro-batches.

**21. Batch vs streaming?**
Batch processes a finished, bounded dataset once. Streaming processes unbounded data continuously as it arrives, with state (windows) and low latency.

**22. Why windowing, and what are watermarks?**
One 10-minute value is noisy; a 30-minute window per sensor gives stable averages. Event-time windows group by when the observation happened. A watermark (10 minutes here) tells Spark how late data may be, so it can finalise old windows and free memory.

**23. What do the trigger and checkpoint do?**
The trigger (3 s) sets how often micro-batches run. The checkpoint stores progress (offsets and state) so the query can recover after failure.

**24. Kafka vs Spark - are they competitors?**
No. Kafka transports and stores event streams; Spark computes on them. Kafka is the pipe and storage, Spark is the processing engine.

**25. What does `foreachBatch` do in your job?**
For every micro-batch it runs our code: writes records, builds the per-tick summary, runs the alert manager and records throughput metrics.

## D. Congestion, peaks, alerts

**26. How is congestion calculated?**
`CI = clip(occupancy / reference, 0, 1)`; the reference is the 99th percentile of training occupancy. Levels: MODERATE from the 90th, HIGH from the 95th, SEVERE from the 99th percentile (converted to CI units). Everything is configurable in `config.yaml`.

**27. Why these thresholds? Are they scientific limits?**
They are data-driven and relative: with only occupancy we cannot use engineering speed limits. They are computed on training days only. The honest limitation: by construction about 10% of observations are above LOW. They can be replaced by manual thresholds.

**28. How are peak periods identified?**
We average the congestion index for each (weekday, hour) cell over all days and call a cell a peak if it is above the 80th percentile of the 168 cells. We did not assume 8 AM / 6 PM; the data gives weekday mornings and afternoons (see `peak_periods.csv`) and no weekend peaks.

**29. How do you avoid thousands of duplicate alerts?**
One alert per sensor per congestion episode, one extra ESCALATION alert if the level rises, an episode ends after 3 ticks below the threshold, and a 6-tick cooldown prevents flapping. The dashboard compares raw HIGH+ sensor-ticks to alerts raised.

## E. Prediction

**30. What is predicted and how is the target made?**
The congestion level (LOW..SEVERE) of the same sensor 3 slots (30 minutes) later. The target is the level computed from the CI at slot t+3.

**31. Why Random Forest?**
It handles non-linear relations and mixed features, needs little scaling, is robust and quick to train on a laptop, and gives feature importances that are easy to explain.

**32. Which features?**
Current CI, three lags, rolling means over 30 and 60 minutes, two trend differences, slot of day, weekday, weekend flag and the sensor's average CI from training days.

**33. How was data leakage avoided?**
Features use only slots up to t; the target is after t; the sensor baseline and the calibration use training days only; duplicates were removed; train and test are different days in chronological order; model settings were chosen on a validation split inside the training days.

**34. Why a chronological split instead of a random one?**
In forecasting, the model must predict the future from the past. A random split lets it see days after the ones it is tested on, which is unrealistic and optimistic.

**35. How did you handle class imbalance, and was the model really better than a trivial one?**
HIGH and SEVERE are rare (about 5% together). We report macro and per-class metrics, not accuracy alone, and tested class weighting on validation data; it lowered accuracy and macro F1, so it is off (configurable). We also compare with a persistence baseline ("future = now") and an always-LOW baseline. The Random Forest beats persistence only modestly; this is stated in the report rather than hidden.

## F. Statistics

**36. What is a t-test and what is your null hypothesis?**
A t-test checks whether a difference between means is larger than chance. H0: mean occupancy in peak periods equals mean occupancy off-peak; H1: they differ.

**37. What does the p-value mean?**
The probability of seeing a difference at least this large if H0 were true. If p < alpha = 0.05 we reject H0. It is not the probability that H0 is true and does not measure importance (we also report the effect size, Cohen's d).

**38. Why paired, why days, why train/test peaks?**
Peak and off-peak means come from the same day, so a paired test is appropriate. Neighbouring 10-minute values are strongly correlated, so one day is the independent unit. Peaks are selected for being high, so we define them on training days and test on later days to avoid circular reasoning.

## G. Performance

**39. What is speedup? Was Spark faster?**
Speedup = serial time / Spark time. Read the real values from `benchmark_results.csv`. On small data Spark is slower because of fixed overheads (job planning, task scheduling, JVM); the serial baseline is a vectorised pandas pass. Spark pays off as data and available cores grow, and the dashboard states honestly where (or whether) the crossover happened on the measured machine.

**40. Why might Spark be slower on small datasets?**
Starting a JVM, planning the query, shipping tasks and shuffling data cost a roughly constant time; for 100K rows the actual work is milliseconds.

**41. How does the system scale?**
More Kafka partitions allow more parallel consumers; Spark splits work into partitions processed in parallel on more cores or machines; the benchmark has a `local[k]` scalability test. Single-node limits are stated.

**42. Why not MPI, OpenMP or CUDA?**
The problem is event-stream processing with aggregation and fault tolerance, which Kafka and Spark provide. MPI/OpenMP target tightly-coupled numeric computation, and CUDA needs GPUs; none of them offers topics, offsets, windows or recovery. The rubric asks for at least two lab concepts; we use Kafka and Spark.

## H. Wrap-up

**43. What happens with missing or invalid values?**
Preprocessing marks NaN or out-of-range values (<0 or >1) invalid and repairs them by interpolation along time (sensor median as fallback); genuine zeros are kept. In streaming, records with missing key fields are filtered out. The supplied data had no NaN.

**44. What are the limitations?**
Occupancy only; relative thresholds; replayed history; approximate dates; local single-node Spark and Kafka; alert state lives in the Spark driver; modest gain over the persistence baseline.

**45. What future improvements are possible?**
Add speed/flow from full PeMS data and absolute thresholds, sensor coordinates for a map, multi-node Kafka/Spark, gradient boosting or sequence models, spatial features from neighbouring sensors, persistent alert state, Spark MLlib for model scoring inside the stream.
