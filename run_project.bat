@echo off
REM ============================================================================
REM  Real-Time Traffic Congestion Detection and Prediction  -  Windows launcher
REM  Usage:  run_project.bat <command>
REM    setup     create venv and install requirements (once)
REM    analysis  preprocess + analysis + t-test + model training + benchmark (offline stages)
REM    kafka     start a single-node Kafka broker (KRaft mode) in a new window
REM    demo      (Kafka must be running) topic + Spark streaming + producer + dashboard
REM    bench     Kafka max-throughput test (Kafka must be running)
REM    test      run the unit tests
REM  Needed environment variables:  JAVA_HOME (Java 17), KAFKA_HOME (e.g. C:\kafka),
REM                                 HADOOP_HOME (folder containing bin\winutils.exe)
REM ============================================================================
setlocal
cd /d "%~dp0"
set PY=venv\Scripts\python.exe
if "%1"=="" goto usage
if /i "%1"=="setup" goto setup
if /i "%1"=="analysis" goto analysis
if /i "%1"=="kafka" goto kafka
if /i "%1"=="demo" goto demo
if /i "%1"=="bench" goto bench
if /i "%1"=="test" goto test
:usage
echo Usage: run_project.bat setup ^| analysis ^| kafka ^| demo ^| bench ^| test
goto :eof

:setup
python -m venv venv || (echo Python 3.10-3.12 not found on PATH & goto :eof)
%PY% -m pip install --upgrade pip
%PY% -m pip install -r requirements.txt
echo.
echo Setup done. Check Java:   java -version   (must be 17)
goto :eof

:checkenv
if "%JAVA_HOME%"=="" echo [WARNING] JAVA_HOME is not set - Spark needs Java 17
if "%HADOOP_HOME%"=="" echo [WARNING] HADOOP_HOME is not set - Spark checkpoints need winutils.exe on Windows (see README)
exit /b 0

:analysis
call :checkenv
if not exist "%~dp0data\raw\PEMS_train" (
  if exist "%~dp0pems_sf.zip" ( %PY% -m preprocessing.preprocess --zip pems_sf.zip ) else (
    echo Put the 6 PEMS-SF files into data\raw  OR  copy pems_sf.zip next to this script & goto :eof ) 
) else ( %PY% -m preprocessing.preprocess )
%PY% -m analysis.historical || goto :fail
%PY% -m stats.t_test || goto :fail
%PY% -m prediction.tune_validation || goto :fail
%PY% -m prediction.train_model || goto :fail
%PY% -m performance.benchmark || goto :fail
%PY% -m performance.benchmark --scaling
echo.
echo Offline stages finished. Results: output\results   Figures: output\results\figures
goto :eof

:kafka
if "%KAFKA_HOME%"=="" (echo Set KAFKA_HOME first, e.g. set KAFKA_HOME=C:\kafka & goto :eof)
set KCFG=%KAFKA_HOME%\config\kraft\server.properties
if not exist "%KCFG%" set KCFG=%KAFKA_HOME%\config\server.properties
for /f %%i in ('"%KAFKA_HOME%\bin\windows\kafka-storage.bat" random-uuid') do set KUUID=%%i
call "%KAFKA_HOME%\bin\windows\kafka-storage.bat" format -t %KUUID% -c "%KCFG%" --ignore-formatted
start "Kafka broker" cmd /k ""%KAFKA_HOME%\bin\windows\kafka-server-start.bat" "%KCFG%""
echo Kafka starting in a new window (wait ~15 s, keep it open).
goto :eof

:demo
call :checkenv
echo [1/4] Creating a clean topic traffic_data ...
%PY% -m ingestion.producer --topic-only --recreate-topic || (echo Kafka is not reachable - run: run_project.bat kafka & goto :eof)
echo Starting Spark Structured Streaming ...
start "Spark Structured Streaming" cmd /k "%PY% -m spark_jobs.spark_streaming --reset"
echo [2/4] Waiting 40 s for Spark to start (first run downloads the Kafka connector jar) ...
timeout /t 40 /nobreak >nul
echo [3/4] Starting Streamlit dashboard ...
start "Dashboard" cmd /k "%PY% -m streamlit run dashboard\app.py"
timeout /t 5 /nobreak >nul
echo [4/4] Starting Kafka producer (replays real observations) ...
start "Kafka producer" cmd /k "%PY% -m ingestion.producer"
echo.
echo Demo running. Dashboard: http://localhost:8501   (Ctrl+C in each window to stop)
goto :eof

:bench
%PY% -m ingestion.producer --max-speed --num-days 5 --recreate-topic
goto :eof

:test
%PY% -m pytest -v
goto :eof

:fail
echo A stage failed - see the message above.
exit /b 1
