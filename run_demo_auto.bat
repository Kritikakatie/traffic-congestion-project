@echo off
setlocal

cd /d C:\dev\traffic-congestion-project

echo ==================================================
echo TRAFFIC CONGESTION PROJECT - LIVE DEMO
echo ==================================================

REM ==================================================
REM ENVIRONMENT
REM ==================================================

call venv\Scripts\activate.bat

set "JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"
set "KAFKA_HOME=C:\kafka"
set "HADOOP_HOME=C:\hadoop"
set "SPARK_HOME=%CD%\venv\Lib\site-packages\pyspark"
set "PYSPARK_PYTHON=%CD%\venv\Scripts\python.exe"
set "PYSPARK_DRIVER_PYTHON=%CD%\venv\Scripts\python.exe"

set "PATH=%JAVA_HOME%\bin;%HADOOP_HOME%\bin;%PATH%"

echo.
echo [1/4] Checking Java...
java -version

REM ==================================================
REM START KAFKA
REM ==================================================

echo.
echo [2/4] Starting Kafka...

start "Kafka Broker" cmd /k "cd /d C:\kafka && set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot && .\bin\windows\kafka-server-start.bat .\config\server.properties"

echo.
echo Waiting for Kafka to become available...

:WAIT_KAFKA
powershell -NoProfile -Command "$c = Test-NetConnection localhost -Port 9092 -WarningAction SilentlyContinue; if ($c.TcpTestSucceeded) { exit 0 } else { exit 1 }"

if errorlevel 1 (
    echo Kafka is not ready yet...
    C:\Windows\System32\timeout.exe /t 3 /nobreak >nul
    goto WAIT_KAFKA
)

echo Kafka is READY on localhost:9092!
REM ==================================================
REM START PROJECT DEMO
REM ==================================================

echo.
echo [3/4] Starting project demo...

call run_project.bat demo

echo.
echo ==================================================
echo [4/4] DEMO COMMAND FINISHED
echo ==================================================
echo.
echo Kafka window should remain open.
echo Close it with Ctrl+C when you are finished.
echo.

endlocal