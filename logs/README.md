# logs/ - System Monitoring & Diagnostics

Centralized logging infrastructure for Tweezer system services with automated log rotation, structured output, and real-time monitoring capabilities. Captures all service lifecycle events, performance metrics, and diagnostic information.

## 📖 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Directory Structure](#directory-structure)
- [Log Formats](#log-formats)
- [Service-Specific Logging](#service-specific-logging)
- [Analysis Tools](#analysis-tools)
- [Troubleshooting Guide](#troubleshooting-guide)

## 🏗️ Architecture Overview

The logging system implements a multi-tier architecture with automatic rotation and structured output:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        LOGGING ARCHITECTURE OVERVIEW                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │                           Service Processes                                │
│  │                                                                             │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  │   Arduino    │  │   Camera     │  │     SLM      │  │      GUI     │   │
│  │  │   Service    │  │   Service    │  │   Services   │  │   Dashboard  │   │
│  │  │              │  │              │  │              │  │              │   │
│  │  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │   │
│  │  │ │stdout    │ │  │ │stdout    │ │  │ │stdout    │ │  │ │stdout    │ │   │
│  │  │ │stderr    │ │  │ │stderr    │ │  │ │stderr    │ │  │ │stderr    │ │   │
│  │  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │   │
│  │  │      │       │  │      │       │  │      │       │  │      │       │   │
│  │  └──────┼───────┘  └──────┼───────┘  └──────┼───────┘  └──────┼───────┘   │
│  │         │                 │                 │                 │           │
│  │         │                 │                 │                 │           │
│  │  ┌──────────────────────────────────────────────────────────────────────────┤
│  │  │                   Service Manager Capture Layer                        │
│  │  │                                                                         │
│  │  │  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  │  │            Output Reading Threads (per service)                    │ │
│  │  │  │                                                                    │ │
│  │  │  │  def _read_output(self, service_key: str):                        │ │
│  │  │  │      """Continuously read subprocess output and log."""           │ │
│  │  │  │                                                                    │ │
│  │  │  │      service = self.services[service_key]                         │ │
│  │  │  │      process = service.process                                    │ │
│  │  │  │      log_file = service.log_file_path                             │ │
│  │  │  │                                                                    │ │
│  │  │  │      # Read stdout/stderr line by line                            │ │
│  │  │  │      while process.poll() is None:                                │ │
│  │  │  │          try:                                                     │ │
│  │  │  │              line = process.stdout.readline()                     │ │
│  │  │  │              if line:                                             │ │
│  │  │  │                  timestamp = datetime.now()                       │ │
│  │  │  │                  formatted_line = format_log_entry(              │ │
│  │  │  │                      timestamp, service_key, line                │ │
│  │  │  │                  )                                                │ │
│  │  │  │                                                                    │ │
│  │  │  │                  # Write to log file                              │ │
│  │  │  │                  log_file.write(formatted_line)                   │ │
│  │  │  │                  log_file.flush()                                 │ │
│  │  │  │                                                                    │ │
│  │  │  │                  # Store in buffer for GUI display                │ │
│  │  │  │                  service.output_buffer.append(formatted_line)     │ │
│  │  │  │                                                                    │ │
│  │  │  │          except Exception as e:                                   │ │
│  │  │  │              log_error(e)                                         │ │
│  │  │  └────────────────────────────────────────────────────────────────────┘ │
│  │  │                                                                         │
│  │  │  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  │  │                    Log File Writers                                │ │
│  │  │  │                                                                    │ │
│  │  │  │  Path: logs/service_logs/{service_name}/{service}_{timestamp}.log │ │
│  │  │  │                                                                    │ │
│  │  │  │  Features:                                                         │ │
│  │  │  │  - Automatic timestamping                                          │ │
│  │  │  │  - UTF-8 encoding                                                  │ │
│  │  │  │  - Unbuffered writes (PYTHONUNBUFFERED=1)                          │ │
│  │  │  │  - Command header with launch details                              │ │
│  │  │  │  - Exit status capture                                             │ │
│  │  │  └────────────────────────────────────────────────────────────────────┘ │
│  │  │                                                                         │
│  │  │  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  │  │                  In-Memory Buffer (GUI Display)                    │ │
│  │  │  │                                                                    │ │
│  │  │  │  service.output_buffer = []  # Last 1000 lines                    │ │
│  │  │  │                                                                    │ │
│  │  │  │  - Ring buffer (1000 lines max)                                    │ │
│  │  │  │  - Real-time GUI updates                                           │ │
│  │  │  │  - Searchable/filterable                                           │ │
│  │  │  │  - Color-coded severity                                            │ │
│  │  │  └────────────────────────────────────────────────────────────────────┘ │
│  │  └─────────────────────────────────────────────────────────────────────────┤
│  └─────────────────────────────────────────────────────────────────────────────┘
│                                      │                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │                          Log File Organization                             │
│  │                                                                             │
│  │  logs/                                                                      │
│  │  ├── service_logs/                   # Managed by service_manager.py       │
│  │  │   ├── arduino/                                                           │
│  │  │   │   ├── arduino_service_20240115_143022.log                           │
│  │  │   │   ├── arduino_service_20240115_150834.log                           │
│  │  │   │   └── arduino_service_20240115_162145.log                           │
│  │  │   │                                                                      │
│  │  │   ├── camera/                                                            │
│  │  │   │   ├── camera_service_20240115_143025.log                            │
│  │  │   │   ├── camera_service_20240115_150840.log                            │
│  │  │   │   └── camera_service_20240115_162150.log                            │
│  │  │   │                                                                      │
│  │  │   ├── slm_generator/                                                     │
│  │  │   │   ├── slm_generator_20240115_143028.log                             │
│  │  │   │   ├── slm_generator_20240115_150845.log                             │
│  │  │   │   └── slm_generator_20240115_162155.log                             │
│  │  │   │                                                                      │
│  │  │   ├── slm_driver/                                                        │
│  │  │   │   ├── slm_driver_20240115_143030.log                                │
│  │  │   │   ├── slm_driver_20240115_150850.log                                │
│  │  │   │   └── slm_driver_20240115_162200.log                                │
│  │  │   │                                                                      │
│  │  │   └── dashboard/                                                         │
│  │  │       ├── dashboard_20240115_143032.log                                 │
│  │  │       ├── dashboard_20240115_150855.log                                 │
│  │  │       └── dashboard_20240115_162205.log                                 │
│  │  │                                                                          │
│  │  └── AutoLogs/                       # Application-generated data logs     │
│  │      ├── tracking_results/                                                  │
│  │      │   ├── experiment_001_20240115_143500.hdf5                           │
│  │      │   ├── experiment_001_20240115_143500_metadata.json                  │
│  │      │   └── experiment_001_20240115_143500.csv                            │
│  │      │                                                                      │
│  │      ├── images/                                                            │
│  │      │   ├── frame_20240115_143501_001.tiff                                │
│  │      │   ├── frame_20240115_143501_002.tiff                                │
│  │      │   └── ...                                                            │
│  │      │                                                                      │
│  │      └── performance/                                                       │
│  │          ├── camera_metrics_20240115.csv                                   │
│  │          ├── slm_timing_20240115.csv                                       │
│  │          └── system_resource_20240115.csv                                  │
│  │                                                                             │
│  └─────────────────────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 📁 Directory Structure

### service_logs/
Service lifecycle logs managed by `service_manager.py`:

```
service_logs/
├── arduino/              # Arduino gRPC service logs
│   └── arduino_service_{timestamp}.log
├── camera/               # Camera tracking service logs
│   └── camera_service_{timestamp}.log
├── slm_generator/        # SLM hologram generator logs
│   └── slm_generator_{timestamp}.log
├── slm_driver/           # SLM hardware driver logs
│   └── slm_driver_{timestamp}.log
└── dashboard/            # GUI dashboard logs
    └── dashboard_{timestamp}.log
```

Each log file contains:
- Launch command with full arguments
- Service configuration parameters
- Real-time stdout/stderr output
- Performance metrics
- Error messages and stack traces
- Exit status and termination details

### AutoLogs/
Application data and experiment results:

```
AutoLogs/
├── tracking_results/     # Particle tracking data
│   ├── {experiment}_*.hdf5       # Compressed tracking data
│   ├── {experiment}_*_metadata.json
│   └── {experiment}_*.csv        # Export format
├── images/               # Saved camera frames
│   └── frame_{timestamp}_{index}.tiff
└── performance/          # System metrics
    ├── camera_metrics_*.csv
    ├── slm_timing_*.csv
    └── system_resource_*.csv
```

## 📝 Log Formats

### Service Log Entry Format

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          SERVICE LOG ENTRY FORMAT                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Header Section (Written on service start):                                    │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ Command: python arduino_rpc/grpc_server_streaming.py --port 50051 ...    │ │
│  │ Log file: logs/service_logs/arduino/arduino_service_20240115_143022.log  │ │
│  │ ========================================================================== │ │
│  │                                                                           │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  Runtime Entries (stdout/stderr capture):                                      │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ [2024-01-15 14:30:22.123] Starting Arduino gRPC server...                │ │
│  │ [2024-01-15 14:30:22.456] Serial port COM3 opened at 2000000 baud        │ │
│  │ [2024-01-15 14:30:22.789] gRPC server listening on 0.0.0.0:50051         │ │
│  │ [2024-01-15 14:30:25.100] Client connected from 127.0.0.1:52341          │ │
│  │ [2024-01-15 14:30:25.234] Command received: SET_DAC ch=0 val=2048        │ │
│  │ [2024-01-15 14:30:25.250] Response sent: OK (CRC: 0xA5)                  │ │
│  │ [2024-01-15 14:30:30.500] ERROR: CRC mismatch (expected 0xB2, got 0xB3)  │ │
│  │ [2024-01-15 14:30:30.501] Retransmission attempt 1/3                     │ │
│  │ [2024-01-15 14:30:35.123] Performance: avg_latency=0.45ms, fps=100.2    │ │
│  │ ...                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  Exit Section (Written on service stop):                                       │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ [2024-01-15 16:21:45.789] Shutdown signal received                       │ │
│  │ [2024-01-15 16:21:45.850] Closing serial port...                         │ │
│  │ [2024-01-15 16:21:45.900] gRPC server stopped                            │ │
│  │ [2024-01-15 16:21:45.950] Process exited with code 0                     │ │
│  │ [2024-01-15 16:21:45.950] Runtime: 1h 51m 23s                            │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tracking Data Format (HDF5)

```python
# HDF5 structure for tracking results
experiment_001_20240115_143500.hdf5
├── /metadata
│   ├── experiment_name: "optical_trap_dynamics"
│   ├── start_time: "2024-01-15T14:35:00Z"
│   ├── duration_seconds: 3600
│   ├── camera_fps: 100.0
│   ├── roi_config: {...}
│   └── tracking_params: {...}
│
├── /frames/                      # Frame-by-frame data
│   ├── frame_0000/
│   │   ├── timestamp: 1705326900.123
│   │   ├── frame_number: 0
│   │   ├── particles: [(x, y, mass, size, ecc), ...]
│   │   └── metrics: {processing_time_ms: 8.5}
│   ├── frame_0001/
│   │   └── ...
│   └── ...
│
├── /trajectories/                # Linked trajectories
│   ├── particle_0/
│   │   ├── positions: [(t0, x0, y0), (t1, x1, y1), ...]
│   │   ├── velocities: [(vx0, vy0), (vx1, vy1), ...]
│   │   └── properties: {mass_avg, size_avg, lifetime}
│   ├── particle_1/
│   │   └── ...
│   └── ...
│
└── /performance/                 # System metrics
    ├── tracking_times: [t0, t1, t2, ...]
    ├── gpu_utilization: [u0, u1, u2, ...]
    └── memory_usage: [m0, m1, m2, ...]
```

### Performance Metrics (CSV)

```csv
# camera_metrics_20240115.csv
timestamp,frame_number,capture_ms,tracking_ms,total_ms,particles_found,gpu_util_%,memory_mb
2024-01-15 14:35:00.100,0,1.2,7.3,8.5,15,45.2,2340
2024-01-15 14:35:00.110,1,1.1,7.5,8.6,16,46.1,2345
2024-01-15 14:35:00.120,2,1.3,7.2,8.5,15,44.8,2342
...

# slm_timing_20240115.csv
timestamp,command_id,generation_ms,transfer_ms,update_ms,tweezers_count,iterations
2024-01-15 14:35:01.200,cmd_001,1.8,0.3,0.5,8,45
2024-01-15 14:35:02.300,cmd_002,2.1,0.3,0.5,12,50
2024-01-15 14:35:03.400,cmd_003,1.9,0.3,0.5,10,48
...

# system_resource_20240115.csv
timestamp,cpu_percent,mem_percent,disk_io_mb_s,net_io_mb_s,gpu_mem_mb,gpu_temp_c
2024-01-15 14:35:00.000,35.2,42.1,15.3,2.1,4800,62
2024-01-15 14:35:01.000,36.8,42.3,16.1,2.3,4850,63
2024-01-15 14:35:02.000,34.5,42.2,14.8,2.0,4820,62
...
```

## 🔍 Service-Specific Logging

### Arduino Service

Typical log entries:
```
[2024-01-15 14:30:22.123] Starting Arduino gRPC server on port 50051
[2024-01-15 14:30:22.456] Serial port COM3 opened at 2000000 baud
[2024-01-15 14:30:22.500] Testing communication... CRC validation OK
[2024-01-15 14:30:22.550] Device ready: firmware_version=2.1.0
[2024-01-15 14:30:25.234] CMD: SET_DAC(ch=0, val=2048) → Response: OK (0.45ms)
[2024-01-15 14:30:25.256] CMD: READ_ADC(ch=3) → Response: 1850 (0.38ms)
[2024-01-15 14:30:30.500] ERROR: CRC mismatch (expected 0xB2, got 0xB3)
[2024-01-15 14:30:30.501] Retransmission successful on attempt 2
```

Key metrics logged:
- Command latency (microseconds)
- CRC validation results
- Retransmission statistics
- Serial buffer state

### Camera Service

Typical log entries:
```
[2024-01-15 14:30:25.100] Camera service starting on port 50052
[2024-01-15 14:30:25.250] Camera initialized: Basler acA1920-40gm (1920x1200)
[2024-01-15 14:30:25.300] ROI configured: 512x512 tiles @ 32 workers
[2024-01-15 14:30:25.350] TrackPy initialized: diameter=11px, minmass=100
[2024-01-15 14:30:26.100] Tracking started: frame 0
[2024-01-15 14:30:26.110] Frame 0: 15 particles tracked in 8.5ms
[2024-01-15 14:30:26.120] Frame 1: 16 particles tracked in 8.3ms
[2024-01-15 14:30:30.500] Performance stats (5s): avg=8.4ms, fps=100.2, particles=15.3
```

Key metrics logged:
- Frame capture timing
- Particle detection results
- Processing pipeline stages
- GPU memory usage

### SLM Services

Generator service:
```
[2024-01-15 14:30:28.100] SLM generator starting on port 50053
[2024-01-15 14:30:28.200] GPU initialized: NVIDIA RTX 4070 (12GB)
[2024-01-15 14:30:28.300] cuFFT plan created for 512x512
[2024-01-15 14:30:29.100] Hologram generated: 8 tweezers, 45 iterations, 1.8ms
[2024-01-15 14:30:29.102] Convergence error: 0.0085 (threshold: 0.01)
[2024-01-15 14:30:30.100] Performance stats: avg=2.1ms, gpu_util=52%
```

Driver service:
```
[2024-01-15 14:30:30.100] SLM driver starting on port 50054
[2024-01-15 14:30:30.200] SDK initialized: Meadowlark 512x512
[2024-01-15 14:30:30.300] Display configured: 60Hz refresh, 8-bit depth
[2024-01-15 14:30:31.100] Hologram received: command_id=cmd_001
[2024-01-15 14:30:31.105] Written to hardware in 0.5ms
```

## 🛠️ Analysis Tools

### Log Parsing Scripts

```python
# parse_service_logs.py
import re
from pathlib import Path
from datetime import datetime

def parse_log_file(log_path: Path):
    """Extract structured data from service log."""
    entries = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Parse timestamp
            match = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\] (.+)', line)
            if match:
                timestamp_str, message = match.groups()
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                
                entries.append({
                    'timestamp': timestamp,
                    'message': message,
                    'severity': classify_severity(message)
                })
    
    return entries

def extract_performance_metrics(log_path: Path):
    """Extract performance metrics from service log."""
    metrics = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Look for performance stats
            if 'Performance stats' in line or 'avg_latency' in line:
                metrics.append(parse_metrics_line(line))
    
    return metrics

# Usage
logs_dir = Path('logs/service_logs/camera/')
for log_file in logs_dir.glob('*.log'):
    entries = parse_log_file(log_file)
    metrics = extract_performance_metrics(log_file)
    
    print(f"Log: {log_file.name}")
    print(f"  Total entries: {len(entries)}")
    print(f"  Errors: {sum(1 for e in entries if e['severity'] == 'ERROR')}")
    print(f"  Average latency: {compute_avg_latency(metrics):.2f}ms")
```

### Error Analysis

```bash
# Find all errors across service logs
grep -r "ERROR:" logs/service_logs/ > error_summary.txt

# Count error frequency by type
awk -F': ' '/ERROR:/ {print $2}' logs/service_logs/*/*.log | sort | uniq -c | sort -rn

# Extract timing information
grep "avg_latency" logs/service_logs/arduino/*.log | awk '{print $NF}' > latency_data.txt

# Find service crashes
grep -B5 "exited with code" logs/service_logs/*/*.log | grep -v "code 0"
```

## 🐛 Troubleshooting Guide

### Common Issues

**Service Not Logging**
```bash
# Check if log directory exists
ls -la logs/service_logs/

# Verify permissions
chmod -R u+w logs/

# Check service configuration
cat GUI/service_config.yaml | grep log_subdir
```

**Log Files Growing Too Large**
```bash
# Find large log files
find logs/ -type f -size +100M

# Compress old logs
gzip logs/service_logs/*/*.log.old

# Automatic cleanup (keep last 7 days)
find logs/service_logs/ -name "*.log" -mtime +7 -delete
```

**Missing Tracking Data**
```python
# Verify HDF5 file integrity
import h5py

with h5py.File('logs/AutoLogs/tracking_results/experiment_001.hdf5', 'r') as f:
    print("Groups:", list(f.keys()))
    print("Frames:", len(f['/frames']))
    print("Trajectories:", len(f['/trajectories']))
```

**Performance Degradation**
```bash
# Check log write performance
time echo "test" >> logs/service_logs/test.log

# Monitor real-time logging
tail -f logs/service_logs/camera/camera_service_*.log

# Analyze system resources
grep "cpu_percent" logs/AutoLogs/performance/system_resource_*.csv | tail -100
```

### Monitoring Commands

```bash
# Watch all services in real-time
watch -n 1 'for f in logs/service_logs/*/; do echo "=== ${f} ==="; tail -3 ${f}*.log 2>/dev/null; done'

# Count active log files
ls -1 logs/service_logs/*/*.log | wc -l

# Find most recent logs
find logs/service_logs/ -name "*.log" -printf "%T@ %p\n" | sort -n | tail -5

# Monitor disk usage
du -sh logs/*
```

## 📊 Log Retention Policy

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          LOG RETENTION STRATEGY                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Service Logs (logs/service_logs/):                                            │
│  ├─ Active logs: Keep all                                                      │
│  ├─ Recent logs (< 7 days): Keep uncompressed                                  │
│  ├─ Old logs (7-30 days): Compress with gzip                                   │
│  └─ Ancient logs (> 30 days): Archive to backup or delete                      │
│                                                                                 │
│  Tracking Data (logs/AutoLogs/tracking_results/):                              │
│  ├─ Current experiments: Keep all                                              │
│  ├─ Recent data (< 30 days): Keep HDF5 + metadata                              │
│  ├─ Archived data (> 30 days): Keep HDF5 only (delete CSV exports)             │
│  └─ Old archives (> 1 year): Move to cold storage                              │
│                                                                                 │
│  Images (logs/AutoLogs/images/):                                               │
│  ├─ Important frames: Keep indefinitely                                        │
│  ├─ Recent images (< 7 days): Keep all                                         │
│  └─ Old images (> 7 days): Delete (data preserved in HDF5)                     │
│                                                                                 │
│  Performance Metrics (logs/AutoLogs/performance/):                             │
│  ├─ Daily CSV files: Keep for 90 days                                          │
│  └─ Aggregated stats: Keep indefinitely                                        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🔧 Configuration

Logging configuration in `GUI/service_config.yaml`:

```yaml
global:
  log_base_dir: "../../logs/service_logs"
  stop_services_on_exit: true

services:
  - key: "arduino_service"
    log_subdir: "arduino"        # Subdirectory in log_base_dir
    # Logs will be written to:
    # logs/service_logs/arduino/arduino_service_{timestamp}.log
    
  - key: "camera_service"
    log_subdir: "camera"
    # Application data saved to:
    # logs/AutoLogs/ (configured in Camera/ImageServer_with_track.py)
```

The logs directory provides comprehensive monitoring, diagnostics, and data archival for all Tweezer system components with structured output and automated retention policies.

## 📄 Related Documentation

- [Main README](../README.md) - System overview
- [GUI README](../GUI/README.md) - Service manager details
- [Camera README](../Camera/README.md) - Tracking data formats
- [service_config.yaml](../GUI/service_config.yaml) - Logging configuration