# SigNoz on Pi 5 — Local Observability for robot_comic

SigNoz is a self-hosted OpenTelemetry observability stack (traces + metrics + logs) that
runs as a Docker Compose bundle.  This guide sets it up on the Pi 5 so you can inspect
robot_comic traces without a cloud service.

The stack is **on-demand** — you bring it up when investigating, shut it down when done.
It does not auto-start at boot.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Pi 5 8 GB | 4 GB will be tight; 8 GB is comfortable |
| Docker Engine ≥ 24 | See install steps below |
| Docker Compose v2 plugin | Included in modern Docker installs |
| USB 3 SSD (recommended) | ClickHouse is write-heavy; SD card will degrade |
| Outbound internet (first run only) | Images + histogram-quantile binary are pulled on first `docker compose up` |

### Install Docker on the Pi (if not already installed)

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in so group membership takes effect
docker --version   # should print Docker version
```

---

## Install

Clone the repo (or copy the `deploy/signoz/` directory) onto the Pi, then:

```bash
bash deploy/signoz/install.sh
```

The script:
1. Verifies Docker and Compose are present.
2. Sets `vm.max_map_count=262144` transiently (required by ClickHouse).
3. Runs `docker compose up -d` in `deploy/signoz/`.
4. Waits for the health check and prints the UI URL.

First run pulls ~2 GB of images and takes 3–5 minutes on a good connection.
Subsequent starts take under 60 seconds.

---

## Configure robot_comic to send traces

Add these two lines to the `.env` file on the robot (CM4 or Pi 5 itself):

```dotenv
ROBOT_INSTRUMENTATION=remote
OTEL_EXPORTER_OTLP_ENDPOINT=http://<pi5-hostname>:4317
```

Replace `<pi5-hostname>` with the Pi 5's LAN hostname (e.g. `reachy-pi5.local`) or IP
address.  If the robot runs on the same Pi as SigNoz, use `http://localhost:4317`.

Then restart robot_comic:

```bash
sudo systemctl restart reachy-app-autostart
# or, in sim mode:
python -m robot_comic.main --sim
```

The app picks up the env vars on startup and switches the OTLP exporter to gRPC port 4317.

---

## Using the SigNoz UI

Open `http://<pi5-hostname>:3301` in a browser.

### First launch

1. **Create an account** — SigNoz prompts for an email and password on first open.
   This is local-only; no data leaves the Pi.
2. **Go to Traces** (left sidebar → Services → robot_comic).
3. Trigger a conversation on the robot.  Within a few seconds you should see spans
   appearing in the trace list.

### Seeing traces flow in

- **Services view** shows `robot_comic` service after the first span arrives.
- **Traces** tab lets you filter by time range, duration, and status.
- Click a trace row to open the **flame graph** and see individual span timings
  (e.g. `llm.generate`, `tts.synthesize`, `tool.dispatch`).
- **Metrics** tab shows histograms like `robot.turn.duration` and
  `gen_ai.client.operation.duration`.

### Useful filters

| Goal | Filter |
|---|---|
| Slow turns | `robot.turn.duration > 3s` |
| LLM calls only | span name contains `llm` |
| Errors | Status = ERROR |
| Specific session | attribute `session.id = <id>` |

---

## Resource considerations on Pi 5 8 GB

| Component | Typical idle RAM | Peak RAM |
|---|---|---|
| ClickHouse | ~400 MB | ~2.5 GB (capped at 35% = ~2.8 GB) |
| ZooKeeper | ~80 MB | ~150 MB |
| OTel Collector | ~60 MB | ~150 MB |
| SigNoz UI/API | ~150 MB | ~300 MB |
| **Total stack** | **~700 MB** | **~3.2 GB** |

With the Pi-tuned `clickhouse-config.xml` (`max_server_memory_usage_to_ram_ratio=0.35`,
512 MB uncompressed cache, 256 MB mark cache), ClickHouse is capped at roughly 2.8 GB.
This leaves ~4.8 GB for robot_comic and the OS when both run simultaneously.

**USB SSD recommendation**: ClickHouse performs many small random writes.  Running on the
stock SD card will accelerate wear and reduce throughput.  Mount a USB 3 SSD and point the
Docker volumes at it:

```bash
# Create bind-mount point on SSD
sudo mkdir -p /mnt/ssd/signoz/clickhouse /mnt/ssd/signoz/sqlite /mnt/ssd/signoz/zookeeper

# Edit deploy/signoz/docker-compose.yml — replace named volumes with bind mounts:
#   clickhouse:
#     driver: local
#     driver_opts: { type: none, o: bind, device: /mnt/ssd/signoz/clickhouse }
```

**Keeping the Pi light at boot**: SigNoz containers restart unless stopped (not `always`).
If you stop the stack with `docker compose down`, it stays down across reboots.
Docker itself also does not need to be enabled at boot:

```bash
sudo systemctl disable docker.service docker.socket
# Re-enable if you want Docker to start automatically:
# sudo systemctl enable docker.service docker.socket
```

---

## Stopping and removing the stack

**Stop (keep data)**:
```bash
cd deploy/signoz
docker compose down
```

**Stop and wipe all trace/metric data**:
```bash
bash deploy/signoz/uninstall.sh
# equivalent to: docker compose down -v
```

---

## Shell aliases (optional convenience)

Add to `~/.bashrc` on the Pi:

```bash
alias signoz-up='bash ~/robot_comic/deploy/signoz/install.sh'
alias signoz-down='bash ~/robot_comic/deploy/signoz/uninstall.sh'
```

---

## Troubleshooting

### Check container status and logs

```bash
cd deploy/signoz
docker compose ps                    # all containers and health
docker compose logs -f               # follow all logs
docker compose logs signoz           # SigNoz API logs only
docker compose logs clickhouse       # ClickHouse logs
docker compose logs otel-collector   # OTel collector logs
```

### Ports

| Port | Service | Notes |
|---|---|---|
| 3301 | SigNoz UI + API | Mapped from container's 8080 to avoid clash with robot app |
| 4317 | OTLP gRPC | Robot sends traces here |
| 4318 | OTLP HTTP | Alternative ingestion endpoint |
| 8123 | ClickHouse HTTP | Internal; not exposed outside Docker network |
| 9000 | ClickHouse native | Internal |

If port 3301 is taken by another service, change the `ports` line in
`deploy/signoz/docker-compose.yml`:
```yaml
ports:
  - "3302:8080"   # use a different host port
```

### Firewall

If the CM4 (robot body) is on a different machine and cannot reach the Pi 5:

```bash
# On Pi 5 — allow OTLP gRPC from local network
sudo ufw allow 4317/tcp comment 'OTLP gRPC for SigNoz'
sudo ufw allow 3301/tcp comment 'SigNoz UI'
```

### ClickHouse fails to start

1. Check `vm.max_map_count`:
   ```bash
   cat /proc/sys/vm/max_map_count   # should be ≥ 262144
   sudo sysctl -w vm.max_map_count=262144
   ```
2. To make it persistent across reboots:
   ```bash
   echo 'vm.max_map_count=262144' | sudo tee /etc/sysctl.d/99-clickhouse.conf
   sudo sysctl --system
   ```

### init-clickhouse fails (histogram-quantile download)

The bootstrap container needs internet access on first run to fetch a small binary from
GitHub.  If your Pi has no outbound internet access, pre-download the binary:

```bash
# On a machine with internet:
wget -O histogramQuantile \
  "https://github.com/SigNoz/signoz/releases/download/histogram-quantile%2Fv0.0.1/histogram-quantile_linux_arm64.tar.gz"
tar -xzf histogram-quantile_linux_arm64.tar.gz

# Copy to the Pi and put it in the Docker volume:
docker run --rm -v signoz-clickhouse-user-scripts:/scripts \
  alpine cp /host/histogramQuantile /scripts/histogramQuantile
```

### No traces appear in SigNoz

1. Confirm `ROBOT_INSTRUMENTATION=remote` is set and the app restarted.
2. Confirm `OTEL_EXPORTER_OTLP_ENDPOINT` points to the Pi 5's port 4317.
3. Check the OTel collector received the spans:
   ```bash
   docker compose logs otel-collector | grep "traces"
   ```
4. Check the app logs for OTel errors:
   ```bash
   journalctl -u reachy-app-autostart | grep -i "otel\|otlp"
   ```
