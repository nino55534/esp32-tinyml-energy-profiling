# ESP32 TinyML Energy Profiling

Experimental analysis of the energy profile of a resource-constrained IoT device (ESP32-S3) that runs on-device machine learning inference and reports results over three different transport protocols.

> Bachelor's thesis (Završni rad br. 2290) — Faculty of Electrical Engineering and Computing (FER), University of Zagreb, 2026.
> *"Eksperimentalna analiza energetskog profila rada uređaja ograničenih resursa u sustavu Interneta stvari"*
> Author: Nino Ljubas · Mentor: prof. dr. sc. Gordan Ježić

## Overview

IoT edge devices typically run a cyclic duty cycle: sleep, wake, read a sensor, optionally run local inference, transmit the result, sleep again. Moving inference on-device (TinyML) can cut the amount of data sent over the network, but it adds compute load of its own. This project measures where the energy actually goes in that cycle.

An ESP32-S3 reads temperature/humidity from a DHT11 sensor, runs one of two TensorFlow Lite Micro models (a quantized CNN or a float RNN) to predict temperature, and sends the result to a host machine over **MQTT**, **UDP**, or **TCP**. Every phase of the cycle is timed and power-profiled with a Nordic **Power Profiler Kit II (PPK2)**, synced to the firmware via a GPIO trigger pin.

## Key results

| Phase | Variant | Duration | Avg. current | Energy |
|---|---|---|---|---|
| Sleep | all | 1000 ms | 54 mA | 178.20 mJ |
| Sensor read (DHT11) | UDP / MQTT | 24 ms | 55 mA | 4.36 mJ |
| Sensor read (DHT11) | TCP | 24 ms | 70 mA | 5.54 mJ |
| CNN inference | UDP / MQTT | 4 ms | 64 mA | 0.86 mJ |
| CNN inference | TCP | 4 ms | 72 mA | 1.00 mJ |
| RNN inference | UDP / MQTT | 9 ms | 62 mA | 1.94 mJ |
| RNN inference | TCP | 9 ms | 65 mA | 1.97 mJ |
| Send result | UDP | 2.3 ms | 105 mA | 0.79 mJ |
| Send result | MQTT | 2.4 ms | 90 mA | 0.71 mJ |
| Send result | TCP | 31.0 ms | 122 mA | 12.48 mJ |

Takeaways:

- **Transport protocol matters far more than model choice.** TCP's per-cycle connect/teardown costs **~12.48 mJ** to send a result, vs **0.79 mJ** (UDP) or **0.71 mJ** (MQTT) — roughly 16–17× more, because MQTT reuses an already-open connection and UDP has no handshake at all, while this implementation opens and closes a new TCP connection every cycle.
- **CNN beats RNN on both latency and energy** (~4 ms / ~0.9 mJ vs ~9 ms / ~1.95 mJ): the CNN processes all 24 input samples in one pass, while the RNN runs sequentially over them.
- **Sleep dominates total energy** (178.2 mJ) purely due to its 1-second duration, not its current draw — the active part of the cycle (sense + infer + send) is an order of magnitude shorter.
- Full active-cycle energy per configuration: **CNN+MQTT ≈ 5.93 mJ**, **CNN+UDP ≈ 6.01 mJ**, **RNN+MQTT ≈ 7.01 mJ**, **RNN+UDP ≈ 7.09 mJ**, vs **CNN+TCP ≈ 19.02 mJ**, **RNN+TCP ≈ 19.99 mJ**.

Full methodology, formulas, PPK2 captures, and discussion of measurement limitations: [`docs/Zavrsni_rad_NinoLjubas.pdf`](docs/Zavrsni_rad_NinoLjubas.pdf) (Croatian).

## Repository structure

```
firmware/
├── esp-idf-project/   Buildable ESP-IDF app (main.cpp currently set to the CNN+MQTT variant)
└── variants/           Six standalone main.cpp files, one per model×protocol combination —
                         each was swapped into esp-idf-project/main/main.cpp for its measurement run
models/                 Trained TFLite Micro models (CNN, int8 quantized; RNN, float)
server/                 Python host-side receivers (serverUDP.py, serverTCP.py; MQTT uses a broker)
results/
├── mqtt/ tcp/ udp/     PPK2 capture screenshots per phase and protocol
├── power_measurements/ Raw PPK2 capture files (open with the nRF Connect Power Profiler app)
├── ukupna_potrosnja.png   Total active-cycle energy per configuration
└── esp32_work_cycle.svg   Duty cycle diagram
docs/                   Thesis PDF, defense presentation
```

## Building the firmware

Requires [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/get-started/index.html) (developed against v5.5, target `esp32s3`).

```
cd firmware/esp-idf-project
idf.py set-target esp32s3
idf.py build flash monitor
```

To reproduce a specific measurement, copy the matching file from `firmware/variants/` over `esp-idf-project/main/main.cpp` before building.

Before building, set your own network credentials — each variant defines these near the top of the file (values are placeholders, not real credentials):

```c
#define WIFI_SSID     "YOUR_WIFI_SSID"
#define WIFI_PASS     "YOUR_WIFI_PASSWORD"
// MQTT variants only:
mqtt_cfg.broker.address.uri = "mqtt://YOUR_BROKER_IP";
```

For the UDP/TCP variants, point the host address in the firmware at the machine running `server/serverUDP.py` / `server/serverTCP.py`. For MQTT, point both the firmware and a broker of your choice at the same address.

## Hardware

- ESP32-S3
- DHT11 temperature/humidity sensor
- Nordic Power Profiler Kit II (PPK2), used only for measurement, not required to run the firmware

## License

Code and measurement data are MIT licensed (see [LICENSE](LICENSE)). The thesis document and presentation in `docs/` remain the author's academic work, included here for reference.
