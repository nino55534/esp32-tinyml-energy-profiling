#include <stdio.h>
#include <string.h>
#include <cstring>
#include <errno.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "nvs_flash.h"
#include "rom/ets_sys.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"
#include "esp_sleep.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "rnn_model_fixed.h"

#define TRIGGER_PIN GPIO_NUM_5

#define TAG           "ZAV_RAD"
#define WIFI_SSID     "YOUR_WIFI_SSID"   
#define WIFI_PASS     "YOUR_WIFI_PASSWORD"
#define SERVER_IP     "172.20.10.3"
#define SERVER_PORT   5005
#define DHT_PIN       GPIO_NUM_4
#define CYCLE_MS      1000

static float rnn_buf[24] = {0};
static int   rnn_buf_count = 0;

#define RNN_MEAN 13.387054f
#define RNN_STD   8.543148f

static EventGroupHandle_t wifi_events;
#define WIFI_CONNECTED_BIT BIT0

static int64_t idle_start_time = 0;

static void wifi_event_handler(void* arg, esp_event_base_t base, int32_t id, void* data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        xEventGroupSetBits(wifi_events, WIFI_CONNECTED_BIT);
        ESP_LOGI(TAG, "WiFi spojen");
    }
}

static void wifi_init() {
    wifi_events = xEventGroupCreate();
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);

    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL);
    esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL);

    wifi_config_t wcfg = {};
    strncpy((char*)wcfg.sta.ssid, WIFI_SSID, sizeof(wcfg.sta.ssid));
    strncpy((char*)wcfg.sta.password, WIFI_PASS, sizeof(wcfg.sta.password));

    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wcfg);
    esp_wifi_start();
    esp_wifi_connect();

    xEventGroupWaitBits(wifi_events, WIFI_CONNECTED_BIT, false, true, pdMS_TO_TICKS(10000));
}

#define TCP_TIMEOUT_MS 5000

static void tcp_send(const char* msg) {
    int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock < 0) {
        ESP_LOGE(TAG, "TCP socket creation failed");
        return;
    }

    struct timeval tv = {
        .tv_sec = TCP_TIMEOUT_MS / 1000,
        .tv_usec = (TCP_TIMEOUT_MS % 1000) * 1000
    };
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    struct sockaddr_in dest = {};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(SERVER_PORT);
    inet_pton(AF_INET, SERVER_IP, &dest.sin_addr);

    if (connect(sock, (struct sockaddr*)&dest, sizeof(dest)) != 0) {
        ESP_LOGE(TAG, "TCP connect failed: %s", strerror(errno));
        close(sock);
        return;
    }

    int len = strlen(msg);
    int total = 0;
    while (total < len) {
        int sent = send(sock, msg + total, len - total, 0);
        if (sent < 0) {
            ESP_LOGE(TAG, "TCP send error: %s", strerror(errno));
            break;
        }
        total += sent;
    }

    close(sock);
    ESP_LOGI(TAG, "TCP sent: %s", msg);
}

static int wait_for_state(gpio_num_t pin, int target, uint32_t timeout_us) {
    uint32_t start = esp_timer_get_time();
    while (gpio_get_level(pin) != target) {
        if ((esp_timer_get_time() - start) > timeout_us) return -1;
        ets_delay_us(1);
    }
    return 0;
}

static esp_err_t dht11_read(gpio_num_t pin, float *temp, float *hum) {
    uint8_t data[5] = {0};
    gpio_reset_pin(pin);
    gpio_set_pull_mode(pin, GPIO_PULLUP_ONLY);
    gpio_set_direction(pin, GPIO_MODE_OUTPUT_OD);

    gpio_set_level(pin, 0);
    ets_delay_us(20000);
    gpio_set_level(pin, 1);
    ets_delay_us(30);
    gpio_set_direction(pin, GPIO_MODE_INPUT);

    if (wait_for_state(pin, 0, 100) < 0) return ESP_FAIL;
    if (wait_for_state(pin, 1, 100) < 0) return ESP_FAIL;
    if (wait_for_state(pin, 0, 100) < 0) return ESP_FAIL;

    for (int i = 0; i < 40; i++) {
        if (wait_for_state(pin, 1, 100) < 0) return ESP_FAIL;
        uint32_t t0 = esp_timer_get_time();
        if (wait_for_state(pin, 0, 100) < 0) return ESP_FAIL;
        data[i / 8] <<= 1;
        if ((esp_timer_get_time() - t0) > 40) data[i / 8] |= 1;
    }

    if ((uint8_t)(data[0]+data[1]+data[2]+data[3]) != data[4]) {
        ESP_LOGW(TAG, "DHT11 checksum greška");
        return ESP_FAIL;
    }
    *hum  = (float)data[0];
    *temp = (float)data[2];
    return ESP_OK;
}

#define ARENA_SIZE (80 * 1024)
static uint8_t rnn_arena[ARENA_SIZE];
static tflite::MicroMutableOpResolver<13> rnn_resolver;
static bool models_registered = false;

void register_ops() {
    if (models_registered) return;
    rnn_resolver.AddFullyConnected();
    rnn_resolver.AddRelu();
    rnn_resolver.AddAdd();
    rnn_resolver.AddReshape();
    rnn_resolver.AddUnpack();
    rnn_resolver.AddTranspose();
    rnn_resolver.AddStridedSlice();
    rnn_resolver.AddShape();
    rnn_resolver.AddQuantize();
    rnn_resolver.AddDequantize();
    rnn_resolver.AddPack();
    rnn_resolver.AddMul();
    rnn_resolver.AddFill();
    models_registered = true;
}

float pokreni_rnn() {
    const tflite::Model* model = tflite::GetModel(RNN_temp_in_C_new_tflite);
    tflite::MicroInterpreter interp(model, rnn_resolver, rnn_arena, ARENA_SIZE);

    if (interp.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "RNN AllocateTensors FAIL");
        return -1.0f;
    }

    TfLiteTensor* input = interp.input(0);
    if (input->type == kTfLiteInt8) {
        float scale = input->params.scale;
        int zero_point = input->params.zero_point;
        for (int i = 0; i < 24; i++) {
            float norm = (rnn_buf[i] - RNN_MEAN) / RNN_STD;
            input->data.int8[i] = (int8_t)(norm / scale + zero_point);
        }
    } else {
        memcpy(input->data.f, rnn_buf, 24 * sizeof(float));
    }

    int64_t t0 = esp_timer_get_time();
    TfLiteStatus s = interp.Invoke();
    int64_t t1 = esp_timer_get_time();

    if (s != kTfLiteOk) {
        ESP_LOGE(TAG, "RNN Invoke FAIL");
        return -1.0f;
    }

    TfLiteTensor* output = interp.output(0);
    float out_norm;
    if (output->type == kTfLiteInt8) {
        float scale = output->params.scale;
        int zero_point = output->params.zero_point;
        out_norm = (output->data.int8[0] - zero_point) * scale;
    } else {
        out_norm = output->data.f[0];
    }
    float predicted_C = out_norm * RNN_STD + RNN_MEAN;

    ESP_LOGI(TAG, "[RNN] %lld us → %.5f (norm) → %.2f °C", (long long)(t1 - t0), out_norm, predicted_C);
    return predicted_C;
}

static void ciklus() {
    int64_t t_wake = esp_timer_get_time();
    float temp = 0, hum = 0;

    gpio_set_level(TRIGGER_PIN, 1);
    int64_t t0 = esp_timer_get_time();
    esp_err_t ok = dht11_read(DHT_PIN, &temp, &hum);
    int64_t t1 = esp_timer_get_time();
    gpio_set_level(TRIGGER_PIN, 0);
    ESP_LOGI(TAG, "[DHT] %lld us", (long long)(t1 - t0));

    if (ok != ESP_OK) {
        temp = rnn_buf[23];
        hum = 50.0f;
    }

    for (int i = 0; i < 23; i++) rnn_buf[i] = rnn_buf[i+1];
    rnn_buf[23] = temp;
    if (rnn_buf_count < 24) rnn_buf_count++;

    if (rnn_buf_count < 24) {
        ESP_LOGW(TAG, "Buffer nije pun jos");
        idle_start_time = esp_timer_get_time();
        return;
    }

    gpio_set_level(TRIGGER_PIN, 1);
    int64_t t2 = esp_timer_get_time();
    float rnn_out = pokreni_rnn();
    int64_t t3 = esp_timer_get_time();
    gpio_set_level(TRIGGER_PIN, 0);
    ESP_LOGI(TAG, "[RNN] %lld us", (long long)(t3 - t2));  

    ets_delay_us(100);                                

    gpio_set_level(TRIGGER_PIN, 1);
    int64_t t_tcp_start = esp_timer_get_time();

    char payload[80];
    snprintf(payload, sizeof(payload), "{\"temp\":%.2f,\"rnn\":%.2f}", temp, rnn_out);
    tcp_send(payload);

    int64_t t_tcp_end = esp_timer_get_time();
    gpio_set_level(TRIGGER_PIN, 0);
    ESP_LOGI(TAG, "[TCP] %lld us", (long long)(t_tcp_end - t_tcp_start));

    int64_t t5 = esp_timer_get_time();
    ESP_LOGI(TAG, "--- KRAJ: %lld us ---", (long long)(t5 - t_wake));

    idle_start_time = esp_timer_get_time();
}

static void init_trigger_pin() {
    gpio_reset_pin(TRIGGER_PIN);
    gpio_set_direction(TRIGGER_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level(TRIGGER_PIN, 0);
    ESP_LOGI(TAG, "Trigger pin GPIO%d inicijaliziran", TRIGGER_PIN);
}

extern "C" void app_main() {
    nvs_flash_init();
    wifi_init();
    esp_wifi_set_ps(WIFI_PS_MIN_MODEM);
    register_ops();
    init_trigger_pin();

    ESP_LOGI(TAG, "Pokrenuto mjerenje ESP32-S3 ");
    ESP_LOGI(TAG, "RNN model: %d bytes", RNN_temp_in_C_new_tflite_len);

    vTaskDelay(pdMS_TO_TICKS(2000));

    while (true) {
        if (idle_start_time != 0) {
            int64_t idle_end = esp_timer_get_time();
            ESP_LOGI(TAG, "[MIROVANJE] %lld us", (long long)(idle_end - idle_start_time));
            idle_start_time = 0;                            
        }
        ciklus();
        vTaskDelay(pdMS_TO_TICKS(CYCLE_MS));
    }
}