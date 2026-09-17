#include <stdio.h>
#include <string.h>
#include <cstring>

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
#include "esp_sleep.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "cnn_model.h"
#include "mqtt_client.h"

#define CNN_MEAN 13.387054f
#define CNN_STD   8.542878f

#define TRIGGER_PIN GPIO_NUM_5

#define TAG           "ZAV_RAD"
#define WIFI_SSID     "YOUR_WIFI_SSID"
#define WIFI_PASS     "YOUR_WIFI_PASSWORD"
#define DHT_PIN       GPIO_NUM_4
#define CYCLE_MS      1000

static float temp_buf[24] = {0};
static int   temp_count = 0;

static EventGroupHandle_t wifi_events;
#define WIFI_CONNECTED_BIT BIT0

static esp_mqtt_client_handle_t mqtt_client = nullptr;
static const char *mqtt_topic = "esp32s3/data";

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

static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
    auto *event = (esp_mqtt_event_handle_t)event_data;
    switch ((esp_mqtt_event_id_t)event_id) {
        case MQTT_EVENT_CONNECTED:
            ESP_LOGI(TAG, "MQTT spojen na broker");
            break;
        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGW(TAG, "MQTT odspojen");
            break;
        case MQTT_EVENT_ERROR:
            ESP_LOGE(TAG, "MQTT error");
            break;
        default:
            break;
    }
}

static void mqtt_init(void) {
    esp_mqtt_client_config_t mqtt_cfg = {};
    mqtt_cfg.broker.address.uri = "mqtt://YOUR_BROKER_IP";
    mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
    esp_mqtt_client_register_event(mqtt_client, MQTT_EVENT_ANY, mqtt_event_handler, nullptr);
    esp_mqtt_client_start(mqtt_client);
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
static uint8_t tensor_arena[ARENA_SIZE];
static tflite::MicroMutableOpResolver<10> cnn_resolver;
static bool models_registered = false;

void register_ops() {
    if (models_registered) return;
    cnn_resolver.AddConv2D();
    cnn_resolver.AddMean();
    cnn_resolver.AddFullyConnected();
    cnn_resolver.AddReshape();
    cnn_resolver.AddPad();
    cnn_resolver.AddRelu();
    cnn_resolver.AddQuantize();
    cnn_resolver.AddDequantize();
    cnn_resolver.AddExpandDims();
    cnn_resolver.AddSqueeze();
    models_registered = true;
}

float pokreni_cnn(float* input) {
    const tflite::Model* model = tflite::GetModel(CNN_int8_tflite);
    tflite::MicroInterpreter interp(model, cnn_resolver, tensor_arena, ARENA_SIZE);
    
    if (interp.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "CNN AllocateTensors FAIL");
        return -1.0f;
    }

    TfLiteTensor* input_tensor = interp.input(0);
    
    if (input_tensor->type == kTfLiteInt8) {
        float scale = input_tensor->params.scale;
        int zero_point = input_tensor->params.zero_point;
        for (int i = 0; i < 24; i++) {
            float normalized = (input[i] - CNN_MEAN) / CNN_STD;
            int8_t q = (int8_t)((normalized / scale) + zero_point);
            input_tensor->data.int8[i] = q;
        }
    } else {
        memcpy(input_tensor->data.f, input, 24 * sizeof(float));
    }
    int64_t t0 = esp_timer_get_time();
    TfLiteStatus s = interp.Invoke();
    int64_t t1 = esp_timer_get_time();
    
    if (s != kTfLiteOk) {
        ESP_LOGE(TAG, "CNN Invoke FAIL");
        return -1.0f;
    }

    TfLiteTensor* output_tensor = interp.output(0);
    float out_norm;
    if (output_tensor->type == kTfLiteInt8) {
        float scale = output_tensor->params.scale;
        int zero_point = output_tensor->params.zero_point;
        out_norm = (output_tensor->data.int8[0] - zero_point) * scale;
    } else {
        out_norm = output_tensor->data.f[0];
    }

    float predicted_temp_c = out_norm * CNN_STD + CNN_MEAN;
    ESP_LOGI(TAG, "[CNN] %lld us -> %.5f (norm)  -> %.2f °C", (long long)(t1 - t0), out_norm, predicted_temp_c);
    return predicted_temp_c;
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
        temp = temp_buf[23];
        hum = 50.0f;
    }

    for (int i = 0; i < 23; i++) temp_buf[i] = temp_buf[i+1];
    temp_buf[23] = temp;
    if (temp_count < 24) temp_count++;

    if (temp_count < 24) {
        ESP_LOGW(TAG, "Buffer nije pun jos");
        idle_start_time = esp_timer_get_time();
        return;
    }

    gpio_set_level(TRIGGER_PIN, 1);
    int64_t t2 = esp_timer_get_time();
    float cnn_out = pokreni_cnn(temp_buf);
    int64_t t3 = esp_timer_get_time();
    gpio_set_level(TRIGGER_PIN, 0);
    ESP_LOGI(TAG, "[CNN] %lld us", (long long)(t3 - t2));  

    ets_delay_us(100);                                    

    gpio_set_level(TRIGGER_PIN, 1);
    int64_t t_mqtt_start = esp_timer_get_time();

    char payload[80];
    snprintf(payload, sizeof(payload), "{\"temp\":%.2f,\"cnn\":%.2f}", temp, cnn_out
    );

    if (mqtt_client) {
        int msg_id = esp_mqtt_client_publish(mqtt_client, mqtt_topic, payload, 0, 0, 0);
        if (msg_id < 0) {
            ESP_LOGE(TAG, "MQTT publish failed");
        } else {
            ESP_LOGI(TAG, "MQTT sent: %s", payload);
        }
    }

    int64_t t_mqtt_end = esp_timer_get_time();
    gpio_set_level(TRIGGER_PIN, 0);
    ESP_LOGI(TAG, "[MQTT] %lld us", (long long)(t_mqtt_end - t_mqtt_start));

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
    mqtt_init();
    register_ops();
    init_trigger_pin();

    ESP_LOGI(TAG, "Pokrenuto mjerenje ESP32-S3 ");
    ESP_LOGI(TAG, "CNN model: %d bytes", CNN_int8_tflite_len);

    vTaskDelay(pdMS_TO_TICKS(2000));

    while (true) {
        if (idle_start_time != 0) {
            int64_t idle_end = esp_timer_get_time();
            ESP_LOGI(TAG, "[MIROVANJE] %lld us", (long long)(idle_end - idle_start_time));
        }
        ciklus();
        vTaskDelay(pdMS_TO_TICKS(CYCLE_MS));
    }
}