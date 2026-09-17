idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    REQUIRES
        esp-tflite-micro
        driver
        nvs_flash
        esp_wifi
        esp_netif
        esp_timer
   PRIV_REQUIRES mqtt
)