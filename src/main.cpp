/*********
 Name: main.cpp
 Authors: Landen Doty, Sepehr Noori
 Description: Main driver code for parksense application
 Date: 12/3/2023

 Adapted from: 
    https://RandomNerdTutorials.com/esp32-cam-take-photo-save-microsd-card
    https://www.instructables.com/ESP32-CAM-Person-Detection-Expreiment-With-TensorF/
    Examples from EECS700, taught by Dr. Heechul Yun
*********/

#include "esp_camera.h"
#include <WiFi.h>
#include <WiFiClient.h>
#include "Arduino.h"
#include "soc/soc.h"           // Disable brownour problems
#include "soc/rtc_cntl_reg.h"  // Disable brownour problems
#include <EEPROM.h>            // read and write from flash memory
#include "CNN.h"
#include "image.h"

#define DEBUG_TFLITE 1

#define CAMERA_MODEL_XIAO_ESP32S3 // Has PSRAM

// define the number of bytes you want to access
#define EEPROM_SIZE 1

#define INPUT_W 96
#define INPUT_H 96
#define LED_BUILT_IN 21

// Pin definition for CAMERA_MODEL_AI_THINKER
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39

#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13

#define SETUP_AP 1 // 1=AP, 0=STA
const char* ssid = "esp32-cam";
const char* password = "super-strong-password";

const String api_ip = "192.168.4.2";
const String api_path = "/parking-lot/lot1";
const int api_port = 5000;
WiFiClient client;

const String path_pred = "/predictions.txt";
const String path_img = "/image";

int pictureNumber = 0;

CNN *cnn;

uint32_t rgb565torgb888(uint16_t color)
{
    uint32_t r, g, b;
    r = g = b = 0; 
    r = (color >> 11) & 0x1F;
    g = (color >> 5) & 0x3F;
    b = color & 0x1F;
    r = (r << 3) | (r >> 2);
    g = (g << 2) | (g >> 4);
    b = (b << 3) | (b >> 2);
    return (r << 16) | (g << 8) | b;
}

// converts the image to rgb888 format and updates the input tensor
int GetImage(camera_fb_t * fb, TfLiteTensor* input) 
{
    assert(fb->format == PIXFORMAT_RGB565);
    // Trimming Image
    int post = 0;
    int startx = (fb->width - INPUT_W) / 2;
    int starty = (fb->height - INPUT_H);
    for (int y = 0; y < INPUT_H; y++) {
        for (int x = 0; x < INPUT_W; x++) {
            int getPos = (starty + y) * fb->width + startx + x;
            uint16_t color = ((uint16_t *)fb->buf)[getPos];
            uint32_t rgb = rgb565torgb888(color);

            float *image_data = tflite::GetTensorData<float>(input);
            //delay(2000);

            image_data[post * 3 + 0] = ((rgb >> 16) & 0xFF) / 255.0;  // R
            image_data[post * 3 + 1] = ((rgb >> 8) & 0xFF) / 255.0;   // G
            image_data[post * 3 + 2] = (rgb & 0xFF) / 255.0;          // B
            post++;
        }
    }
    return 0;
}

// have a function here to write to web server like l3

// also have a function to setup server? 

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); //disable brownout detector

  Serial.begin(115200);
  while(!Serial);
  Serial.setDebugOutput(false);
  
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_96X96;
  config.pixel_format = PIXFORMAT_RGB565; // PIXFORMAT_JPEG; // for streaming
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  
  cnn = new CNN();
  // Init Camera
  pinMode(LED_BUILT_IN, OUTPUT); // Set the pin as output
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }
  Serial.println("camera init");
  // Wi-Fi connection
  // SETUP_AP allows MCU to function as an access point for local testing
  #if SETUP_AP==1
    WiFi.softAP(ssid, password);
    Serial.println("AP available");
    Serial.println(WiFi.softAPIP());
  #else
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
      delay(500);
      Serial.print(".");
    }
    Serial.println("");
    Serial.println("WiFi connected");
    Serial.println(WiFi.localIP());
  #endif

}

void loop() {

  // take picture
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  uint64_t start, dur_prep, dur_infer;

  fb = esp_camera_fb_get();

  if(!fb) {
    Serial.println("Camera capture failed");
    res = ESP_FAIL;
  }
  // classify
  else{
    #if DEBUG_TFLITE==0
      // Use camera image
      start = esp_timer_get_time();
      GetImage(fb, cnn->getInput());
      dur_prep = esp_timer_get_time() - start;
    #else
      // Use a static image for debugging
       memcpy(cnn->getInput()->data.f, img_data, sizeof(img_data));
      Serial.printf("input: %.3f %.3f %.3f %.3f...\n", 
        cnn->getInput()->data.f[0], cnn->getInput()->data.f[1], cnn->getInput()->data.f[2], cnn->getInput()->data.f[3]);
    #endif
    Serial.println("making prediction");
    // do inference
    start = esp_timer_get_time();
    cnn->predict();
    dur_infer = esp_timer_get_time() - start;
    printf("output: %.3f %.3f %.3f %.3f...\n", 
        cnn->getOutput()->data.f[0], cnn->getOutput()->data.f[1], cnn->getOutput()->data.f[2], cnn->getOutput()->data.f[3]);
  }
  Serial.printf("Preprocessing: %llu ms, Inference: %llu ms\n", dur_prep/1000, dur_infer/1000);
  esp_camera_fb_return(fb); 
}
