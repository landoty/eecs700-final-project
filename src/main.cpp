/*********
 Name: main.cpp
 Authors: Landen Doty, Sepehr Noori
 Description: Main driver code for parksense application
 Date: 12/6/2023

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
#include "esp_http_server.h"
#include "CNN.h"
#include "image.h"

#define DEBUG_TFLITE 1

#define CAMERA_MODEL_XIAO_ESP32S3 // Has PSRAM

#define INPUT_W 96
#define INPUT_H 96

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

#define LED_BUILT_IN 21

// Wifi Definitions
#define SETUP_AP 1 // 1=AP, 0=STA
const char* ssid = "esp32-cam";
const char* password = "super-strong-password";

// Constants for streaming content
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// Global pointer to the TFLite Model
CNN *cnn;

// Bounding box coordinates
int xmin, ymin, xmax, ymax;

// Convert rgb565 colors to rgb888, Credit: Dr. Heechul Yun
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

// converts the image to rgb888 format and updates the input tensor, Credit: Dr. Heechul Yun
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

// Draw a bounding box on the iamge
int draw_bb(camera_fb_t * fb, int xmin, int ymin, int xmax, int ymax) 
{
    assert(fb->format == PIXFORMAT_RGB565);
    int post = 0;
    int startx = (fb->width - INPUT_W) / 2;
    int starty = (fb->height - INPUT_H);
    int x, y, getPos;
    uint16_t WHITE = 0xFFFF;

    // Draw vertical bounds
    for(y = ymin; y <= ymax; y++){
      x = xmin;
      getPos = (starty + y) * fb->width + startx + x;
      ((uint16_t *)fb->buf)[getPos] = WHITE;

      x = xmax;
      getPos = (starty + y) * fb->width + startx + x;
      ((uint16_t *)fb->buf)[getPos] = WHITE;
    }

    // Draw horiztonal bounds
    for(x = xmin; x <= xmax; x++){
      y = ymin;
      getPos = (starty + y) * fb->width + startx + x;
      ((uint16_t *)fb->buf)[getPos] = WHITE;

      y = ymax;
      getPos = (starty + y) * fb->width + startx + x;
      ((uint16_t *)fb->buf)[getPos] = WHITE;
    }
    return 0;
}

// Convert normalized yolo coordinates (x_center, y_center, width, height) to pascal voc bb coordinates (xmin, yminx, xmax, ymax)
void yolo_to_pascal_voc(float yolo_x, float yolo_y, float yolo_width, float yolo_height, int imgWidth, int imgHeight, int *pascal_xmin, int *pascal_ymin, int *pascal_xmax, int *pascal_ymax) {
    // Convert YOLO coordinates to Pascal VOC coordinates
    *pascal_xmin = (int)((yolo_x - yolo_width / 2) * imgWidth);
    *pascal_ymin = (int)((yolo_y - yolo_height / 2) * imgHeight);
    *pascal_xmax = (int)((yolo_x + yolo_width / 2) * imgWidth);
    *pascal_ymax = (int)((yolo_y + yolo_height / 2) * imgHeight);

    // Clamp coordinates to image boundaries
    *pascal_xmin = (*pascal_xmin < 0) ? 0 : *pascal_xmin;
    *pascal_ymin = (*pascal_ymin < 0) ? 0 : *pascal_ymin;
    *pascal_xmax = (*pascal_xmax > imgWidth - 1) ? imgWidth - 1 : *pascal_xmax;
    *pascal_ymax = (*pascal_ymax > imgHeight - 1) ? imgHeight - 1 : *pascal_ymax;
}

// Helper function to run model inference 
void do_inference(camera_fb_t * fb) {
  uint64_t start, dur_prep, dur_infer, dur_post;
  #if DEBUG_TFLITE==0
      // Use camera image
      start = esp_timer_get_time();
      GetImage(fb, cnn->getInput());
      dur_prep = esp_timer_get_time() - start;
    #else
      // Use a static image for debugging
       memcpy(cnn->getInput()->data.f, img_data, sizeof(img_data));
       dur_prep = 0;
    #endif
    // do inference
    start = esp_timer_get_time();
    cnn->predict();
    dur_infer = esp_timer_get_time() - start;

    start = esp_timer_get_time();
    yolo_to_pascal_voc(cnn->getOutput()->data.f[0], cnn->getOutput()->data.f[1], cnn->getOutput()->data.f[2], cnn->getOutput()->data.f[3], 96, 96, &xmin, &ymin, &xmax, &ymax);
    draw_bb(fb, xmin, ymin, xmax, ymax);
    dur_post = esp_timer_get_time() - start;

    Serial.printf("Preprocessing: %llu ms, Inference: %llu ms, Postprocessing: %llu ms\n", dur_prep/1000, dur_infer/1000, dur_post/1000);
}

// Streaming results to HTTP, Credit Dr. Heechul Yun
httpd_handle_t stream_httpd = NULL;
static esp_err_t prediction_stream_handler(httpd_req_t *req){
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len = 0;
  uint8_t * _jpg_buf = NULL;
  char * part_buf[64];
  
  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if(res != ESP_OK){
    return res;
  }

  while(true){
    fb = esp_camera_fb_get();
    do_inference(fb);
    if (!fb) {
      Serial.println("Camera capture failed");
      res = ESP_FAIL;
    } else {
      if(fb->width > 0){
        if(fb->format != PIXFORMAT_JPEG){
          bool jpeg_converted = frame2jpg(fb, 80, &_jpg_buf, &_jpg_buf_len);
          esp_camera_fb_return(fb);
          fb = NULL;
          if(!jpeg_converted){
            Serial.println("JPEG compression failed");
            res = ESP_FAIL;
          }
        } else {
          _jpg_buf_len = fb->len;
          Serial.printf("%d\n", _jpg_buf_len);
          _jpg_buf = fb->buf;
        }
      }
    }
    if(res == ESP_OK){
      size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, _jpg_buf_len);
      res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
    }
    if(res == ESP_OK){
      res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }
    if(res == ESP_OK){
      res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
    }
    if(fb){
      esp_camera_fb_return(fb);
      fb = NULL;
      _jpg_buf = NULL;
    } else if(_jpg_buf){
      free(_jpg_buf);
      _jpg_buf = NULL;
    }
    if(res != ESP_OK){
      break;
    }
  }
  return res;
}

// Starting HTTP Server, Credit Dr. Heechul Yun
void startCameraServer(){
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;

  httpd_uri_t index_uri = {
    .uri       = "/",
    .method    = HTTP_GET,
    .handler   = prediction_stream_handler,
    .user_ctx  = NULL
  };
  
  //Serial.printf("Starting web server on port: '%d'\n", config.server_port);
  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &index_uri);
  }
}

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); //disable brownout detector

  Serial.begin(115200);

  #if DEBUG_TFLITE==1
    while(!Serial);
  #else
  #endif
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
  config.pixel_format = PIXFORMAT_RGB565;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  // Instantiate model object  
  cnn = new CNN();
  pinMode(LED_BUILT_IN, OUTPUT); // Set the pin as output
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }
  // Wi-Fi connection
  // SETUP_AP allows MCU to function as an access point for local testing
  #if SETUP_AP==1
    WiFi.softAP(ssid, password);
    Serial.print("Stream ready! Connect to: ");
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
  startCameraServer();
}

void loop() {
  delay(0);
}
