/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "spi.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "stm32l4xx_hal.h" // so VSCode syntax checking can find HAL definitions

#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>

// AI model
#include "ai_platform.h"
#include "network.h"
#include "network_data.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

// The number of accelerometer channels and samples per channel to collect for inference.  
#define INFERENCE_ACCEL_CHANNELS 3 // X, Y, Z
#define INFERENCE_SAMPLES_PER_CHANNEL 26 // Number of samples to collect per channel for inference

// In this lab, inference input data is expected to be in units of 1/4 g based on the data scale
// factor used in training (see W27-W6-PA).
// Raw values returned by the accelerometer are in units of milli-gravities, so divide by 4000 to
// convert raw values into the value ranges expected for inference.
#define INFERENCE_ACCEL_RAW_DATA_DIVISOR (4000.0)  // divide milli-gravities by 4000 to pre-process

// See the LIS3DH datasheet for more details on these register addresses and settings:
// Read/Write Bit
#define LIS3DH_READ 0x80
#define LIS3DH_WRITE 0x00
#define LIS3DH_READ_TXFR (true) // cleans up function calls to accel_txfr
#define LIS3DH_WRITE_TXFR (false) 
// Register Addresses
#define LIS3DH_WHOAMI_REG 0x0F
#define LIS3DH_CTRL_REG1 0x20 
#define LIS3DH_CTRL_REG3 0x22
#define LIS3DH_CTRL_REG4 0x23
#define LIS3DH_STATUS_REG 0x27
#define LIS3DH_OUT_X_L 0x28 // Output registers for X, Y, and Z acceleration data (6 bytes total)  
#define LIS3DH_OUT_X_H 0x29
#define LIS3DH_OUT_Y_L 0x2A    
#define LIS3DH_OUT_Y_H 0x2B
#define LIS3DH_OUT_Z_L 0x2C
#define LIS3DH_OUT_Z_H 0x2D 
// Register Values
#define LIS3DH_WHOAMI_REG_VAL 0x33 // Expected value in WHO_AM_I register for LIS3DH
#define LIS3DH_CTRL_REG1_25HZ 0x37 // 25 Hz data rate, all axes enabled
#define LIS3DH_CTRL_REG3_INT1_ZYXDA 0x10 // Enable interrupt on INT1 pin when new ZYX data is available
#define LIS3DH_CTRL_REG4_HR 0x08 // High resolution mode enabled  

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
// Set by the USART2 Rx Complete callback when a byte is received, 
// used in the EXTI callback to determine when to read data and run inference.  
static volatile bool sample_data = false; 

// Set by the EXTI callback function when a buffer of accelerometer data is ready for inference,
// used in the main loop to determine when to run inference on the collected data.
static volatile bool data_ready_for_inference = false;

// Buffer to hold data received from accelerometer for inference
// static int16_t accel_raw_data_buffer[INFERENCE_ACCEL_CHANNELS * INFERENCE_SAMPLES_PER_CHANNEL]; // Original Line from Checkpoints 1 & 2

// Double buffer for data collection: 2 rows each holding a full inference batch
static volatile int16_t accel_raw_data_buffer[2][INFERENCE_ACCEL_CHANNELS * INFERENCE_SAMPLES_PER_CHANNEL];

// Tracks which buffer the EXTI interrupt is currently filling (0 or 1)
static volatile uint8_t current_fill_buffer = 0;

// Tracks which buffer is full and ready for the main to process (reserve -1 for neither)
static volatile int8_t buffer_ready_for_inference = -1;

// Used by the inference code:
static ai_handle network;                                    // Pointer to the AI network instance
static float aiInData[AI_NETWORK_IN_1_SIZE];                 // Input data for inference - will hold the pre-processed accelerometer data
static float aiOutData[AI_NETWORK_OUT_1_SIZE];               // Output data from inference - will hold the inference results
static ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];  // Buffer to hold the activations used internally by the AI network during inference
// Labels for the output classes of the AI model - used for printing inference results.  Update as needed based on your specific model.
static const char* activities[AI_NETWORK_OUT_1_SIZE] = {     
  "stationary", 
  "walking", 
  "running",
  "spinning"
};
// Pointers to the input and output buffers used for AI inference - these are passed to the ai_run function
static ai_buffer *ai_input;     
static ai_buffer *ai_output;


/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

// Perform a SPI transfer to the LIS3DH accelerometer using manual GPIO control of the CS pin
// reg: accel register to read/write (7 bits)
// write: data to write to register, if writing.  Ignored if performing a read operation.
// is_read: set to true to perform a read operation, false for write operation
// returns the value read from the accelerometer, or 0 if performing a write operation
static uint8_t accel_txfr(const uint8_t reg, const uint8_t write, const bool is_read);

// Read the X, Y, and Z acceleration data from the LIS3DH accelerometer and store in the provided pointers
// x, y, z: pointers to int16_t variables where the read acceleration data will be stored.  
// Returns signed 16-bit integers in units of milli-gravities
static void accel_read(int16_t *x, int16_t *y, int16_t *z);

// Initialize the inference engine
static void AI_Init(void);

// Perform inference on input data, generate predicted class probabilities in the output buffer
static void AI_Run(float *pIn, float *pOut);

// Utility function to find the index of the maximum value in an array - used for determining the predicted class 
// from the output probabilities of the AI model 
static uint32_t argmax(const float *values, uint32_t len);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
// Generic HAL timer callback function
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) {
  // Toggle the LED connected to PB3 every time the TIM6 timer period elapses
  if (htim->Instance == TIM6) {
    HAL_GPIO_TogglePin(GPIOB, GPIO_PIN_3);
  }
}

/* Retarget printf to use USART2 for character output */
int __io_putchar(int data)
{
  // Send one byte at a time, using the maximum timeout delay possible
  HAL_UART_Transmit(&huart2, (uint8_t *)&data, 1, HAL_MAX_DELAY);
  return data;
}

// Handle UART Rx Complete interrupt - triggered when a byte is received over USART2
// This is used by the template code to start collecting a buffer of data for inference.
// Inference is run one time per key press.  Note this is not well-defended code in that
// multiple key-presses within the data collection window will restart data collection
// which may cause unexpected behavior.  This is template code only, to be removed when
// you convert this project to continuously collect data and run inference without
// waiting for user input.
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if(huart->Instance == USART2) // Check if the interrupt is from the correct UART instance
    {
      printf("Collecting Data for Inference...\r\n");
      // Set a flag to indicate to EXTI callback that it should fill a buffer of data for inference
      sample_data = true; 
    }
}

// Handle EXTI interrupts on GPIO pins
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
  static uint8_t accel_sample_count = 0; // Keep track of how many samples have been collected for inference
  // static bool collecting_data = false; // Flag to indicate whether we are currently collecting data for inference
  // PORT A Pin #3?
  if (GPIO_Pin == GPIO_PIN_3) {
    // Read the latest accelerometer data
      int16_t x, y, z;
      accel_read(&x, &y, &z);

    // Check if we should start collecting data for inference (set by USART Rx callback) 
    // Checkpoint 2 Logic
    // if(sample_data) {
    //   accel_sample_count = 0; // Reset sample count
    //   collecting_data = true; // Start collecting data for inference
    //   sample_data = false; // Reset flag set by USART Rx callback until the next key press
    // }
    // if( collecting_data ) {
    //   // Store the read data in the buffer for inference, converting to float and applying any necessary scaling (not applied here since it depends on the specific LIS3DH configuration settings you choose)
    //   accel_raw_data_buffer[accel_sample_count * INFERENCE_ACCEL_CHANNELS + 0] = x;
    //   accel_raw_data_buffer[accel_sample_count * INFERENCE_ACCEL_CHANNELS + 1] = y; 
    //   accel_raw_data_buffer[accel_sample_count * INFERENCE_ACCEL_CHANNELS + 2] = z;
    //   printf("%5d, %5d, %5d\r\n", x, y, z); // Print the raw data values for debugging
    //   accel_sample_count++; // Increment sample count
    //   // Check if we have collected enough samples for inference
    //   if( accel_sample_count >= INFERENCE_SAMPLES_PER_CHANNEL ) { 
    //     data_ready_for_inference = true; // Set flag to indicate to main loop that data is ready for inference
    //     collecting_data = false; // Stop collecting data until the next trigger from USART Rx callback
        
    //   }
    // }
    // Checkpoint 3 Logic:
    // Continuously store data in the current active buffer
    accel_raw_data_buffer[current_fill_buffer][accel_sample_count * INFERENCE_ACCEL_CHANNELS + 0] = x;
    accel_raw_data_buffer[current_fill_buffer][accel_sample_count * INFERENCE_ACCEL_CHANNELS + 1] = y;
    accel_raw_data_buffer[current_fill_buffer][accel_sample_count * INFERENCE_ACCEL_CHANNELS + 2] = z;

    accel_sample_count++;

    // Check if the current buffer is full (26 samples)
    if(accel_sample_count >= INFERENCE_SAMPLES_PER_CHANNEL) {
      // 1. Signal main() that this buffer is ready for inference
      buffer_ready_for_inference = current_fill_buffer;

      // 2. Switch the active fill buffer (toggle between 0 and 1)
      current_fill_buffer = 1 - current_fill_buffer;

      // 3. Reset sample count tot start filling the new buffer from the beginning
      accel_sample_count = 0;
    }
  }
}

// Perform a SPI transfer to the LIS3DH accelerometer using manual GPIO control of the CS pin
// Use the GPIOA Pin #4 as the CS pin (active low) and the SPI1 peripheral for communication
// reg: accel register to read/write (7 bits)
// write: data to write to register, if writing.  Ignored if performing a read operation.
// is_read: set to true to perform a read operation, false for write operation
// returns the value read from the accelerometer, or 0 if performing a write operation
static uint8_t accel_txfr(const uint8_t reg, const uint8_t write, const bool is_read) {
  // Set up transfer data values
  uint16_t rx_data; // return value from SPI transfer, if relevant
  // Upper byte is the register address with R/W bit, lower byte is the data to write (ignored for reads)
  uint16_t tx_data = ((reg & 0x7f) << 8) | (write & 0x00ff); 
  if( is_read ) {
    tx_data |= (LIS3DH_READ << 8); // Set the R/W bit for read operations
  }; // if write, LSB is 0 so no need to modify tx_data further

  // Assert CS pin (active low)
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_RESET); 
  // HAL API expects pointer to uint8_t, but configured for 16-bit transfers, so cast the 
  // pointer appropriately.  Also, only transfer 1 "unit" of data, which is 16 bits in this configuration.
  // Use the MAX_DELAY timeout to block indefinitely until the transfer is complete.
  HAL_SPI_TransmitReceive(&hspi1, (uint8_t*)&tx_data, (uint8_t *)&rx_data, 1, HAL_MAX_DELAY);
  // Deassert CS pin
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_SET); 

  // Mask off the upper byte, since the LIS3DH only returns 8 bits of data per transfer, even in 16-bit mode
  // (not strictly required due to casting to uint8_t, but good for clarity)
  return (uint8_t)(rx_data&0x00ff);
}

// Initialize the LIS3HD accelerometer
// (1) Check WHO_AM_I for valid response
// (2) Configure for high resolution mode, 25 Hz sampling
// (3) Generate interrupts when new ZYX data is available
void accel_init(void) {
  uint8_t who_am_i = accel_txfr(LIS3DH_WHOAMI_REG, 0x00, LIS3DH_READ_TXFR);
  if(who_am_i != LIS3DH_WHOAMI_REG_VAL) {
    printf("Failed to detect LIS3DH accelerometer. WHOAMI register returned: 0x%02X\r\n", who_am_i);
    while(1); // Halt execution if the accelerometer is not detected  
  } else {
    printf("LIS3DH accelerometer detected successfully!  Configuring...\r\n");
    accel_txfr(LIS3DH_CTRL_REG1, LIS3DH_CTRL_REG1_25HZ, LIS3DH_WRITE_TXFR); // 25 Hz data rate, all axes enabled
    accel_txfr(LIS3DH_CTRL_REG4, LIS3DH_CTRL_REG4_HR, LIS3DH_WRITE_TXFR); // High resolution mode enabled
    accel_txfr(LIS3DH_CTRL_REG3, LIS3DH_CTRL_REG3_INT1_ZYXDA, LIS3DH_WRITE_TXFR); // Enable interrupt on INT1 pin when new ZYX data is available
    printf("LIS3DH accelerometer configured successfully!\r\n");

    // Check if there is already data available (in case new data interrupt was triggered before we configured the accelerometer)
    // Flush any data waiting to be read - de-asserts interrupt pin if it is triggered
    if( HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_3) == GPIO_PIN_SET) {
      int16_t dummy;
      accel_read(&dummy, &dummy, &dummy); // Read and discard the available data to flush the accelerometer's output registers
    }
  }
}

// Read the X, Y, and Z acceleration data from the LIS3DH accelerometer and store in the provided pointers
// x, y, z: pointers to int16_t variables where the read acceleration data will be stored.  
// The LIS3DH outputs 16 bits of data for each axis, with the lower 8 bits in the _L register and the 
// upper 8 bits in the _H register.
void accel_read(int16_t *x, int16_t *y, int16_t *z) {
  // Register list to read
  uint8_t read_regs[6] = { 
    LIS3DH_OUT_X_L, 
    LIS3DH_OUT_X_H, 
    LIS3DH_OUT_Y_L, 
    LIS3DH_OUT_Y_H, 
    LIS3DH_OUT_Z_L, 
    LIS3DH_OUT_Z_H 
  }; 
  // Return values - use uint16_t to support bit-manipulations before converting to final signed 12-bit values
  uint16_t data[6];
  // Read the 6 acceleration data registers in a loop using accel_txfr.  
  for( int i = 0; i < 6; i++) {
    data[i] = accel_txfr(read_regs[i], 0x00, LIS3DH_READ_TXFR);
  }
  // Pack signed 12-bit values into 12 MSBs of 16-bit integers
  *x = ((data[1]&0x00FF) << 8) | (data[0] & 0x00FF); // Combine high and low bytes for X-axis
  *y = ((data[3]&0x00FF) << 8) | (data[2] & 0x00FF); // Combine high and low bytes for Y-axis
  *z = ((data[5]&0x00FF) << 8) | (data[4] & 0x00FF); // Combine high and low bytes for Z-axis
  // Now sign extend and shift right by 4 to get the final signed 12-bit values
  *x = (*x >> 4); // Shift right by 4 to discard unused bits
  *y = (*y >> 4); // Shift right by 4 to discard unused bits
  *z = (*z >> 4); // Shift right by 4 to discard unused bits
}

// Initialize the inference engine (based on the tensorflow-lite model initialization code generated by X-CUBE-AI)
static void AI_Init(void)
{
  ai_error err;

  /* Create a local array with the addresses of the activations buffers */
  const ai_handle act_addr[] = { activations };

  /* Create an instance of the model */
  err = ai_network_create_and_init(&network, act_addr, NULL);
  if (err.type != AI_ERROR_NONE) {
    printf("ai_network_create error - type=%d code=%d\r\n", err.type, err.code);
    Error_Handler();
  }

  /* Retrieve pointers to the model's input and output tensors */
  ai_input = ai_network_inputs_get(network, NULL);
  ai_output = ai_network_outputs_get(network, NULL);
}

// Run inference on input data, generate predicted class probabilities in the output buffer
static void AI_Run(float *pIn, float *pOut)
{
  ai_i32 batch;
  ai_error err;

  /* Update IO handlers with the data payload */
  ai_input[0].data = AI_HANDLE_PTR(pIn);
  ai_output[0].data = AI_HANDLE_PTR(pOut);

  // Call the network run function generated by X-CUBE-AI to perform inference.  
  // The function will return the number of batches processed, which should be 1 in this case since we are only 
  // unning inference on one input at a time.  If the return value is not 1, then an error occurred during inference 
  // and we can retrieve details about the error using ai_network_get_error.
  batch = ai_network_run(network, ai_input, ai_output);
  if (batch != 1) {
    err = ai_network_get_error(network);
    printf("AI ai_network_run error - type=%d code=%d\r\n", err.type, err.code);
    Error_Handler();
  }
}

// Utility function to find the index of the maximum value in an array - used for determining the predicted class
// from the output probabilities of the AI model
// values: pointer to the array of float values to search through 
// len: length of the array
// returns the index of the maximum value in the array
static uint32_t argmax(const float * values, uint32_t len)
{
  float max_value = values[0];
  uint32_t max_index = 0;
  for (uint32_t i = 1; i < len; i++) {
    if (values[i] > max_value) {
      max_value = values[i];
      max_index = i;
    }
  }
  return max_index;
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
  uint8_t rx_byte; // variable to hold received byte from UART interrupt
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */


  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART2_UART_Init();
  MX_SPI1_Init();
  MX_TIM6_Init();
  /* USER CODE BEGIN 2 */

  // 1. DIRECT HARDWARE TEST: Bypass printf entirely. Some debugging statements when I was having trouble getting information to display on the serial terminal after I regenereated my code again after the initial generation because of new model.
  // char test_msg[] = "\r\n--- UART HARDWARE ALIVE ---\r\n";
  // HAL_UART_Transmit(&huart2, (uint8_t*)test_msg, 31, HAL_MAX_DELAY);

  // Print a startup message to know the program is running
  printf("Lab 7: Human Motion Classification with a Neural Network\r\n");

  // Force printf to flush immediately (if it didn't fail from memory)
  fflush(stdout);

  // Initialize the AI inference engine - do this early, before anything tries 
  // to access the AI input and output buffers, since the initialization function 
  // sets up the necessary pointers and structures for inference to work correctly.
  AI_Init();

  // Start the Timer6 timer in interrupt mode
  HAL_TIM_Base_Start_IT(&htim6);

  // Initialize the accelerometer (LIS3DH)
  accel_init();

  // Template code: wait for a user keypress to fill a buffer of data and run inference on it using a neural network. 
  // Note this does not permanently enable the UART Rx interrupt - it only enables it once at the beginning and then 
  // again after inference is run, so inference will only be triggered once per key press.  
  // You will remove this when you convert the code to continuously collect data and run inference 
  // HAL_UART_Receive_IT(&huart2, (uint8_t*)&rx_byte, 1); // Start UART reception in interrupt mode.  

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */
    if(buffer_ready_for_inference != -1){
      HAL_GPIO_WritePin(GPIOB, GPIO_PIN_5, GPIO_PIN_SET);
      // Get the index of the ready buffer and reset the flag so we don't process it twice
      int process_buf = buffer_ready_for_inference;
      buffer_ready_for_inference = -1;

      // Scale raw data from the READY buffer and convert to float for inference
      for(int i = 0; i < INFERENCE_ACCEL_CHANNELS * INFERENCE_SAMPLES_PER_CHANNEL; i++){
        aiInData[i] = (float)accel_raw_data_buffer[process_buf][i] / INFERENCE_ACCEL_RAW_DATA_DIVISOR;
      }

      // Run the AI inference on scaled input data
      AI_Run(aiInData, aiOutData);

      // Print the predicted probability for each class
        printf("Inference Results: ");
        for( int i = 0; i < AI_NETWORK_OUT_1_SIZE; i++ ) {
          printf("%s: %.3f ", activities[i], aiOutData[i]); 
        } 
        printf("\r\n");

        // Find the index of the maximum output value
        int predicted_class = argmax(aiOutData, AI_NETWORK_OUT_1_SIZE);

        // Print the predicted class
        printf("Predicted class: %d (%s)\r\n\n", predicted_class, activities[predicted_class]);

        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_5, GPIO_PIN_RESET);
    }
    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 1;
  RCC_OscInitStruct.PLL.PLLN = 10;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV7;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */
// Needed to add this because I had to retrain the model after initial code generation and printf was having some problems. From AI.
int _write(int file, char *ptr, int len) {
    HAL_UART_Transmit(&huart2, (uint8_t*)ptr, len, HAL_MAX_DELAY);
    return len;
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
