/*
 * Copyright (c) 2021 Arm Limited and Contributors. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 * 
 */

#include <stdio.h>

#include "pico/stdlib.h"
#include "hardware/pwm.h"

extern "C" {
#include "pico/pdm_microphone.h"
}

#include "tflite_model.h"

#include "dsp_pipeline.h"
#include "ml_model.h"
#include "ssd1306_i2c.h"

// constants
#define SAMPLE_RATE       16000
#define FFT_SIZE          256
#define SPECTRUM_SHIFT    4
#define INPUT_BUFFER_SIZE ((FFT_SIZE / 2) * SPECTRUM_SHIFT)
#define INPUT_SHIFT       0

// microphone configuration
const struct pdm_microphone_config pdm_config = {
    // GPIO pin for the PDM DAT signal
    .gpio_data = 5,

    // GPIO pin for the PDM CLK signal
    .gpio_clk = 4,

    // PIO instance to use
    .pio = pio0,

    // PIO State Machine instance to use
    .pio_sm = 0,

    // sample rate in Hz
    .sample_rate = SAMPLE_RATE,

    // number of samples to buffer
    .sample_buffer_size = INPUT_BUFFER_SIZE,
};

q15_t capture_buffer_q15[INPUT_BUFFER_SIZE];
volatile int new_samples_captured = 0;

q15_t input_q15[INPUT_BUFFER_SIZE + (FFT_SIZE / 2)];

DSPPipeline dsp_pipeline(FFT_SIZE);
MLModel ml_model(tflite_model, 128 * 1024);

int8_t* scaled_spectrum = nullptr;
int32_t spectogram_divider;
float spectrogram_zero_point;

void on_pdm_samples_ready();

int main( void )
{
    // initialize stdio
    stdio_init_all();
    // let's not wait for USB to be connected, we have OLED
    //while (!stdio_usb_connected())
    //{
    //   tight_loop_contents();
    //}
    printf("hello pico chainsaw detection\n");

    i2c_init(I2C_PORT, 400 * 1000);
    gpio_set_function(14, GPIO_FUNC_I2C);
    gpio_set_function(15, GPIO_FUNC_I2C);
    gpio_pull_up(14);
    gpio_pull_up(15);

    ssd1306_main_init();
    ssd1306_display_text_line("OLED READY");

    if (!ml_model.init()) {
        printf("Failed to initialize ML model!\n");
        while (1) { tight_loop_contents(); }
    }

    if (!dsp_pipeline.init()) {
        printf("Failed to initialize DSP Pipeline!\n");
        while (1) { tight_loop_contents(); }
    }

    scaled_spectrum = (int8_t*)ml_model.input_data();
    spectogram_divider = 64 * ml_model.input_scale(); 
    spectrogram_zero_point = ml_model.input_zero_point();

    // initialize the PDM microphone
    if (pdm_microphone_init(&pdm_config) < 0) {
        printf("PDM microphone initialization failed!\n");
        while (1) { tight_loop_contents(); }
    }

    // set callback that is called when all the samples in the library
    // internal sample buffer are ready for reading
    pdm_microphone_set_samples_ready_handler(on_pdm_samples_ready);

    // start capturing data from the PDM microphone
    if (pdm_microphone_start() < 0) {
        printf("PDM microphone start failed!\n");
        while (1) { tight_loop_contents(); }
    }
    if (1){
        while (1) {
                // wait for new samples
                while (new_samples_captured == 0) {
                    tight_loop_contents();
                }
                new_samples_captured = 0;

                dsp_pipeline.shift_spectrogram(scaled_spectrum, SPECTRUM_SHIFT, 124);

                // move input buffer values over by INPUT_BUFFER_SIZE samples
                memmove(input_q15, &input_q15[INPUT_BUFFER_SIZE], (FFT_SIZE / 2));

                // copy new samples to end of the input buffer with a bit shift of INPUT_SHIFT
                arm_shift_q15(capture_buffer_q15, INPUT_SHIFT, input_q15 + (FFT_SIZE / 2), INPUT_BUFFER_SIZE);
            
                for (int i = 0; i < SPECTRUM_SHIFT; i++) {
                    dsp_pipeline.calculate_spectrum(
                        input_q15 + i * ((FFT_SIZE / 2)),
                        scaled_spectrum + (129 * (124 - SPECTRUM_SHIFT + i)),
                        spectogram_divider, spectrogram_zero_point
                    );
                }

                float prediction = ml_model.predict();

                if (prediction < 0.5) {
                printf("\tChainsaw\t\tdetected!\t(prediction = %f)\n\n", prediction);
                ssd1306_display_text_line("DETECTED");
                } else {
                printf("\tChainsaw\t\tnot detected!\t(prediction = %f)\n\n", prediction);
                ssd1306_display_text_line("NOT DETECTED");
                }
        }
            // pwm_set_chan_level(pwm_slice_num, pwm_chan_num, prediction * 255);
    }else{
            while (1) {
                printf("On standby");
                ssd1306_display_text_line("On standby");
            }

        
    }

    return 0;
}

void on_pdm_samples_ready()
{
    // callback from library when all the samples in the library
    // internal sample buffer are ready for reading 

    // read in the new samples
    new_samples_captured = pdm_microphone_read(capture_buffer_q15, INPUT_BUFFER_SIZE);
}
