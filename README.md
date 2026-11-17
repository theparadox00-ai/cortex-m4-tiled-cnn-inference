# cortex-m4-tiled-cnn-inference

Bare-metal tiled int8 CNN inference on STM32F401RE (Cortex-M4, 84 MHz, 96 KB RAM):
a MobileNetV2-style pollinator classifier run row by row with SD-card spilling,
following an offline tiling plan.

## Model
- MobileNetV2-style int8 classifier, 54 TFLite ops
- Input: 224x448x1 greyscale spectrogram
- Output: 1 sigmoid value (pollinator / no pollinator)

## How it fits in 96 KB
- Offline tiling plan: 15 layer groups, layers run row by row through small
  ring buffers inside a fixed arena
- Input and one spilled tensor (t94) are streamed to/from the SD card by DMA:
  62 SD loads and 13 SD stores per inference

## Progress log
### v0 - first working version
- Plain C int8 kernels following TFLite reference requantisation
- Built at -O0 (CubeIDE Debug default)
- Measured latency: 27,603.23 ms per inference

## Build
- STM32CubeIDE, NUCLEO-F401RE
- Optimization: -O3
- Flags: -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard
- Copy src/ to Core/Src and inc/ to Core/Inc
- Generate model_data.c/h and model_plan.c/h with tools/extract_model.py

## How latency is measured
- The Cortex-M4 DWT cycle counter (DWT->CYCCNT) increases by 1 every CPU clock.
- At 84 MHz: 84,000,000 cycles per second = 84,000 cycles per millisecond.
- The counter is read before and after the inference:

      cycles     = CYCCNT_end - CYCCNT_start
      latency_ms = cycles / 84000

- Included: all 54 layers and every SD load/store of the plan.
- Not included: copying the input image into place before the run.
- SD wait is counted separately (cycles the CPU spent waiting for a DMA buffer):

      compute_ms = latency_ms - sd_wait_ms

  ### v0.1 - build fix
- Mistake: -O0 means no compiler optimisation (every value goes back to the
  stack after each statement, no inlining, no loop optimisation), so the v0
  timing measured unoptimised code
- Learned: -O0 is for debugging; performance must be measured on an optimised build
- Change: build at -O3
