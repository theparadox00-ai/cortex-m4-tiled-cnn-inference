#ifndef ENGINE_H
#define ENGINE_H

#include <stdint.h>
#include "model_plan.h"

#define NN_SD_BASE_SECTOR 2048u

#define NN_OK 0
#define NN_ERR_DEADLOCK (-1)
#define NN_ERR_SD (-2)

int nn_run(uint32_t input_sector, int8_t *out, uint32_t out_len);
uint32_t nn_group_cycles(int g);
uint32_t nn_group_wait(int g);
uint32_t nn_io_stall_cycles(void);
int nn_fail_layer(void);
uint8_t *nn_scratch(void);

uint32_t nn_now(void);
void nn_row_hook(int tensor, int row, const int8_t *p, uint32_t n);

#endif
