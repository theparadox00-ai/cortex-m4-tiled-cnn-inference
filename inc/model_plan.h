#ifndef MODEL_PLAN_H
#define MODEL_PLAN_H
#include "nn_types.h"
#define NN_N_TENSORS 55
#define NN_N_LAYERS 54
#define NN_N_GROUPS 15
#define NN_ARENA_BYTES 77184u
#define NN_INPUT_TENSOR 0
#define NN_OUTPUT_TENSOR 54
#define NN_INPUT_BYTES 100352u
#define NN_INPUT_PART_OFFSET 0x00100000u
#define NN_SPILL_END_BYTE 0x00124C00u
#define NN_PLAN_LOADS 62u
#define NN_PLAN_STORES 13u
#define NN_PLAN_RD_BYTES 150528u
#define NN_PLAN_WR_BYTES 50176u
extern const nn_tensor_t nn_tensors[NN_N_TENSORS];
extern const nn_layer_t nn_layers[NN_N_LAYERS];
extern const nn_group_t nn_groups[NN_N_GROUPS];
extern const nn_slot_t nn_slots[];
extern const nn_dma_t nn_dmas[];
extern const float nn_out_scale;
extern const int32_t nn_out_zp;
#endif
