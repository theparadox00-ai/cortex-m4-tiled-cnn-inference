#ifndef NN_TYPES_H
#define NN_TYPES_H

#include <stdint.h>

enum {
    OP_CONCAT,
    OP_MUL,
    OP_SUB,
    OP_ADD,
    OP_CONV,
    OP_DW,
    OP_AVGPOOL,
    OP_RESHAPE,
    OP_FC,
    OP_LOGISTIC
};

enum {
    MEM_RING,
    MEM_SD_IN,
    MEM_SD_OUT
};

typedef struct {
    uint16_t h, w, c;
    int16_t zp;
    uint32_t row_bytes;
    int16_t prod_layer;
    int8_t prod_group;
    int32_t whole_off;
    uint8_t spilled;
    uint32_t sd_off;
} nn_tensor_t;

typedef struct {
    uint8_t op;
    uint8_t n_in;
    uint16_t in[3];
    uint16_t out;
    uint8_t k_h, k_w, s_h, s_w, pad_t, pad_l;
    int16_t act_min, act_max;
    const int8_t *w;
    const int32_t *b;
    const int32_t *mult;
    const int8_t *shift;
    const int8_t *cst;
    int16_t cst_zp;
    uint8_t cst_first;
    int32_t m1;
    int8_t s1;
    int32_t m2;
    int8_t s2;
    int32_t mo;
    int8_t so;
    int8_t left_shift;
    const int8_t *lut;
} nn_layer_t;

typedef struct {
    uint8_t first, last, whole;
    uint8_t slot_first, slot_count;
    uint8_t dma_first, dma_count;
} nn_group_t;

typedef struct {
    uint16_t tensor;
    uint8_t mem;
    uint8_t rows;
    uint32_t off;
} nn_slot_t;

typedef struct {
    uint16_t tensor;
    uint8_t dir;
    uint8_t nbuf;
    uint32_t chunk;
    uint32_t off;
} nn_dma_t;

#endif
