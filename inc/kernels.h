#ifndef KERNELS_H
#define KERNELS_H

#include "nn_types.h"

void k_conv_row(const nn_layer_t *L, const nn_tensor_t *ti, const nn_tensor_t *to, const int8_t *const *rows, int8_t *out);
void k_pw_row(const nn_layer_t *L, const nn_tensor_t *ti, const nn_tensor_t *to, const int8_t *in, int8_t *out);
void k_dw_row(const nn_layer_t *L, const nn_tensor_t *ti, const nn_tensor_t *to, const int8_t *const *rows, int8_t *out);
void k_avgpool_row(const nn_layer_t *L, const nn_tensor_t *ti, const nn_tensor_t *to, const int8_t *const *rows, int8_t *out);
void k_add_row(const nn_layer_t *L, const nn_tensor_t *t1, const nn_tensor_t *t2, const nn_tensor_t *to, const int8_t *a, const int8_t *b, int8_t *out);
void k_cst_row(const nn_layer_t *L, const nn_tensor_t *ti, const nn_tensor_t *to, const int8_t *in, int8_t *out);
void k_concat_row(const nn_layer_t *L, const nn_tensor_t *const *ti, const int8_t *const *in, const nn_tensor_t *to, int8_t *out);
void k_fc(const nn_layer_t *L, const nn_tensor_t *ti, const nn_tensor_t *to, const int8_t *in, int8_t *out);
void k_logistic(const nn_layer_t *L, const int8_t *in, int8_t *out, uint32_t n);

#endif
