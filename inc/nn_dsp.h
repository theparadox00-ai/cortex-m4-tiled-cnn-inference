#ifndef NN_DSP_H
#define NN_DSP_H

#include <stdint.h>
#include <string.h>

#if defined(__ARM_FEATURE_DSP)

static inline uint32_t d_sxtb16(uint32_t x) { uint32_t r; __asm volatile("sxtb16 %0, %1" : "=r"(r) : "r"(x)); return r; }
static inline uint32_t d_sxtb16_ror8(uint32_t x) { uint32_t r; __asm volatile("sxtb16 %0, %1, ror #8" : "=r"(r) : "r"(x)); return r; }
static inline uint32_t d_sxtab16(uint32_t a, uint32_t x) { uint32_t r; __asm volatile("sxtab16 %0, %1, %2" : "=r"(r) : "r"(a), "r"(x)); return r; }
static inline uint32_t d_sxtab16_ror8(uint32_t a, uint32_t x) { uint32_t r; __asm volatile("sxtab16 %0, %1, %2, ror #8" : "=r"(r) : "r"(a), "r"(x)); return r; }
static inline int32_t d_smlad(uint32_t a, uint32_t b, int32_t acc) { int32_t r; __asm volatile("smlad %0, %1, %2, %3" : "=r"(r) : "r"(a), "r"(b), "r"(acc)); return r; }

#else

static inline uint32_t d_pack(int32_t lo, int32_t hi) { return ((uint32_t)lo & 0xFFFFu) | ((uint32_t)hi << 16); }
static inline uint32_t d_sxtb16(uint32_t x) { return d_pack((int8_t)(x & 0xFF), (int8_t)((x >> 16) & 0xFF)); }
static inline uint32_t d_sxtb16_ror8(uint32_t x) { return d_sxtb16((x >> 8) | (x << 24)); }
static inline uint32_t d_sxtab16(uint32_t a, uint32_t x)
{
    uint32_t b = d_sxtb16(x);
    return d_pack((int16_t)(a & 0xFFFF) + (int16_t)(b & 0xFFFF), (int16_t)(a >> 16) + (int16_t)(b >> 16));
}
static inline uint32_t d_sxtab16_ror8(uint32_t a, uint32_t x) { return d_sxtab16(a, (x >> 8) | (x << 24)); }
static inline int32_t d_smlad(uint32_t a, uint32_t b, int32_t acc)
{
    return acc + (int16_t)(a & 0xFFFF) * (int16_t)(b & 0xFFFF) + (int16_t)(a >> 16) * (int16_t)(b >> 16);
}

#endif

static inline uint32_t d_rd32(const void *p) { uint32_t v; memcpy(&v, p, 4); return v; }

#endif
