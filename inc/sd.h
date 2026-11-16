#ifndef SD_H
#define SD_H

#include <stdint.h>

#define SD_OK 0
#define SD_BUSY 1
#define SD_ERR (-1)

int sd_init(void);
int sd_start_read(uint32_t sector, uint8_t *buf, uint32_t nblocks);
int sd_start_write(uint32_t sector, const uint8_t *buf, uint32_t nblocks);
int sd_service(void);
int sd_read(uint32_t sector, uint8_t *buf, uint32_t nblocks);
int sd_write(uint32_t sector, const uint8_t *buf, uint32_t nblocks);

#endif
