#ifndef USART_H
#define USART_H

#include <stdint.h>

void uart_init(void);
void uart_putc(char c);
void uart_puts(const char *s);
void uart_putu(uint32_t v);
void uart_puti(int32_t v);
void uart_ms(uint32_t cycles);

#endif
