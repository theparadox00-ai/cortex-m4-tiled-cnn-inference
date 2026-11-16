#include "usart.h"
#include "stm32f4xx.h"

void uart_init(void)
{
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
    RCC->APB1ENR |= RCC_APB1ENR_USART2EN;
    (void)RCC->APB1ENR;
    GPIOA->MODER = (GPIOA->MODER & ~(0xFu << 4)) | (0xAu << 4);
    GPIOA->AFR[0] = (GPIOA->AFR[0] & ~(0xFFu << 8)) | (0x77u << 8);
    USART2->BRR = (42000000u + 57600u) / 115200u;
    USART2->CR1 = USART_CR1_TE | USART_CR1_RE | USART_CR1_UE;
}

void uart_putc(char c)
{
    while (!(USART2->SR & USART_SR_TXE)) {}
    USART2->DR = (uint8_t)c;
}

void uart_puts(const char *s)
{
    while (*s) {
        if (*s == '\n') uart_putc('\r');
        uart_putc(*s++);
    }
}

void uart_putu(uint32_t v)
{
    char b[11];
    int i = 10;
    b[i] = 0;
    do {
        b[--i] = (char)('0' + v % 10u);
        v /= 10u;
    } while (v);
    uart_puts(&b[i]);
}

void uart_puti(int32_t v)
{
    if (v < 0) {
        uart_putc('-');
        uart_putu((uint32_t)(-(int64_t)v));
    } else {
        uart_putu((uint32_t)v);
    }
}

void uart_ms(uint32_t cycles)
{
    uint32_t us = cycles / 84u;
    uart_putu(us / 1000u);
    uart_putc('.');
    uint32_t f = (us % 1000u) / 10u;
    if (f < 10) uart_putc('0');
    uart_putu(f);
}
