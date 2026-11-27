/*
 * led.h - Interface between the C code and Shub's assembly LED driver (led.s).
 *
 * The C side only calls these two functions. Shub implements them in GNU
 * assembly, following the pattern from Lab 7 (setup / func_led_on).
 *
 * Calling convention (AAPCS, Lab 5): first argument in r0. r0-r3 and r12 may
 * be changed freely; r4-r11 must be saved with push/pop if used.
 *
 * GenAI declaration: this header (the interface only) was written with
 * Claude (Anthropic). The implementation in led.s is Shub's own work.
 */
#ifndef LED_H
#define LED_H

#include <stdint.h>

/* Colours as a bit mask: one bit per LED. LED_RED | LED_GREEN = yellow. */
#define LED_OFF   0u
#define LED_RED   (1u << 0)
#define LED_GREEN (1u << 1)
#define LED_BLUE  (1u << 2)

/* Make the three RGB LED pins GPIO outputs, with the LED off. Call once. */
void led_init(void);

/* Show a colour. r0 = bit mask of LED_RED / LED_GREEN / LED_BLUE.
 * A bit that is 0 turns that LED off. */
void led_set(uint32_t color);

#endif /* LED_H */
