# Sum of the min & max of (a, b, c)
import torch


def hilo(a, b, c):
    if c < b: b, c = c, b
    if b < a: a, b = b, a
    if c < b: b, c = c, b
    return a + c

def complement(pixel_RGB):
    r, g, b = pixel_RGB
    k = hilo(r, g, b)
    return tuple(k - u for u in (r, g, b))
