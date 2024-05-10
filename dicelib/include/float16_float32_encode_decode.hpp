//reference: https://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion

#ifndef FLOAT16_FLOAT32_ENCODE_DECODE_H
#define FLOAT16_FLOAT32_ENCODE_DECODE_H

#include <iostream>

unsigned int as_uint(const float value) {
    return *(unsigned int*) & value;
}

float as_float(const unsigned int value) {
    return *(float*) & value;
}

float float16_to_float32(const unsigned short value) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits

    const unsigned int e = (value & 0x7C00)>>10; // exponent
    const unsigned int m = (value & 0x03FF)<<13; // mantissa
    const unsigned int v = as_uint( (float) m ) >> 23; // evil log2 bit hack to count leading zeros in denormalized format

    return as_float( ( value & 0x8000) << 16 | ( e!= 0) * ((e+112) << 23 | m) |
        ((e==0) & (m!=0)) * ( (v-37) << 23 | ( (m<<(150-v)) & 0x007FE000))); // sign : normalized : denormalized
}

unsigned short float32_to_float16(const float value) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits

    const unsigned int b = as_uint(value) + 0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
    const unsigned int e = (b & 0x7F800000) >> 23; // exponent
    const unsigned int m = b & 0x007FFFFF; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding

    return (b&0x80000000) >> 16 | (e > 112) *((( (e-112) << 10 ) & 0x7C00) | m >> 13) |
        ((e<113) & (e>101)) * (((( 0x007FF000 + m) >> (125-e)) + 1) >> 1) | (e>143) * 0x7FFF; // sign : normalized : denormalized : saturate
}

#endif