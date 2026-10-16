#include "CFloatEnv.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <xmmintrin.h>
int misago_enable_flush_to_zero(void) {
    // FTZ (bit 15): results that would be denormal become zero. DAZ (bit 6): denormal inputs read as zero.
    // Without these, x86 handles every denormal with a microcode assist that costs around a hundred cycles,
    // and the GeLU and decaying recurrence of the model produce denormals constantly.
    unsigned int csr = _mm_getcsr();
    _mm_setcsr(csr | 0x8040u);
    return 1;
}
#else
int misago_enable_flush_to_zero(void) {
    return 0;
}
#endif
