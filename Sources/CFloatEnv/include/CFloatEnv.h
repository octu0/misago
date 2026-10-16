#ifndef CFLOATENV_H
#define CFLOATENV_H

/// Enables flush-to-zero and denormals-are-zero for the calling thread on x86-64 (MXCSR bits FTZ and DAZ).
/// No-op on other architectures. Returns 1 when the mode was set, 0 otherwise.
int misago_enable_flush_to_zero(void);

#endif
