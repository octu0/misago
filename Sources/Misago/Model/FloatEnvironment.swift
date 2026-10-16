import CFloatEnv

/// Per-thread floating-point mode for the inference kernels.
public enum FloatEnvironment {
    /// Flushes denormals to zero on x86-64 (a per-thread setting, so it is called at the start of every
    /// inference). On Apple Silicon denormals are free and the call does nothing.
    @inline(__always)
    public static func prepareForInference() {
        _ = misago_enable_flush_to_zero()
    }
}
