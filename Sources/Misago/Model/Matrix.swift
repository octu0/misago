import Foundation

/// Row-major single-precision matrix product in pure Swift: c = a · b + beta · c, with a [m, k], b [k, n],
/// c [m, n] and arbitrary leading dimensions. The kernel handles 4 rows x 16 columns at a time with SIMD8
/// fused multiply-adds, which the compiler lowers to vector FMAs on arm64 and x86-64; the remainders fall back
/// to 8-wide and scalar loops. This replaces Accelerate so the inference library has no platform dependency.
public enum Matrix {
    @inline(__always)
    static func load(_ p: UnsafePointer<Float>) -> SIMD8<Float> {
        return UnsafeRawPointer(p).loadUnaligned(as: SIMD8<Float>.self)
    }

    @inline(__always)
    static func store(_ v: SIMD8<Float>, _ p: UnsafeMutablePointer<Float>) {
        UnsafeMutableRawPointer(p).storeBytes(of: v, as: SIMD8<Float>.self)
    }

    /// Writes an 8-wide accumulator into c, scaled by beta.
    @inline(__always)
    static func emit(_ acc: SIMD8<Float>, _ p: UnsafeMutablePointer<Float>, _ beta: Float) {
        if beta == 0.0 {
            store(acc, p)
        } else {
            store(acc + load(UnsafePointer(p)) * beta, p)
        }
    }

    public static func gemm(_ a: UnsafePointer<Float>, _ lda: Int, _ b: UnsafePointer<Float>, _ ldb: Int,
                            _ c: UnsafeMutablePointer<Float>, _ ldc: Int, m: Int, n: Int, k: Int, beta: Float) {
        var i = 0
        // Four rows at a time.
        while i + 4 <= m {
            let a0 = a + i * lda
            let a1 = a0 + lda
            let a2 = a1 + lda
            let a3 = a2 + lda
            let c0 = c + i * ldc
            let c1 = c0 + ldc
            let c2 = c1 + ldc
            let c3 = c2 + ldc
            var j = 0
            while j + 16 <= n {
                var s00 = SIMD8<Float>(repeating: 0.0)
                var s01 = SIMD8<Float>(repeating: 0.0)
                var s10 = SIMD8<Float>(repeating: 0.0)
                var s11 = SIMD8<Float>(repeating: 0.0)
                var s20 = SIMD8<Float>(repeating: 0.0)
                var s21 = SIMD8<Float>(repeating: 0.0)
                var s30 = SIMD8<Float>(repeating: 0.0)
                var s31 = SIMD8<Float>(repeating: 0.0)
                var bp = b + j
                var p = 0
                while p < k {
                    let b0 = load(bp)
                    let b1 = load(bp + 8)
                    let v0 = SIMD8<Float>(repeating: a0[p])
                    let v1 = SIMD8<Float>(repeating: a1[p])
                    let v2 = SIMD8<Float>(repeating: a2[p])
                    let v3 = SIMD8<Float>(repeating: a3[p])
                    s00 = s00.addingProduct(v0, b0)
                    s01 = s01.addingProduct(v0, b1)
                    s10 = s10.addingProduct(v1, b0)
                    s11 = s11.addingProduct(v1, b1)
                    s20 = s20.addingProduct(v2, b0)
                    s21 = s21.addingProduct(v2, b1)
                    s30 = s30.addingProduct(v3, b0)
                    s31 = s31.addingProduct(v3, b1)
                    bp += ldb
                    p += 1
                }
                emit(s00, c0 + j, beta)
                emit(s01, c0 + j + 8, beta)
                emit(s10, c1 + j, beta)
                emit(s11, c1 + j + 8, beta)
                emit(s20, c2 + j, beta)
                emit(s21, c2 + j + 8, beta)
                emit(s30, c3 + j, beta)
                emit(s31, c3 + j + 8, beta)
                j += 16
            }
            while j + 8 <= n {
                var s0 = SIMD8<Float>(repeating: 0.0)
                var s1 = SIMD8<Float>(repeating: 0.0)
                var s2 = SIMD8<Float>(repeating: 0.0)
                var s3 = SIMD8<Float>(repeating: 0.0)
                var bp = b + j
                var p = 0
                while p < k {
                    let b0 = load(bp)
                    s0 = s0.addingProduct(SIMD8<Float>(repeating: a0[p]), b0)
                    s1 = s1.addingProduct(SIMD8<Float>(repeating: a1[p]), b0)
                    s2 = s2.addingProduct(SIMD8<Float>(repeating: a2[p]), b0)
                    s3 = s3.addingProduct(SIMD8<Float>(repeating: a3[p]), b0)
                    bp += ldb
                    p += 1
                }
                emit(s0, c0 + j, beta)
                emit(s1, c1 + j, beta)
                emit(s2, c2 + j, beta)
                emit(s3, c3 + j, beta)
                j += 8
            }
            while j < n {
                var s0: Float = 0.0
                var s1: Float = 0.0
                var s2: Float = 0.0
                var s3: Float = 0.0
                var p = 0
                while p < k {
                    let bv = b[p * ldb + j]
                    s0 += a0[p] * bv
                    s1 += a1[p] * bv
                    s2 += a2[p] * bv
                    s3 += a3[p] * bv
                    p += 1
                }
                c0[j] = s0 + (beta == 0.0 ? 0.0 : c0[j] * beta)
                c1[j] = s1 + (beta == 0.0 ? 0.0 : c1[j] * beta)
                c2[j] = s2 + (beta == 0.0 ? 0.0 : c2[j] * beta)
                c3[j] = s3 + (beta == 0.0 ? 0.0 : c3[j] * beta)
                j += 1
            }
            i += 4
        }
        // Remaining rows one at a time.
        while i < m {
            let a0 = a + i * lda
            let c0 = c + i * ldc
            var j = 0
            while j + 8 <= n {
                var s0 = SIMD8<Float>(repeating: 0.0)
                var bp = b + j
                var p = 0
                while p < k {
                    s0 = s0.addingProduct(SIMD8<Float>(repeating: a0[p]), load(bp))
                    bp += ldb
                    p += 1
                }
                emit(s0, c0 + j, beta)
                j += 8
            }
            while j < n {
                var s0: Float = 0.0
                var p = 0
                while p < k {
                    s0 += a0[p] * b[p * ldb + j]
                    p += 1
                }
                c0[j] = s0 + (beta == 0.0 ? 0.0 : c0[j] * beta)
                j += 1
            }
            i += 1
        }
    }
}
