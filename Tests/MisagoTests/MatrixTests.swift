import XCTest
@testable import Misago

final class MatrixTests: XCTestCase {
    private func reference(_ a: [Float], _ b: [Float], m: Int, n: Int, k: Int) -> [Float] {
        var c = [Float](repeating: 0.0, count: m * n)
        var i = 0
        while i < m {
            var j = 0
            while j < n {
                var s: Float = 0.0
                var p = 0
                while p < k {
                    s += a[i * k + p] * b[p * n + j]
                    p += 1
                }
                c[i * n + j] = s
                j += 1
            }
            i += 1
        }
        return c
    }

    /// Odd shapes exercise the 16-wide, 8-wide and scalar paths and the row remainder.
    func testMatchesReferenceOnOddShapes() {
        for (m, n, k) in [(1, 1, 1), (5, 3, 7), (9, 24, 13), (4, 16, 1), (7, 37, 29), (36, 128, 768), (3, 11, 64)] {
            let a = (0..<(m * k)).map { Float(($0 * 7) % 13) / 13.0 - 0.5 }
            let b = (0..<(k * n)).map { Float(($0 * 11) % 17) / 17.0 - 0.5 }
            var c = [Float](repeating: 2.0, count: m * n)
            HawkNetwork.gemm(a, b, &c, m: m, n: n, k: k)
            let want = reference(a, b, m: m, n: n, k: k)
            var worst: Float = 0.0
            var i = 0
            while i < c.count {
                worst = max(worst, abs(c[i] - want[i]))
                i += 1
            }
            XCTAssertLessThan(worst, 1e-3, "shape \(m)x\(n)x\(k)")
            // beta accumulates into the existing contents.
            var c2 = [Float](repeating: 2.0, count: m * n)
            HawkNetwork.gemm(a, b, &c2, m: m, n: n, k: k, beta: 1.0)
            i = 0
            worst = 0.0
            while i < c2.count {
                worst = max(worst, abs(c2[i] - (want[i] + 2.0)))
                i += 1
            }
            XCTAssertLessThan(worst, 1e-3, "beta shape \(m)x\(n)x\(k)")
        }
    }

    /// Sub-matrix views through leading dimensions (as used for the block-diagonal gates).
    func testLeadingDimensions() {
        let m = 5, n = 8, k = 8
        let lda = 24, ldb = 24, ldc = 24
        let a = (0..<(m * lda)).map { Float($0 % 9) * 0.1 }
        let b = (0..<(k * ldb)).map { Float($0 % 5) * 0.2 }
        var c = [Float](repeating: 0.0, count: m * ldc)
        a.withUnsafeBufferPointer { ap in
            b.withUnsafeBufferPointer { bp in
                c.withUnsafeMutableBufferPointer { cp in
                    Matrix.gemm(ap.baseAddress! + 8, lda, bp.baseAddress! + 8, ldb, cp.baseAddress! + 8, ldc, m: m, n: n, k: k, beta: 0.0)
                }
            }
        }
        var i = 0
        while i < m {
            var j = 0
            while j < n {
                var s: Float = 0.0
                var p = 0
                while p < k {
                    s += a[i * lda + 8 + p] * b[p * ldb + 8 + j]
                    p += 1
                }
                XCTAssertEqual(c[i * ldc + 8 + j], s, accuracy: 1e-4)
                j += 1
            }
            XCTAssertEqual(c[i * ldc], 0.0)
            i += 1
        }
    }
}
