import XCTest
import Foundation
@testable import ImageDecoding

/// Prints decode times for real files when MISAGO_DECODE_SAMPLES lists paths (colon separated); skipped otherwise.
final class DecodeTimingTests: XCTestCase {
    func testTiming() throws {
        guard let list = ProcessInfo.processInfo.environment["MISAGO_DECODE_SAMPLES"], list.isEmpty != true else {
            throw XCTSkip("set MISAGO_DECODE_SAMPLES to time real files")
        }
        for path in list.split(separator: ":").map(String.init) {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            var best = Double.greatestFiniteMagnitude
            var size = (0, 0)
            var i = 0
            while i < 5 {
                let start = DispatchTime.now().uptimeNanoseconds
                let decoded = try ImageDecoder.decodeRGB(data)
                best = min(best, Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000.0)
                size = (decoded.width, decoded.height)
                i += 1
            }
            print("DECODE \((path as NSString).lastPathComponent) \(size.0)x\(size.1) \(data.count / 1024) KB: \(String(format: "%.1f", best)) ms")
        }
    }
}
