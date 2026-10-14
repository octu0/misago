import Foundation

/// A screenshot reduced to the model input: 128 x 72 RGB, row-major, interleaved channels,
/// values in [0, 1]. `pixels[(y * width + x) * channels + c]`, row 0 is the top of the image.
public struct ScreenImage: Sendable, Equatable {
    public static let width = 128
    public static let height = 72
    public static let channels = 3
    public static let pixelCount = width * height * channels

    public var pixels: [Float]

    /// All black.
    public init() {
        pixels = [Float](repeating: 0.0, count: Self.pixelCount)
    }

    public init(pixels: [Float]) throws {
        if pixels.count != Self.pixelCount {
            throw ScreenImageError.pixelCountMismatch(expected: Self.pixelCount, actual: pixels.count)
        }
        self.pixels = pixels
    }

    /// From 8-bit samples (0...255).
    public init(bytes: [UInt8]) throws {
        if bytes.count != Self.pixelCount {
            throw ScreenImageError.pixelCountMismatch(expected: Self.pixelCount, actual: bytes.count)
        }
        var values = [Float](repeating: 0.0, count: Self.pixelCount)
        var i = 0
        while i < bytes.count {
            values[i] = Float(bytes[i]) * (1.0 / 255.0)
            i += 1
        }
        pixels = values
    }

    /// 8-bit samples, rounded, for the dataset file.
    public var bytes: [UInt8] {
        var out = [UInt8](repeating: 0, count: Self.pixelCount)
        var i = 0
        while i < pixels.count {
            let v = Self.clamp01(pixels[i]) * 255.0
            out[i] = UInt8(v.rounded())
            i += 1
        }
        return out
    }

    @inline(__always)
    public static func clamp01(_ v: Float) -> Float {
        if v < 0.0 {
            return 0.0
        }
        if 1.0 < v {
            return 1.0
        }
        return v
    }
}

public enum ScreenImageError: Error, Equatable {
    case pixelCountMismatch(expected: Int, actual: Int)
    case invalidSize(width: Int, height: Int)
}
