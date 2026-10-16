import Foundation

/// A cropped number field reduced to the reader's input: 320 x 56 RGB, same layout as `ScreenImage`.
/// The crop is scaled to the strip height keeping its aspect ratio, placed at the left, and the rest is black.
public struct DigitStrip: Sendable, Equatable {
    public static let width = 320
    public static let height = 56
    public static let channels = 3
    public static let pixelCount = width * height * channels

    public var pixels: [Float]

    public init() {
        pixels = [Float](repeating: 0.0, count: Self.pixelCount)
    }

    public init(pixels: [Float]) throws {
        if pixels.count != Self.pixelCount {
            throw ScreenImageError.pixelCountMismatch(expected: Self.pixelCount, actual: pixels.count)
        }
        self.pixels = pixels
    }

    public init(bytes: [UInt8]) throws {
        if bytes.count != Self.pixelCount {
            throw ScreenImageError.pixelCountMismatch(expected: Self.pixelCount, actual: bytes.count)
        }
        pixels = Resampler.floats(bytes)
    }

    public var bytes: [UInt8] {
        var out = [UInt8](repeating: 0, count: Self.pixelCount)
        var i = 0
        while i < pixels.count {
            out[i] = UInt8((Resampler.clamp01(pixels[i]) * 255.0).rounded())
            i += 1
        }
        return out
    }
}

public enum StripNormalizer {
    /// Cuts `rect` (input pixels) out of the bitmap and scales it to the strip height. Crops wider than the
    /// strip are scaled down to the strip width instead (then shorter than the strip height, bottom padded).
    public static func extract(width: Int, height: Int, rgb: [UInt8], x: Int, y: Int, cropWidth: Int, cropHeight: Int) throws -> DigitStrip {
        if width <= 0 || height <= 0 {
            throw ScreenImageError.invalidSize(width: width, height: height)
        }
        let expected = width * height * DigitStrip.channels
        if rgb.count != expected {
            throw ScreenImageError.pixelCountMismatch(expected: expected, actual: rgb.count)
        }
        if cropWidth <= 0 || cropHeight <= 0 {
            throw ScreenImageError.invalidSize(width: cropWidth, height: cropHeight)
        }
        // Scale so the crop height becomes the strip height; if that overflows the width, scale by width.
        var scale = Float(DigitStrip.height) / Float(cropHeight)
        if Float(DigitStrip.width) < Float(cropWidth) * scale {
            scale = Float(DigitStrip.width) / Float(cropWidth)
        }
        let outW = max(1, min(DigitStrip.width, Int((Float(cropWidth) * scale).rounded())))
        let outH = max(1, min(DigitStrip.height, Int((Float(cropHeight) * scale).rounded())))
        // Copy the crop (clipped to the image) so nothing outside it can bleed in, resample it into outW x outH,
        // then paste at the top-left of the strip.
        let ch = DigitStrip.channels
        let x0 = max(0, x)
        let y0 = max(0, y)
        let x1 = min(width, x + cropWidth)
        let y1 = min(height, y + cropHeight)
        let cw = max(0, x1 - x0)
        let chh = max(0, y1 - y0)
        var crop = [UInt8](repeating: 0, count: cw * chh * ch)
        var yy = 0
        while yy < chh {
            let src = ((y0 + yy) * width + x0) * ch
            crop.replaceSubrange((yy * cw * ch)..<((yy + 1) * cw * ch), with: rgb[src..<(src + cw * ch)])
            yy += 1
        }
        // The requested rectangle may extend past the image; keep its geometry relative to the copied part.
        let small = Resampler.resample(
            bytes: crop, width: cw, height: chh, channels: ch,
            cropX: Float(x - x0), cropY: Float(y - y0), cropWidth: Float(cropWidth), cropHeight: Float(cropHeight),
            outWidth: outW, outHeight: outH
        )
        var strip = DigitStrip()
        var oy = 0
        while oy < outH {
            let src = oy * outW * ch
            let dst = oy * DigitStrip.width * ch
            strip.pixels.replaceSubrange(dst..<(dst + outW * ch), with: small[src..<(src + outW * ch)])
            oy += 1
        }
        return strip
    }

    public static func extract(width: Int, height: Int, rgb: [UInt8], region: CropRegion) throws -> DigitStrip {
        let r = region.pixelRect(width: width, height: height)
        return try extract(width: width, height: height, rgb: rgb, x: r.x, y: r.y, cropWidth: r.width, cropHeight: r.height)
    }
}
