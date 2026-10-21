import Foundation

/// Turns an RGB bitmap of any size into a `ScreenImage`.
///
/// The whole input is resampled to 128 x 72 regardless of its aspect ratio (area averaging when shrinking,
/// bilinear when enlarging). Game HUDs are anchored to the screen edges, so stretching a 4:3 tablet or a
/// 19.5:9 phone screenshot puts the same elements in the same places, and no device adds black bars that the
/// classifier would have to learn to ignore.
public enum ScreenNormalizer {
    /// `rgb` is interleaved RGB, row-major, `width * height * 3` samples (0...255), row 0 at the top.
    public static func normalize(width: Int, height: Int, rgb: [UInt8]) throws -> ScreenImage {
        try validate(width: width, height: height, count: rgb.count)
        return fit(width: width, height: height, source: BytePixels(rgb), cropX: 0, cropY: 0, cropWidth: width, cropHeight: height)
    }

    /// The input turned clockwise by `degrees` (a multiple of 90), at the cost of turning the 128 x 72 result:
    /// for 90 and 270 the input is resampled to 72 x 128 first, then that small image is turned.
    public static func normalize(width: Int, height: Int, rgb: [UInt8], rotation degrees: Int) throws -> ScreenImage {
        try validate(width: width, height: height, count: rgb.count)
        let quarter = ((degrees / 90) % 2 + 2) % 2 == 1
        let outW = quarter ? ScreenImage.height : ScreenImage.width
        let outH = quarter ? ScreenImage.width : ScreenImage.height
        let small = Resampler.resample(
            source: BytePixels(rgb), width: width, height: height, channels: ScreenImage.channels,
            cropX: 0, cropY: 0, cropWidth: Float(width), cropHeight: Float(height),
            outWidth: outW, outHeight: outH
        )
        let turned = ImageRotation.rotate(width: outW, height: outH, pixels: small, degrees: degrees)
        return (try? ScreenImage(pixels: turned.pixels)) ?? ScreenImage()
    }

    /// Normalizes a sub-rectangle of the input (in input pixels, clipped to the image).
    public static func normalize(width: Int, height: Int, rgb: [UInt8], cropX: Int, cropY: Int, cropWidth: Int, cropHeight: Int) throws -> ScreenImage {
        try validate(width: width, height: height, count: rgb.count)
        let x0 = min(max(0, cropX), width - 1)
        let y0 = min(max(0, cropY), height - 1)
        let x1 = min(width, max(x0 + 1, cropX + cropWidth))
        let y1 = min(height, max(y0 + 1, cropY + cropHeight))
        // Copy the crop so nothing outside it can bleed into the canvas, then fit that smaller image.
        let ch = ScreenImage.channels
        let cw = x1 - x0
        let chh = y1 - y0
        var crop = [UInt8](repeating: 0, count: cw * chh * ch)
        var y = 0
        while y < chh {
            let src = ((y0 + y) * width + x0) * ch
            crop.replaceSubrange((y * cw * ch)..<((y + 1) * cw * ch), with: rgb[src..<(src + cw * ch)])
            y += 1
        }
        return fit(width: cw, height: chh, source: BytePixels(crop), cropX: 0, cropY: 0, cropWidth: cw, cropHeight: chh)
    }

    /// Float variant, samples in [0, 1].
    public static func normalize(width: Int, height: Int, rgb: [Float]) throws -> ScreenImage {
        try validate(width: width, height: height, count: rgb.count)
        return fit(width: width, height: height, source: FloatPixels(rgb), cropX: 0, cropY: 0, cropWidth: width, cropHeight: height)
    }

    private static func validate(width: Int, height: Int, count: Int) throws {
        if width <= 0 || height <= 0 {
            throw ScreenImageError.invalidSize(width: width, height: height)
        }
        let expected = width * height * ScreenImage.channels
        if count != expected {
            throw ScreenImageError.pixelCountMismatch(expected: expected, actual: count)
        }
    }

    /// Resamples the crop (here always the whole source image) onto the 128 x 72 grid.
    private static func fit<S: PixelSource>(width: Int, height: Int, source: S, cropX: Int, cropY: Int, cropWidth: Int, cropHeight: Int) -> ScreenImage {
        let out = Resampler.resample(
            source: source, width: width, height: height, channels: ScreenImage.channels,
            cropX: Float(cropX), cropY: Float(cropY), cropWidth: Float(cropWidth), cropHeight: Float(cropHeight),
            outWidth: ScreenImage.width, outHeight: ScreenImage.height
        )
        return (try? ScreenImage(pixels: out)) ?? ScreenImage()
    }
}
