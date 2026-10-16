import Foundation

/// Turns an RGB bitmap of any size into a `ScreenImage`.
///
/// The input is fitted into a 16:9 canvas with black bars (centered) so that nothing is cropped and the
/// aspect ratio is kept, then resampled to 128 x 72: area averaging when shrinking, bilinear when enlarging.
/// A 16:9 input fills the canvas exactly; letterboxed or pillarboxed inputs keep their bars.
public enum ScreenNormalizer {
    /// `rgb` is interleaved RGB, row-major, `width * height * 3` samples (0...255), row 0 at the top.
    public static func normalize(width: Int, height: Int, rgb: [UInt8]) throws -> ScreenImage {
        try validate(width: width, height: height, count: rgb.count)
        return fit(width: width, height: height, source: BytePixels(rgb), cropX: 0, cropY: 0, cropWidth: width, cropHeight: height)
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

    /// Canvas: the smallest 16:9 rectangle (in input pixels) containing the crop, centered on it. The crop here
    /// is always the whole source image, so everything outside it is black by construction.
    private static func fit<S: PixelSource>(width: Int, height: Int, source: S, cropX: Int, cropY: Int, cropWidth: Int, cropHeight: Int) -> ScreenImage {
        let targetAspect = Float(ScreenImage.width) / Float(ScreenImage.height)
        var canvasW = Float(cropWidth)
        var canvasH = Float(cropHeight)
        if canvasW / canvasH < targetAspect {
            canvasW = canvasH * targetAspect
        } else {
            canvasH = canvasW / targetAspect
        }
        let originX = Float(cropX) + (Float(cropWidth) - canvasW) * 0.5
        let originY = Float(cropY) + (Float(cropHeight) - canvasH) * 0.5
        let out = Resampler.resample(
            source: source, width: width, height: height, channels: ScreenImage.channels,
            cropX: originX, cropY: originY, cropWidth: canvasW, cropHeight: canvasH,
            outWidth: ScreenImage.width, outHeight: ScreenImage.height
        )
        return (try? ScreenImage(pixels: out)) ?? ScreenImage()
    }
}
