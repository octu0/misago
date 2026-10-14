import Foundation

/// Turns an RGB bitmap of any size into a `ScreenImage`.
///
/// The input is fitted into a 16:9 canvas with black bars (centered) so that nothing is cropped and the
/// aspect ratio is kept, then resampled to 128 x 72: area averaging when shrinking, bilinear when enlarging.
/// A 16:9 input fills the canvas exactly; letterboxed or pillarboxed inputs keep their bars.
public enum ScreenNormalizer {
    /// `rgb` is interleaved RGB, row-major, `width * height * 3` samples (0...255), row 0 at the top.
    public static func normalize(width: Int, height: Int, rgb: [UInt8]) throws -> ScreenImage {
        if width <= 0 || height <= 0 {
            throw ScreenImageError.invalidSize(width: width, height: height)
        }
        let expected = width * height * ScreenImage.channels
        if rgb.count != expected {
            throw ScreenImageError.pixelCountMismatch(expected: expected, actual: rgb.count)
        }
        var values = [Float](repeating: 0.0, count: expected)
        var i = 0
        while i < expected {
            values[i] = Float(rgb[i]) * (1.0 / 255.0)
            i += 1
        }
        return resample(width: width, height: height, rgb: values)
    }

    /// Float variant, samples in [0, 1].
    public static func normalize(width: Int, height: Int, rgb: [Float]) throws -> ScreenImage {
        if width <= 0 || height <= 0 {
            throw ScreenImageError.invalidSize(width: width, height: height)
        }
        let expected = width * height * ScreenImage.channels
        if rgb.count != expected {
            throw ScreenImageError.pixelCountMismatch(expected: expected, actual: rgb.count)
        }
        return resample(width: width, height: height, rgb: rgb)
    }

    // MARK: - Internals

    /// Canvas: the smallest 16:9 rectangle (in input pixels) containing the input, centered on it.
    /// The outside of the input counts as black.
    private static func resample(width: Int, height: Int, rgb: [Float]) -> ScreenImage {
        let outW = ScreenImage.width
        let outH = ScreenImage.height
        let targetAspect = Float(outW) / Float(outH)
        var canvasW = Float(width)
        var canvasH = Float(height)
        if canvasW / canvasH < targetAspect {
            canvasW = canvasH * targetAspect
        } else {
            canvasH = canvasW / targetAspect
        }
        let originX = (Float(width) - canvasW) * 0.5
        let originY = (Float(height) - canvasH) * 0.5
        let scaleX = canvasW / Float(outW)
        let scaleY = canvasH / Float(outH)

        var out = [Float](repeating: 0.0, count: ScreenImage.pixelCount)
        if 1.0 < scaleX && 1.0 < scaleY {
            areaAverage(rgb: rgb, width: width, height: height, originX: originX, originY: originY, scaleX: scaleX, scaleY: scaleY, into: &out)
        } else {
            bilinear(rgb: rgb, width: width, height: height, originX: originX, originY: originY, scaleX: scaleX, scaleY: scaleY, into: &out)
        }
        return (try? ScreenImage(pixels: out)) ?? ScreenImage()
    }

    /// Shrinking: each output pixel is the mean of the input rectangle it covers, with partial pixels weighted
    /// by their overlap. Area outside the input is black and still counts toward the mean.
    private static func areaAverage(
        rgb: [Float], width: Int, height: Int,
        originX: Float, originY: Float, scaleX: Float, scaleY: Float,
        into out: inout [Float]
    ) {
        let outW = ScreenImage.width
        let outH = ScreenImage.height
        let ch = ScreenImage.channels
        let cellArea = scaleX * scaleY
        var oy = 0
        while oy < outH {
            let y0 = originY + Float(oy) * scaleY
            let y1 = y0 + scaleY
            let iy0 = max(0, Int(y0.rounded(.down)))
            let iy1 = min(height, Int(y1.rounded(.up)))
            var ox = 0
            while ox < outW {
                let x0 = originX + Float(ox) * scaleX
                let x1 = x0 + scaleX
                let ix0 = max(0, Int(x0.rounded(.down)))
                let ix1 = min(width, Int(x1.rounded(.up)))
                var r: Float = 0.0
                var g: Float = 0.0
                var b: Float = 0.0
                var iy = iy0
                while iy < iy1 {
                    let wy = min(y1, Float(iy + 1)) - max(y0, Float(iy))
                    if wy <= 0.0 {
                        iy += 1
                        continue
                    }
                    var ix = ix0
                    while ix < ix1 {
                        let wx = min(x1, Float(ix + 1)) - max(x0, Float(ix))
                        if 0.0 < wx {
                            let w = wx * wy
                            let base = (iy * width + ix) * ch
                            r += rgb[base] * w
                            g += rgb[base + 1] * w
                            b += rgb[base + 2] * w
                        }
                        ix += 1
                    }
                    iy += 1
                }
                let o = (oy * outW + ox) * ch
                out[o] = ScreenImage.clamp01(r / cellArea)
                out[o + 1] = ScreenImage.clamp01(g / cellArea)
                out[o + 2] = ScreenImage.clamp01(b / cellArea)
                ox += 1
            }
            oy += 1
        }
    }

    /// Enlarging (or mixed): bilinear interpolation at each output pixel center. An output pixel whose center
    /// lies outside the input is black (the bars); inside, samples are clamped to the input edge so solid
    /// colors stay solid up to the border.
    private static func bilinear(
        rgb: [Float], width: Int, height: Int,
        originX: Float, originY: Float, scaleX: Float, scaleY: Float,
        into out: inout [Float]
    ) {
        let outW = ScreenImage.width
        let outH = ScreenImage.height
        let ch = ScreenImage.channels
        var oy = 0
        while oy < outH {
            let cy = originY + (Float(oy) + 0.5) * scaleY
            if cy < 0.0 || Float(height) <= cy {
                oy += 1
                continue
            }
            let sy = min(max(cy - 0.5, 0.0), Float(height - 1))
            let fy = sy.rounded(.down)
            let ty = sy - fy
            let iy0 = Int(fy)
            let iy1 = min(height - 1, iy0 + 1)
            var ox = 0
            while ox < outW {
                let cx = originX + (Float(ox) + 0.5) * scaleX
                if cx < 0.0 || Float(width) <= cx {
                    ox += 1
                    continue
                }
                let sx = min(max(cx - 0.5, 0.0), Float(width - 1))
                let fx = sx.rounded(.down)
                let tx = sx - fx
                let ix0 = Int(fx)
                let ix1 = min(width - 1, ix0 + 1)
                var c = 0
                while c < ch {
                    let top = rgb[(iy0 * width + ix0) * ch + c] * (1.0 - tx) + rgb[(iy0 * width + ix1) * ch + c] * tx
                    let bottom = rgb[(iy1 * width + ix0) * ch + c] * (1.0 - tx) + rgb[(iy1 * width + ix1) * ch + c] * tx
                    out[(oy * outW + ox) * ch + c] = ScreenImage.clamp01(top * (1.0 - ty) + bottom * ty)
                    c += 1
                }
                ox += 1
            }
            oy += 1
        }
    }
}
