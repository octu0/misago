import Foundation

/// Deterministic training-time augmentation for screenshots.
///
/// Two stages: `cropRect` picks a slightly zoomed and shifted window of the full-resolution image before
/// normalization (sub-pixel realistic shifts), and `augment` works on the 128 x 72 image: brightness,
/// contrast and per-channel color jitter, random black bars (letterbox / pillarbox of other devices),
/// cutout rectangles (streaming overlays and avatars hiding parts of the screen) and light noise.
/// The same seed always produces the same augmentation.
public struct Augmenter {
    public struct Options: Sendable, Equatable {
        /// Minimum fraction of the width and height kept by the crop (1.0 disables cropping).
        public var minCropScale: Float = 0.85
        public var maxBrightnessShift: Float = 0.2
        public var contrastRange: ClosedRange<Float> = 0.7...1.3
        public var maxChannelGain: Float = 0.2
        /// Probability of adding bars, and their maximum fraction of the height (letterbox: a 19.5:9 phone in a
        /// 16:9 canvas has 9% bars) or width (pillarbox: a 4:3 tablet has 25% bars).
        public var barProbability: Float = 0.5
        public var maxLetterboxFraction: Float = 0.12
        public var maxPillarboxFraction: Float = 0.25
        /// Number of cutout rectangles, each up to this fraction of the width and height.
        public var cutoutCount: ClosedRange<Int> = 0...3
        public var maxCutoutFraction: Float = 0.35
        public var noiseAmplitude: Float = 0.03

        public init() {}
    }

    private var rng: SplitMix64
    public let options: Options

    public init(seed: UInt64, options: Options = Options()) {
        self.rng = SplitMix64(seed: seed)
        self.options = options
    }

    /// Uniform in [0, 1).
    private mutating func unit() -> Float {
        return Float(rng.next() >> 40) / Float(1 << 24)
    }

    private mutating func uniform(_ range: ClosedRange<Float>) -> Float {
        return range.lowerBound + (range.upperBound - range.lowerBound) * unit()
    }

    /// A crop window in input pixels: a random scale in [minCropScale, 1] of each side at a random position.
    public mutating func cropRect(width: Int, height: Int) -> (x: Int, y: Int, width: Int, height: Int) {
        let scaleX = uniform(options.minCropScale...1.0)
        let scaleY = uniform(options.minCropScale...1.0)
        let w = max(1, Int((Float(width) * scaleX).rounded()))
        let h = max(1, Int((Float(height) * scaleY).rounded()))
        let x = Int((Float(width - w) * unit()).rounded(.down))
        let y = Int((Float(height - h) * unit()).rounded(.down))
        return (x, y, w, h)
    }

    /// Color, bars, cutout and noise on the normalized image.
    public mutating func augment(_ image: ScreenImage) -> ScreenImage {
        let w = ScreenImage.width
        let h = ScreenImage.height
        let ch = ScreenImage.channels
        var p = image.pixels

        let brightness = uniform(-options.maxBrightnessShift...options.maxBrightnessShift)
        let contrast = uniform(options.contrastRange)
        var gain = [Float](repeating: 1.0, count: ch)
        var c = 0
        while c < ch {
            gain[c] = 1.0 + uniform(-options.maxChannelGain...options.maxChannelGain)
            c += 1
        }
        var i = 0
        while i < p.count {
            let v = (p[i] - 0.5) * contrast + 0.5 + brightness
            p[i] = v * gain[i % ch]
            i += 1
        }

        if unit() < options.barProbability {
            // Bars replace the edges of the picture: the content is squeezed into the remaining area, as a
            // narrower or wider device would be after fitting into the 16:9 canvas.
            if unit() < 0.5 {
                let rows = Int(Float(h) * uniform(0.0...options.maxLetterboxFraction))
                p = squeeze(p, left: 0, right: 0, top: rows, bottom: rows)
            } else {
                let cols = Int(Float(w) * uniform(0.0...options.maxPillarboxFraction))
                p = squeeze(p, left: cols, right: cols, top: 0, bottom: 0)
            }
        }

        let cutouts = options.cutoutCount.lowerBound + Int(unit() * Float(options.cutoutCount.upperBound - options.cutoutCount.lowerBound + 1))
        var k = 0
        while k < min(cutouts, options.cutoutCount.upperBound) {
            let cw = max(1, Int(Float(w) * uniform(0.05...options.maxCutoutFraction)))
            let chh = max(1, Int(Float(h) * uniform(0.05...options.maxCutoutFraction)))
            let x0 = Int(Float(w - cw) * unit())
            let y0 = Int(Float(h - chh) * unit())
            fill(&p, x0: x0, y0: y0, x1: x0 + cw, y1: y0 + chh, value: unit())
            k += 1
        }

        if 0.0 < options.noiseAmplitude {
            i = 0
            while i < p.count {
                p[i] += uniform(-options.noiseAmplitude...options.noiseAmplitude)
                i += 1
            }
        }

        i = 0
        while i < p.count {
            p[i] = ScreenImage.clamp01(p[i])
            i += 1
        }
        return (try? ScreenImage(pixels: p)) ?? image
    }

    /// Resamples the picture (bilinear) into the rectangle left after the bars and fills the bars with black.
    private func squeeze(_ p: [Float], left: Int, right: Int, top: Int, bottom: Int) -> [Float] {
        let w = ScreenImage.width
        let h = ScreenImage.height
        let ch = ScreenImage.channels
        let innerW = w - left - right
        let innerH = h - top - bottom
        if innerW <= 0 || innerH <= 0 || (left == 0 && right == 0 && top == 0 && bottom == 0) {
            return p
        }
        var out = [Float](repeating: 0.0, count: p.count)
        var y = 0
        while y < innerH {
            let sy = min(Float(h - 1), (Float(y) + 0.5) * Float(h) / Float(innerH) - 0.5)
            let iy0 = max(0, Int(sy.rounded(.down)))
            let iy1 = min(h - 1, iy0 + 1)
            let ty = max(0.0, sy - Float(iy0))
            var x = 0
            while x < innerW {
                let sx = min(Float(w - 1), (Float(x) + 0.5) * Float(w) / Float(innerW) - 0.5)
                let ix0 = max(0, Int(sx.rounded(.down)))
                let ix1 = min(w - 1, ix0 + 1)
                let tx = max(0.0, sx - Float(ix0))
                var c = 0
                while c < ch {
                    let a = p[(iy0 * w + ix0) * ch + c] * (1.0 - tx) + p[(iy0 * w + ix1) * ch + c] * tx
                    let b = p[(iy1 * w + ix0) * ch + c] * (1.0 - tx) + p[(iy1 * w + ix1) * ch + c] * tx
                    out[((top + y) * w + left + x) * ch + c] = a * (1.0 - ty) + b * ty
                    c += 1
                }
                x += 1
            }
            y += 1
        }
        return out
    }

    private func fill(_ p: inout [Float], x0: Int, y0: Int, x1: Int, y1: Int, value: Float) {
        let w = ScreenImage.width
        let ch = ScreenImage.channels
        var y = max(0, y0)
        while y < min(ScreenImage.height, y1) {
            var x = max(0, x0)
            while x < min(w, x1) {
                var c = 0
                while c < ch {
                    p[(y * w + x) * ch + c] = value
                    c += 1
                }
                x += 1
            }
            y += 1
        }
    }
}

/// Seeded generator for reproducible augmentation.
public struct SplitMix64: RandomNumberGenerator, Sendable {
    public var state: UInt64

    public init(seed: UInt64) {
        self.state = seed
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
