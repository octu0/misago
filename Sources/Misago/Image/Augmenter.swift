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
        /// Probability of adding bars, and their maximum fraction of the height (letterbox) or width (pillarbox).
        /// Kept small: the content is squeezed inwards, and wide bars push the edge UI into a center mask.
        public var barProbability: Float = 0.5
        public var maxLetterboxFraction: Float = 0.06
        public var maxPillarboxFraction: Float = 0.08
        /// Number of cutout rectangles, each up to this fraction of the width and height.
        public var cutoutCount: ClosedRange<Int> = 0...3
        public var maxCutoutFraction: Float = 0.35
        public var noiseAmplitude: Float = 0.03
        /// Probability of resolution loss on a strip, and the smallest intermediate scale: the strip is shrunk
        /// (area average) and enlarged back (bilinear), as a crop from a 720p screenshot would look.
        public var degradeProbability: Float = 0.5
        public var minDegradeScale: Float = 0.4
        /// Probability of hiding the middle of the screen behind a mosaic or blur, so the classifier learns
        /// the screen from the edge HUD rather than from whatever character or popup sits in the centre.
        public var centerObscureProbability: Float = 0.35
        /// Probability of overlays a tutorial or popup would add: 1-2 light panels with text-like stripes and
        /// a skin-coloured blob (a pointing hand), placed anywhere.
        public var overlayProbability: Float = 0.35

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

        if unit() < options.centerObscureProbability {
            let cw = Int(Float(w) * uniform(0.35...0.7))
            let chh = Int(Float(h) * uniform(0.35...0.7))
            let x0 = (w - cw) / 2 + Int(Float(w) * uniform(-0.08...0.08))
            let y0 = (h - chh) / 2 + Int(Float(h) * uniform(-0.08...0.08))
            if unit() < 0.5 {
                mosaic(&p, x0: x0, y0: y0, x1: x0 + cw, y1: y0 + chh, block: 3 + Int(unit() * 6))
            } else {
                blur(&p, x0: x0, y0: y0, x1: x0 + cw, y1: y0 + chh, radius: 2 + Int(unit() * 3))
            }
        }

        if unit() < options.overlayProbability {
            let panels = 1 + Int(unit() * 2)
            var k = 0
            while k < panels {
                let pw = Int(Float(w) * uniform(0.15...0.4))
                let ph = Int(Float(h) * uniform(0.1...0.25))
                let px = Int(Float(w - pw) * unit())
                let py = Int(Float(h - ph) * unit())
                let shade = uniform(0.8...0.95)
                fill(&p, x0: px, y0: py, x1: px + pw, y1: py + ph, value: shade)
                // Dark text-like stripes inside the panel.
                var ty = py + max(1, ph / 4)
                while ty < py + ph - 1 {
                    let tx0 = px + max(1, pw / 10)
                    let tx1 = tx0 + Int(Float(pw) * uniform(0.4...0.8))
                    fill(&p, x0: tx0, y0: ty, x1: min(px + pw - 1, tx1), y1: ty + 1, value: uniform(0.0...0.25))
                    ty += max(2, ph / 3)
                }
                k += 1
            }
            if unit() < 0.7 {
                // Pointing hand: a skin-coloured ellipse with a narrower finger above it.
                let r = Float(h) * uniform(0.06...0.12)
                let cx = Float(w) * unit()
                let cy = Float(h) * unit()
                let skin: (Float, Float, Float) = (uniform(0.9...1.0), uniform(0.7...0.85), uniform(0.5...0.7))
                ellipse(&p, cx: cx, cy: cy, rx: r, ry: r * 0.85, color: skin)
                ellipse(&p, cx: cx - r * 0.3, cy: cy - r * 1.2, rx: r * 0.3, ry: r * 0.8, color: skin)
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

    /// Resolution loss, color jitter and noise on a digit strip (no bars or cutouts: the field must stay readable).
    public mutating func augment(_ strip: DigitStrip) -> DigitStrip {
        var p = strip.pixels
        let ch = DigitStrip.channels
        if unit() < options.degradeProbability {
            let scale = uniform(options.minDegradeScale...1.0)
            let smallW = max(8, Int((Float(DigitStrip.width) * scale).rounded()))
            let smallH = max(4, Int((Float(DigitStrip.height) * scale).rounded()))
            let small = Resampler.resample(
                rgb: p, width: DigitStrip.width, height: DigitStrip.height, channels: ch,
                cropX: 0, cropY: 0, cropWidth: Float(DigitStrip.width), cropHeight: Float(DigitStrip.height),
                outWidth: smallW, outHeight: smallH
            )
            p = Resampler.resample(
                rgb: small, width: smallW, height: smallH, channels: ch,
                cropX: 0, cropY: 0, cropWidth: Float(smallW), cropHeight: Float(smallH),
                outWidth: DigitStrip.width, outHeight: DigitStrip.height
            )
        }
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
            var v = ((p[i] - 0.5) * contrast + 0.5 + brightness) * gain[i % ch]
            if 0.0 < options.noiseAmplitude {
                v += uniform(-options.noiseAmplitude...options.noiseAmplitude)
            }
            p[i] = ScreenImage.clamp01(v)
            i += 1
        }
        return (try? DigitStrip(pixels: p)) ?? strip
    }

    /// A jittered version of a crop rectangle: position moved by up to `shift` of the height in each
    /// direction and size scaled by up to ±`scale`, so the reader tolerates imprecise regions.
    public mutating func jitter(x: Int, y: Int, width: Int, height: Int, shift: Float = 0.3, scale: Float = 0.1) -> (x: Int, y: Int, width: Int, height: Int) {
        let dx = uniform(-shift...shift) * Float(height)
        let dy = uniform(-shift...shift) * Float(height)
        let sw = 1.0 + uniform(-scale...scale)
        let sh = 1.0 + uniform(-scale...scale)
        let w = max(1, Int((Float(width) * sw).rounded()))
        let h = max(1, Int((Float(height) * sh).rounded()))
        return (Int((Float(x) + dx).rounded()), Int((Float(y) + dy).rounded()), w, h)
    }

    /// Replaces the rectangle with blocks of their mean colour.
    private func mosaic(_ p: inout [Float], x0: Int, y0: Int, x1: Int, y1: Int, block: Int) {
        let w = ScreenImage.width
        let ch = ScreenImage.channels
        let ax = max(0, x0), bx = min(w, x1), ay = max(0, y0), by = min(ScreenImage.height, y1)
        var by0 = ay
        while by0 < by {
            var bx0 = ax
            while bx0 < bx {
                var sum: [Float] = [0, 0, 0]
                var n: Float = 0
                var y = by0
                while y < min(by, by0 + block) {
                    var x = bx0
                    while x < min(bx, bx0 + block) {
                        var c = 0
                        while c < ch {
                            sum[c] += p[(y * w + x) * ch + c]
                            c += 1
                        }
                        n += 1
                        x += 1
                    }
                    y += 1
                }
                y = by0
                while y < min(by, by0 + block) {
                    var x = bx0
                    while x < min(bx, bx0 + block) {
                        var c = 0
                        while c < ch {
                            p[(y * w + x) * ch + c] = sum[c] / max(1, n)
                            c += 1
                        }
                        x += 1
                    }
                    y += 1
                }
                bx0 += block
            }
            by0 += block
        }
    }

    /// Box blur of the rectangle (reads from a copy so the result does not smear across passes).
    private func blur(_ p: inout [Float], x0: Int, y0: Int, x1: Int, y1: Int, radius: Int) {
        let w = ScreenImage.width
        let h = ScreenImage.height
        let ch = ScreenImage.channels
        let src = p
        var y = max(0, y0)
        while y < min(h, y1) {
            var x = max(0, x0)
            while x < min(w, x1) {
                var sum: [Float] = [0, 0, 0]
                var n: Float = 0
                var yy = max(0, y - radius)
                while yy <= min(h - 1, y + radius) {
                    var xx = max(0, x - radius)
                    while xx <= min(w - 1, x + radius) {
                        var c = 0
                        while c < ch {
                            sum[c] += src[(yy * w + xx) * ch + c]
                            c += 1
                        }
                        n += 1
                        xx += 1
                    }
                    yy += 1
                }
                var c = 0
                while c < ch {
                    p[(y * w + x) * ch + c] = sum[c] / n
                    c += 1
                }
                x += 1
            }
            y += 1
        }
    }

    private func ellipse(_ p: inout [Float], cx: Float, cy: Float, rx: Float, ry: Float, color: (Float, Float, Float)) {
        let w = ScreenImage.width
        let ch = ScreenImage.channels
        var y = max(0, Int(cy - ry))
        while y <= min(ScreenImage.height - 1, Int(cy + ry)) {
            var x = max(0, Int(cx - rx))
            while x <= min(w - 1, Int(cx + rx)) {
                let dx = (Float(x) - cx) / max(rx, 0.5)
                let dy = (Float(y) - cy) / max(ry, 0.5)
                if dx * dx + dy * dy <= 1.0 {
                    let i = (y * w + x) * ch
                    p[i] = color.0
                    p[i + 1] = color.1
                    p[i + 2] = color.2
                }
                x += 1
            }
            y += 1
        }
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
