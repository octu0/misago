import Foundation
import CoreGraphics
import CoreText
import Misago

setbuf(stdout, nil)

// Renders synthetic number fields for pretraining the digit reader.
//
// Usage: synth --font <ttf> ... --background <dir> ... --count <N> --out <out.msgd> [--seed 1]
// Each strip gets a random background (a crop of a real screenshot, or a flat color), a random number in a
// game-like bold font with a dark outline, and optional distractors: a name-like word and an icon blob next to
// the number. The label is the number. Fonts and backgrounds are tool inputs; nothing game-specific is coded.

var fontPaths: [String] = []
var backgroundDirs: [String] = []
var count = 10000
var outPath = ""
var seed: UInt64 = 1
var numWorkers = 8

var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    let arg = args[argIdx]
    func value() -> String? {
        if (argIdx + 1) < args.count {
            argIdx += 1
            return args[argIdx]
        }
        return nil
    }
    switch arg {
    case "-h", "--help":
        print("Usage: synth --font <ttf> ... --background <dir> ... --count <N> --out <out.msgd> [--seed 1] [-p 8]")
        exit(0)
    case "--font":
        if let v = value() { fontPaths.append(v) }
    case "--background":
        if let v = value() { backgroundDirs.append(v) }
    case "--count":
        if let v = value(), let n = Int(v) { count = max(1, n) }
    case "--out":
        if let v = value() { outPath = v }
    case "--seed":
        if let v = value(), let n = UInt64(v) { seed = n }
    case "-p", "--parallel":
        if let v = value(), let n = Int(v) { numWorkers = max(1, n) }
    default:
        print("error: unknown argument \(arg)")
        exit(1)
    }
    argIdx += 1
}
if fontPaths.isEmpty || outPath.isEmpty {
    print("error: specify at least one --font and --out. See synth --help.")
    exit(1)
}

// Fonts.
var fonts: [CTFont] = []
for path in fontPaths {
    guard let descriptors = CTFontManagerCreateFontDescriptorsFromURL(URL(fileURLWithPath: path) as CFURL) as? [CTFontDescriptor], let first = descriptors.first else {
        print("error: cannot load font \(path)")
        exit(1)
    }
    fonts.append(CTFontCreateWithFontDescriptor(first, 40, nil))
}
print("fonts: \(fonts.count)")

// Backgrounds: decoded lazily and cached (whole images), sampled by random crops.
final class BackgroundBank: @unchecked Sendable {
    let paths: [String]
    private var cache: [Int: (width: Int, height: Int, rgb: [UInt8])] = [:]
    private let lock = NSLock()
    init(paths: [String]) {
        self.paths = paths
    }
    func image(_ index: Int) -> (width: Int, height: Int, rgb: [UInt8])? {
        lock.lock()
        if let cached = cache[index] {
            lock.unlock()
            return cached
        }
        lock.unlock()
        guard let decoded = try? loadRGB(path: paths[index]) else {
            return nil
        }
        lock.lock()
        cache[index] = decoded
        lock.unlock()
        return decoded
    }
}
var backgroundPaths: [String] = []
for dir in backgroundDirs {
    backgroundPaths.append(contentsOf: listImages(directory: dir))
}
let backgrounds = BackgroundBank(paths: backgroundPaths)
print("backgrounds: \(backgroundPaths.count) images")

struct Rng {
    var state: SplitMix64
    init(seed: UInt64) {
        state = SplitMix64(seed: seed)
    }
    mutating func unit() -> Double {
        return Double(state.next() >> 11) / Double(1 << 53)
    }
    mutating func range(_ lo: Double, _ hi: Double) -> Double {
        return lo + (hi - lo) * unit()
    }
    mutating func int(_ lo: Int, _ hi: Int) -> Int {
        return lo + Int(unit() * Double(hi - lo + 1))
    }
    mutating func chance(_ p: Double) -> Bool {
        return unit() < p
    }
}

let w = DigitStrip.width
let h = DigitStrip.height
let letters = Array("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

/// Draws one attributed string with a stroke outline and fill; returns its advance width.
func drawText(_ context: CGContext, _ text: String, font: CTFont, size: CGFloat, at origin: CGPoint, fill: CGColor, stroke: CGColor?, strokeWidth: CGFloat) -> CGFloat {
    let sized = CTFontCreateCopyWithAttributes(font, size, nil, nil)
    // Colors come from the context (fill for the glyphs, stroke for the outline), not from the string.
    let attributes: [NSAttributedString.Key: Any] = [
        kCTFontAttributeName as NSAttributedString.Key: sized,
        kCTForegroundColorFromContextAttributeName as NSAttributedString.Key: true
    ]
    let line = CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: attributes))
    let width = CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil))
    context.saveGState()
    context.textPosition = origin
    if let stroke = stroke, 0 < strokeWidth {
        context.setTextDrawingMode(.stroke)
        context.setLineWidth(strokeWidth * 2)
        context.setLineJoin(.round)
        context.setStrokeColor(stroke)
        CTLineDraw(line, context)
        context.textPosition = origin
    }
    context.setTextDrawingMode(.fill)
    context.setFillColor(fill)
    CTLineDraw(line, context)
    context.restoreGState()
    return width
}

/// Everything the renderer needs, so worker threads never touch main-actor globals.
final class Config: @unchecked Sendable {
    let seed: UInt64
    let fonts: [CTFont]
    let backgrounds: BackgroundBank
    init(seed: UInt64, fonts: [CTFont], backgrounds: BackgroundBank) {
        self.seed = seed
        self.fonts = fonts
        self.backgrounds = backgrounds
    }
}

/// One synthetic strip and its label.
func render(index: Int, config: Config) -> DigitSample? {
    let seed = config.seed
    let fonts = config.fonts
    let backgrounds = config.backgrounds
    let backgroundPaths = backgrounds.paths
    var rng = Rng(seed: seed &* 0x9E3779B97F4A7C15 &+ UInt64(index))
    guard let context = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
        return nil
    }
    // Background: a real crop (scaled into the whole strip, or into the top part leaving black rows like a
    // width-limited crop) or a flat color.
    context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: w, height: h))
    let usedHeight = rng.chance(0.5) ? h : rng.int(Int(Double(h) * 0.75), h)
    let usedWidth = rng.chance(0.3) ? rng.int(Int(Double(w) * 0.5), w) : w
    var drewBackground = false
    if backgroundPaths.isEmpty != true && rng.chance(0.85) {
        let index = rng.int(0, backgroundPaths.count - 1)
        if let img = backgrounds.image(index) {
            let cropH = max(8, Int(Double(img.height) * rng.range(0.05, 0.2)))
            let cropW = min(img.width, max(8, Int(Double(cropH) * Double(usedWidth) / Double(usedHeight))))
            let cx = rng.int(0, max(0, img.width - cropW))
            let cy = rng.int(0, max(0, img.height - cropH))
            var rgba = [UInt8](repeating: 255, count: cropW * cropH * 4)
            var y = 0
            while y < cropH {
                var x = 0
                while x < cropW {
                    let src = ((cy + y) * img.width + cx + x) * 3
                    let dst = (y * cropW + x) * 4
                    rgba[dst] = img.rgb[src]
                    rgba[dst + 1] = img.rgb[src + 1]
                    rgba[dst + 2] = img.rgb[src + 2]
                    x += 1
                }
                y += 1
            }
            if let provider = CGDataProvider(data: Data(rgba) as CFData),
               let cg = CGImage(width: cropW, height: cropH, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: cropW * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) {
                context.interpolationQuality = .high
                context.draw(cg, in: CGRect(x: 0, y: h - usedHeight, width: usedWidth, height: usedHeight))
                drewBackground = true
            }
        }
    }
    if drewBackground != true {
        context.setFillColor(CGColor(red: rng.range(0, 1), green: rng.range(0, 1), blue: rng.range(0, 1), alpha: 1))
        context.fill(CGRect(x: 0, y: h - usedHeight, width: usedWidth, height: usedHeight))
    }
    // A dark panel behind the number (UI boxes), sometimes.
    if rng.chance(0.5) {
        let shade = rng.range(0.1, 0.35)
        context.setFillColor(CGColor(red: shade, green: shade, blue: shade + rng.range(0, 0.1), alpha: rng.range(0.6, 1.0)))
        let px = rng.range(0, Double(w) * 0.3)
        let py = rng.range(0, Double(usedHeight) * 0.3)
        context.fill(CGRect(x: px, y: Double(h - usedHeight) + py, width: rng.range(Double(w) * 0.4, Double(w) - px), height: rng.range(Double(usedHeight) * 0.4, Double(usedHeight) - py)))
    }

    // The number: 1-6 digits, weighted toward 4-5; leading zeros only for "0".
    let lengthRoll = rng.unit()
    let length: Int
    if lengthRoll < 0.05 { length = 1 } else if lengthRoll < 0.15 { length = 2 } else if lengthRoll < 0.3 { length = 3 } else if lengthRoll < 0.55 { length = 4 } else if lengthRoll < 0.9 { length = 5 } else { length = 6 }
    var digits = ""
    var k = 0
    while k < length {
        let d = (k == 0 && 1 < length) ? rng.int(1, 9) : rng.int(0, 9)
        digits.append(Character(String(d)))
        k += 1
    }
    let font = fonts[rng.int(0, fonts.count - 1)]
    // Digit height 35-75% of the used height.
    let digitHeight = Double(usedHeight) * rng.range(0.35, 0.75)
    let fontSize = CGFloat(digitHeight / 0.72)
    let yellow = rng.chance(0.55)
    let fill: CGColor
    if yellow {
        fill = CGColor(red: rng.range(0.9, 1.0), green: rng.range(0.7, 0.9), blue: rng.range(0.0, 0.3), alpha: 1)
    } else if rng.chance(0.8) {
        fill = CGColor(red: rng.range(0.9, 1.0), green: rng.range(0.9, 1.0), blue: rng.range(0.9, 1.0), alpha: 1)
    } else {
        fill = CGColor(red: rng.range(0.5, 1.0), green: rng.range(0.5, 1.0), blue: rng.range(0.5, 1.0), alpha: 1)
    }
    let outline = CGColor(red: rng.range(0, 0.25), green: rng.range(0, 0.2), blue: rng.range(0, 0.2), alpha: 1)
    let strokeWidth = CGFloat(rng.range(0.5, 2.5) * digitHeight / 20.0)
    // Measure to place the number fully inside the used area.
    let sized = CTFontCreateCopyWithAttributes(font, fontSize, nil, nil)
    let measureLine = CTLineCreateWithAttributedString(NSAttributedString(string: digits, attributes: [kCTFontAttributeName as NSAttributedString.Key: sized]))
    let textWidth = CGFloat(CTLineGetTypographicBounds(measureLine, nil, nil, nil))
    let iconWidth = rng.chance(0.7) ? digitHeight * rng.range(0.9, 1.4) : 0.0
    let maxX = Double(usedWidth) - Double(textWidth) - 4.0
    if maxX < 2.0 {
        return render(index: index + 1_000_003, config: config)  // too wide: try other random values
    }
    let textX = rng.range(min(2.0 + iconWidth, maxX), maxX)
    let baseline = Double(h - usedHeight) + rng.range(2.0, max(2.0, Double(usedHeight) - digitHeight - 2.0)) + digitHeight * 0.05
    // Icon left of the number: a cup-like emoji (its stem looks like a "1" and must not be read as one) or a blob.
    if 0.0 < iconWidth && iconWidth < textX {
        if rng.chance(0.6) {
            let emojiFont = CTFontCreateWithName("AppleColorEmoji" as CFString, CGFloat(digitHeight * 1.1), nil)
            let line = CTLineCreateWithAttributedString(NSAttributedString(string: ["🏆", "🥇", "🏅", "⭐"][rng.int(0, 3)], attributes: [kCTFontAttributeName as NSAttributedString.Key: emojiFont]))
            context.saveGState()
            context.textPosition = CGPoint(x: textX - iconWidth + digitHeight * 0.1, y: baseline - digitHeight * 0.05)
            CTLineDraw(line, context)
            context.restoreGState()
        } else {
            context.setFillColor(CGColor(red: rng.range(0.8, 1.0), green: rng.range(0.5, 0.8), blue: rng.range(0.0, 0.2), alpha: 1))
            let r = iconWidth * 0.45
            context.fillEllipse(in: CGRect(x: textX - iconWidth + r * 0.2, y: baseline, width: r * 2, height: r * 2 * rng.range(0.8, 1.2)))
        }
    }
    // Name-like distractor: above the number (two-line layout) or to its right.
    if rng.chance(0.6) {
        var word = ""
        var n = rng.int(3, 10)
        while 0 < n {
            word.append(letters[rng.int(0, letters.count - 1)])
            n -= 1
        }
        if rng.chance(0.3) {
            word.append(Character(String(rng.int(0, 99))))  // names may contain digits
        }
        let nameSize = fontSize * CGFloat(rng.range(0.6, 1.0))
        let nameFill = CGColor(red: rng.range(0.7, 1.0), green: rng.range(0.7, 1.0), blue: rng.range(0.7, 1.0), alpha: 1)
        if rng.chance(0.5) && digitHeight * 2.2 < Double(usedHeight) {
            _ = drawText(context, word, font: font, size: nameSize, at: CGPoint(x: textX - iconWidth * 0.5, y: baseline + digitHeight * 1.25), fill: nameFill, stroke: outline, strokeWidth: strokeWidth * 0.7)
        } else {
            _ = drawText(context, word, font: font, size: nameSize, at: CGPoint(x: textX + Double(textWidth) + digitHeight * rng.range(0.5, 2.0), y: baseline), fill: nameFill, stroke: outline, strokeWidth: strokeWidth * 0.7)
        }
    }
    _ = drawText(context, digits, font: font, size: fontSize, at: CGPoint(x: textX, y: baseline), fill: fill, stroke: outline, strokeWidth: strokeWidth)

    guard let data = context.data else {
        return nil
    }
    let bytes = data.bindMemory(to: UInt8.self, capacity: w * h * 4)
    var pixels = [Float](repeating: 0.0, count: DigitStrip.pixelCount)
    var i = 0
    while i < w * h {
        pixels[i * 3] = Float(bytes[i * 4]) / 255.0
        pixels[i * 3 + 1] = Float(bytes[i * 4 + 1]) / 255.0
        pixels[i * 3 + 2] = Float(bytes[i * 4 + 2]) / 255.0
        i += 1
    }
    // Light color jitter and noise so the renderer's exact colors are not memorized.
    var augmenter = Augmenter(seed: seed &+ UInt64(index) &* 31)
    let strip = augmenter.augment((try? DigitStrip(pixels: pixels)) ?? DigitStrip())
    return DigitSample(digits: digits, strip: strip)
}

final class Box: @unchecked Sendable {
    var samples: [DigitSample?]
    init(count: Int) {
        samples = [DigitSample?](repeating: nil, count: count)
    }
}
let start = CFAbsoluteTimeGetCurrent()
let url = URL(fileURLWithPath: outPath)
try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
let writer: DigitDatasetWriter
do {
    writer = try DigitDatasetWriter(url: url)
} catch {
    print("error: cannot create \(outPath): \(error)")
    exit(1)
}
let config = Config(seed: seed, fonts: fonts, backgrounds: backgrounds)
let chunk = 512
var done = 0
while done < count {
    let n = min(chunk, count - done)
    let box = Box(count: n)
    let base = done
    DispatchQueue.concurrentPerform(iterations: max(1, min(numWorkers, n))) { worker in
        var i = worker
        while i < n {
            box.samples[i] = autoreleasepool { render(index: base + i, config: config) }
            i += numWorkers
        }
    }
    for sample in box.samples {
        if let sample = sample {
            try? writer.append(sample)
        }
    }
    done += n
    if done % 5000 < chunk {
        print("  \(done)/\(count) (\(String(format: "%.0f", CFAbsoluteTimeGetCurrent() - start)) s)")
    }
}
try? writer.close()
print("wrote \(outPath): \(writer.writtenCount) samples (\(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - start)) s)")
