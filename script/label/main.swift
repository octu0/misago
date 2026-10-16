import Foundation
import ImageIO
import CoreGraphics
import UniformTypeIdentifiers
import Vision
import Misago

setbuf(stdout, nil)

// Labels number fields with the system OCR (Vision) for review and training.
//
// Usage: label --regions <regions.json> --class <name>=<dir> ... --out <labels.jsonl> [--sheets <dir>] [--limit N]
// For every image, the class's crop region is cut out at source resolution, upscaled and read with Vision;
// the longest run of digits becomes the label. The JSONL holds path, class, digits, the raw OCR text and
// confidence, so wrong labels can be fixed by hand before training. Contact sheets show the crops in JSONL
// order, 16 per sheet.

struct ClassFolder {
    let name: String
    let directory: String
}

var regionsPath = ""
var folders: [ClassFolder] = []
var outPath = ""
var sheetsDir = ""
var limit = Int.max
var upscale: CGFloat = 3.0

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
        print("Usage: label --regions <regions.json> --class <name>=<dir> ... --out <labels.jsonl> [--sheets <dir>] [--limit N] [--upscale 3]")
        exit(0)
    case "--regions":
        if let v = value() { regionsPath = v }
    case "--class":
        guard let v = value(), let eq = v.firstIndex(of: "=") else {
            print("error: --class expects <name>=<dir>")
            exit(1)
        }
        folders.append(ClassFolder(name: String(v[v.startIndex..<eq]), directory: String(v[v.index(after: eq)...])))
    case "--out":
        if let v = value() { outPath = v }
    case "--sheets":
        if let v = value() { sheetsDir = v }
    case "--limit":
        if let v = value(), let n = Int(v) { limit = max(1, n) }
    case "--upscale":
        if let v = value(), let f = Double(v) { upscale = CGFloat(max(1.0, f)) }
    default:
        print("error: unknown argument \(arg)")
        exit(1)
    }
    argIdx += 1
}
if regionsPath.isEmpty || folders.isEmpty || outPath.isEmpty {
    print("error: specify --regions, at least one --class and --out. See label --help.")
    exit(1)
}
let regions: RegionTable
do {
    regions = try RegionTable.load(from: URL(fileURLWithPath: regionsPath))
} catch {
    print("error: cannot read regions \(regionsPath): \(error)")
    exit(1)
}

/// CGImage from interleaved RGB bytes of a crop.
func makeImage(width: Int, height: Int, rgb: [UInt8], x: Int, y: Int, cropWidth: Int, cropHeight: Int) -> CGImage? {
    var rgba = [UInt8](repeating: 255, count: cropWidth * cropHeight * 4)
    var yy = 0
    while yy < cropHeight {
        var xx = 0
        while xx < cropWidth {
            let src = ((y + yy) * width + (x + xx)) * 3
            let dst = (yy * cropWidth + xx) * 4
            rgba[dst] = rgb[src]
            rgba[dst + 1] = rgb[src + 1]
            rgba[dst + 2] = rgb[src + 2]
            xx += 1
        }
        yy += 1
    }
    let data = Data(rgba)
    guard let provider = CGDataProvider(data: data as CFData) else {
        return nil
    }
    return CGImage(
        width: cropWidth, height: cropHeight, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: cropWidth * 4,
        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
        provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent
    )
}

func upscaled(_ image: CGImage, by factor: CGFloat) -> CGImage {
    let w = Int(CGFloat(image.width) * factor)
    let h = Int(CGFloat(image.height) * factor)
    guard let context = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
        return image
    }
    context.interpolationQuality = .high
    context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    return context.makeImage() ?? image
}

/// Recognized strings with confidence and box height, left to right.
func recognize(_ image: CGImage) -> [(text: String, confidence: Float, minX: CGFloat, height: CGFloat)] {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    request.recognitionLanguages = ["en-US"]
    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    do {
        try handler.perform([request])
    } catch {
        return []
    }
    var out: [(String, Float, CGFloat, CGFloat)] = []
    for observation in request.results ?? [] {
        if let candidate = observation.topCandidates(1).first {
            out.append((candidate.string, candidate.confidence, observation.boundingBox.minX, observation.boundingBox.height))
        }
    }
    return out.sorted { $0.2 < $1.2 }
}

/// Longest run of ASCII digits in the text (ties: the first).
func longestDigitRun(_ text: String) -> String {
    var best = ""
    var current = ""
    for ch in text {
        if ch.isASCII && ch.isNumber {
            current.append(ch)
        } else {
            if best.count < current.count {
                best = current
            }
            current = ""
        }
    }
    if best.count < current.count {
        best = current
    }
    return best
}

struct Row {
    let path: String
    let className: String
    let digits: String
    let text: String
    let confidence: Float
    let crop: (x: Int, y: Int, width: Int, height: Int)
    let image: CGImage?
}

var rows: [Row] = []
for folder in folders {
    guard let region = regions.regions[folder.name] else {
        print("warning: no region for class \(folder.name); skipped")
        continue
    }
    var count = 0
    for path in listImages(directory: folder.directory) {
        if limit <= count {
            break
        }
        count += 1
        autoreleasepool {
            guard let decoded = try? loadRGB(path: path) else {
                print("  skipped (cannot decode): \(path)")
                return
            }
            let r = region.pixelRect(width: decoded.width, height: decoded.height)
            let image = makeImage(width: decoded.width, height: decoded.height, rgb: decoded.rgb, x: r.x, y: r.y, cropWidth: r.width, cropHeight: r.height)
            var digits = ""
            var text = ""
            var confidence: Float = 0.0
            if let image = image {
                let results = recognize(upscaled(image, by: upscale))
                text = results.map { $0.text }.joined(separator: " | ")
                // The field's number is the tallest text with digits (secondary numbers such as a personal best
                // are printed smaller); among equal heights take the longer run.
                var bestHeight: CGFloat = -1.0
                for result in results {
                    let run = longestDigitRun(result.text)
                    if run.isEmpty {
                        continue
                    }
                    if bestHeight < result.height * 0.9 || (abs(result.height - bestHeight) <= bestHeight * 0.1 && digits.count < run.count) {
                        digits = run
                        confidence = result.confidence
                        bestHeight = max(bestHeight, result.height)
                    }
                }
            }
            rows.append(Row(path: path, className: folder.name, digits: digits, text: text, confidence: confidence, crop: r, image: image))
            print("\(folder.name) \(digits.isEmpty ? "-" : digits) (\(String(format: "%.2f", confidence))) \"\(text)\" \(path)")
        }
    }
}

// JSONL
var lines: [String] = []
for row in rows {
    let object: [String: Any] = [
        "path": row.path, "class": row.className, "digits": row.digits, "text": row.text,
        "confidence": Double(String(format: "%.3f", row.confidence)) ?? 0.0,
        "crop": [row.crop.x, row.crop.y, row.crop.width, row.crop.height]
    ]
    if let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]), let text = String(data: data, encoding: .utf8) {
        lines.append(text)
    }
}
do {
    try FileManager.default.createDirectory(at: URL(fileURLWithPath: outPath).deletingLastPathComponent(), withIntermediateDirectories: true)
    try (lines.joined(separator: "\n") + "\n").write(toFile: outPath, atomically: true, encoding: .utf8)
} catch {
    print("error: cannot write \(outPath): \(error)")
    exit(1)
}
let labeled = rows.filter { $0.digits.isEmpty != true }.count
print("wrote \(outPath): \(rows.count) rows, \(labeled) with digits")

// Contact sheets: 16 crops per sheet, one numbered series per class, each crop scaled to 800 px wide.
// index.txt maps sheet and row to the file and its label so corrections can be written straight into an
// overrides file.
if sheetsDir.isEmpty != true {
    try? FileManager.default.createDirectory(atPath: sheetsDir, withIntermediateDirectories: true)
    let perSheet = 16
    let cellW = 800
    var indexLines: [String] = []
    var classOrder: [String] = []
    for row in rows where classOrder.contains(row.className) != true {
        classOrder.append(row.className)
    }
    for className in classOrder {
        let classRows = rows.filter { $0.className == className }
        var sheet = 0
        var start = 0
        while start < classRows.count {
            let end = min(classRows.count, start + perSheet)
            let name = String(format: "sheet_%@_%03d.png", className, sheet)
            var heights: [Int] = []
            var i = start
            while i < end {
                if let img = classRows[i].image {
                    heights.append(max(1, Int(Double(img.height) * Double(cellW) / Double(img.width))))
                } else {
                    heights.append(20)
                }
                indexLines.append("\(name)\trow \(i - start)\t\(classRows[i].digits.isEmpty ? "-" : classRows[i].digits)\t\((classRows[i].path as NSString).lastPathComponent)")
                i += 1
            }
            let total = heights.reduce(0, +) + (end - start) * 4
            if let context = CGContext(data: nil, width: cellW, height: total, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) {
                context.setFillColor(CGColor(red: 1, green: 0, blue: 1, alpha: 1))
                context.fill(CGRect(x: 0, y: 0, width: cellW, height: total))
                context.interpolationQuality = .high
                var yTop = 0
                i = start
                while i < end {
                    let h = heights[i - start]
                    if let img = classRows[i].image {
                        // CGContext origin is bottom-left.
                        context.draw(img, in: CGRect(x: 0, y: total - yTop - h, width: cellW, height: h))
                    }
                    yTop += h + 4
                    i += 1
                }
                if let image = context.makeImage() {
                    let url = URL(fileURLWithPath: sheetsDir).appendingPathComponent(name)
                    if let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) {
                        CGImageDestinationAddImage(dest, image, nil)
                        CGImageDestinationFinalize(dest)
                    }
                }
            }
            sheet += 1
            start = end
        }
        print("wrote \(sheet) sheets for \(className) to \(sheetsDir)")
    }
    try? (indexLines.joined(separator: "\n") + "\n").write(toFile: (sheetsDir as NSString).appendingPathComponent("index.txt"), atomically: true, encoding: .utf8)
    print("wrote \(sheetsDir)/index.txt (sheet, row, label, file)")
}
