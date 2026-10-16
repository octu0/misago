import Foundation
import Misago

setbuf(stdout, nil)

// Runs the whole pipeline on image files: classify, crop the class's number field, read the digits.
// Usage: read --weights <classifier.json> --digit-weights <digits.json> --regions <regions.json> [--threshold 0.5] <image>...
// Prints one JSON object per line: class, probability, number (-1 when none), digits, confidence, ms.

var weightsPath = ""
var digitWeightsPath = ""
var regionsPath = ""
var threshold: Float = 0.5
var profile = false
var imagePaths: [String] = []
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
        print("Usage: read --weights <classifier.json> --digit-weights <digits.json> --regions <regions.json> [--threshold 0.5] [--profile] <image>...")
        exit(0)
    case "-w", "--weights":
        if let v = value() { weightsPath = v }
    case "--digit-weights":
        if let v = value() { digitWeightsPath = v }
    case "--regions":
        if let v = value() { regionsPath = v }
    case "--threshold":
        if let v = value(), let f = Float(v) { threshold = f }
    case "--profile":
        profile = true
    default:
        if arg.hasPrefix("-") {
            print("error: unknown argument \(arg)")
            exit(1)
        }
        imagePaths.append(arg)
    }
    argIdx += 1
}
if weightsPath.isEmpty || digitWeightsPath.isEmpty || regionsPath.isEmpty || imagePaths.isEmpty {
    print("error: specify --weights, --digit-weights, --regions and at least one image. See read --help.")
    exit(1)
}
let reader: ScreenReader
do {
    let classifier = try ImageClassifier(weights: try HawkWeights.load(from: URL(fileURLWithPath: weightsPath)), threshold: threshold)
    let digits = try DigitReader(weights: try DigitReaderWeights.load(from: URL(fileURLWithPath: digitWeightsPath)))
    let regions = try RegionTable.load(from: URL(fileURLWithPath: regionsPath))
    reader = ScreenReader(classifier: classifier, digitReader: digits, regions: regions)
} catch {
    print("error: cannot load models: \(error)")
    exit(1)
}
for path in imagePaths {
    do {
        let decoded = try loadRGB(path: path)
        let start = CFAbsoluteTimeGetCurrent()
        let result = try reader.read(width: decoded.width, height: decoded.height, rgb: decoded.rgb)
        let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        var object: [String: Any] = [
            "path": path,
            "class": result.verdict.className,
            "probability": Double(String(format: "%.4f", result.verdict.probability)) ?? 0.0,
            "number": result.number,
            "outcome": result.outcome,
            "ms": Double(String(format: "%.2f", ms)) ?? 0.0
        ]
        if let reading = result.reading {
            object["digits"] = reading.digits
            object["confidence"] = Double(String(format: "%.4f", reading.confidence)) ?? 0.0
        }
        if profile {
            // Stage timings, repeated so one-off costs do not dominate.
            var normalizeMs = 0.0, classifyMs = 0.0, extractMs = 0.0, readMs = 0.0
            var image = ScreenImage()
            var verdict: ImageVerdict? = nil
            let runs = 5
            var r = 0
            while r < runs {
                var t = CFAbsoluteTimeGetCurrent()
                image = try ScreenNormalizer.normalize(width: decoded.width, height: decoded.height, rgb: decoded.rgb)
                normalizeMs += (CFAbsoluteTimeGetCurrent() - t) * 1000.0
                t = CFAbsoluteTimeGetCurrent()
                verdict = reader.classifier.classify(image)
                classifyMs += (CFAbsoluteTimeGetCurrent() - t) * 1000.0
                if let v = verdict, let region = reader.regions.regions[v.className] {
                    t = CFAbsoluteTimeGetCurrent()
                    let strip = try StripNormalizer.extract(width: decoded.width, height: decoded.height, rgb: decoded.rgb, region: region)
                    extractMs += (CFAbsoluteTimeGetCurrent() - t) * 1000.0
                    t = CFAbsoluteTimeGetCurrent()
                    _ = reader.digitReader.read(strip)
                    readMs += (CFAbsoluteTimeGetCurrent() - t) * 1000.0
                }
                r += 1
            }
            let n = Double(runs)
            object["profile_ms"] = ["normalize": normalizeMs / n, "classify": classifyMs / n, "extract": extractMs / n, "read": readMs / n].mapValues { Double(String(format: "%.2f", $0)) ?? 0.0 }
        }
        if let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]), let text = String(data: data, encoding: .utf8) {
            print(text)
        }
    } catch {
        print("{\"path\": \"\(path)\", \"error\": \"\(error)\"}")
    }
}
