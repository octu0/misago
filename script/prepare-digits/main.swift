import Foundation
import Misago

setbuf(stdout, nil)

// Builds digit-strip datasets from a labels JSONL (output of `label`, reviewed by hand).
//
// Usage: prepare-digits --labels <labels.jsonl> --train <out.msgd> [--eval <out.msgd>] [options]
// Each row needs "path", "digits" and "crop" [x, y, w, h]. Rows with empty digits are skipped unless
// --keep-empty is given. The split uses a stable hash of the file name, so images never move between splits.

var labelsPath = ""
var trainPath = ""
var evalPath = ""
var evalFraction: Double = 0.15
var augmentCount = 10
var keepEmpty = false
var overridesPath = ""
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
        print("Usage: prepare-digits --labels <labels.jsonl> --train <out.msgd> [--eval <out.msgd>] [options]")
        print("Options:")
        print("  --eval-fraction <0-1>   Share of files sent to --eval (default 0.15)")
        print("  --augment <Int>         Augmented copies per training strip: crop jitter, color, noise (default 10)")
        print("  --keep-empty            Keep rows whose digits are empty (fields without a number)")
        print("  --overrides <json>      Hand corrections {\"<file name>\": \"<digits>\"} applied over the OCR labels")
        print("  -p, --parallel <Int>    Decoding threads (default 8)")
        exit(0)
    case "--labels":
        if let v = value() { labelsPath = v }
    case "--train":
        if let v = value() { trainPath = v }
    case "--eval":
        if let v = value() { evalPath = v }
    case "--eval-fraction":
        if let v = value(), let f = Double(v) { evalFraction = min(1.0, max(0.0, f)) }
    case "--augment":
        if let v = value(), let n = Int(v) { augmentCount = max(0, n) }
    case "--keep-empty":
        keepEmpty = true
    case "--overrides":
        if let v = value() { overridesPath = v }
    case "-p", "--parallel":
        if let v = value(), let n = Int(v) { numWorkers = max(1, n) }
    default:
        print("error: unknown argument \(arg)")
        exit(1)
    }
    argIdx += 1
}
if labelsPath.isEmpty || trainPath.isEmpty {
    print("error: specify --labels and --train. See prepare-digits --help.")
    exit(1)
}

struct Item {
    let path: String
    let digits: String
    let crop: (x: Int, y: Int, width: Int, height: Int)
}

func stableFraction(_ path: String) -> Double {
    var hash: UInt64 = 0xcbf29ce484222325
    for byte in (path as NSString).lastPathComponent.utf8 {
        hash ^= UInt64(byte)
        hash = hash &* 0x100000001b3
    }
    return Double(hash >> 11) / Double(1 << 53)
}

func seed(_ path: String, _ k: Int) -> UInt64 {
    var hash: UInt64 = 0x84222325cbf29ce4
    for byte in (path as NSString).lastPathComponent.utf8 {
        hash ^= UInt64(byte)
        hash = hash &* 0x100000001b3
    }
    return hash &+ UInt64(k) &* 0x9E3779B97F4A7C15
}

guard let content = try? String(contentsOfFile: labelsPath, encoding: .utf8) else {
    print("error: cannot read \(labelsPath)")
    exit(1)
}
var overrides: [String: String] = [:]
if overridesPath.isEmpty != true {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: overridesPath)),
          let object = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
        print("error: cannot read overrides \(overridesPath)")
        exit(1)
    }
    overrides = object
}
var trainItems: [Item] = []
var evalItems: [Item] = []
var skippedEmpty = 0
var skippedInvalid = 0
var overridden = 0
for line in content.components(separatedBy: "\n") {
    let trimmed = line.trimmingCharacters(in: .whitespaces)
    if trimmed.isEmpty {
        continue
    }
    guard let object = try? JSONSerialization.jsonObject(with: Data(trimmed.utf8)) as? [String: Any],
          let path = object["path"] as? String,
          var digits = object["digits"] as? String,
          let crop = object["crop"] as? [Int], crop.count == 4 else {
        skippedInvalid += 1
        continue
    }
    if let fixed = overrides[(path as NSString).lastPathComponent] {
        digits = fixed
        overridden += 1
    }
    if digits.isEmpty && keepEmpty != true {
        skippedEmpty += 1
        continue
    }
    if DigitVocabulary.ids(of: digits) == nil || DigitDatasetFormat.maxDigits < digits.count {
        skippedInvalid += 1
        continue
    }
    let item = Item(path: path, digits: digits, crop: (crop[0], crop[1], crop[2], crop[3]))
    if evalPath.isEmpty != true && stableFraction(path) < evalFraction {
        evalItems.append(item)
    } else {
        trainItems.append(item)
    }
}
print("labels: \(trainItems.count) train, \(evalItems.count) eval (skipped: \(skippedEmpty) empty, \(skippedInvalid) invalid; \(overridden) overridden)")

final class ResultBox: @unchecked Sendable {
    var strips: [[DigitStrip]]
    var failures: [String] = []
    let lock = NSLock()
    init(count: Int) {
        strips = [[DigitStrip]](repeating: [], count: count)
    }
}

func write(items: [Item], to path: String, workers: Int, augment: Int) {
    if items.isEmpty {
        return
    }
    let box = ResultBox(count: items.count)
    DispatchQueue.concurrentPerform(iterations: max(1, min(workers, items.count))) { worker in
        var i = worker
        while i < items.count {
            let item = items[i]
            let result: [DigitStrip] = autoreleasepool {
                guard let decoded = try? loadRGB(path: item.path),
                      let original = try? StripNormalizer.extract(width: decoded.width, height: decoded.height, rgb: decoded.rgb, x: item.crop.x, y: item.crop.y, cropWidth: item.crop.width, cropHeight: item.crop.height) else {
                    return []
                }
                var out = [original]
                var k = 0
                while k < augment {
                    var augmenter = Augmenter(seed: seed(item.path, k))
                    let r = augmenter.jitter(x: item.crop.x, y: item.crop.y, width: item.crop.width, height: item.crop.height)
                    if let strip = try? StripNormalizer.extract(width: decoded.width, height: decoded.height, rgb: decoded.rgb, x: r.x, y: r.y, cropWidth: r.width, cropHeight: r.height) {
                        out.append(augmenter.augment(strip))
                    }
                    k += 1
                }
                return out
            }
            box.lock.lock()
            box.strips[i] = result
            if result.isEmpty {
                box.failures.append(item.path)
            }
            box.lock.unlock()
            i += workers
        }
    }
    let url = URL(fileURLWithPath: path)
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    do {
        let writer = try DigitDatasetWriter(url: url)
        var meta: [[String: Any]] = []
        var i = 0
        while i < items.count {
            var k = 0
            for strip in box.strips[i] {
                try writer.append(DigitSample(digits: items[i].digits, strip: strip))
                meta.append(["path": items[i].path, "digits": items[i].digits, "augmented": k])
                k += 1
            }
            i += 1
        }
        try writer.close()
        print("wrote \(path): \(writer.writtenCount) samples (\(items.count) images, \(augment) augmented copies each)")
        let metaData = try JSONSerialization.data(withJSONObject: ["samples": meta], options: [.prettyPrinted, .sortedKeys])
        try metaData.write(to: URL(fileURLWithPath: path + ".meta.json"))
    } catch {
        print("error: cannot write \(path): \(error)")
        exit(1)
    }
    for failure in box.failures {
        print("  skipped (cannot decode): \(failure)")
    }
}

let start = CFAbsoluteTimeGetCurrent()
write(items: trainItems, to: trainPath, workers: numWorkers, augment: augmentCount)
if evalPath.isEmpty != true {
    write(items: evalItems, to: evalPath, workers: numWorkers, augment: 0)
}
print("done (\(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - start)) s)")
