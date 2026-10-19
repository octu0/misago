import Foundation
import Misago

setbuf(stdout, nil)

// Builds dataset files from image folders.
//
// Usage: prepare --class <name>=<dir> ... --train <out.msgo> [--eval <out.msgo>] [options]
// Each --class names a class and a folder of its images (walked recursively). The class `other` is always
// class 0 and holds everything that is not a recognized screen; it may be given with --class other=<dir>.
// Other class names are free, for example <title>_home. Without --eval-class folders, every file is assigned
// to the eval split with a stable hash of its file name, so re-running with more images keeps earlier
// assignments.

struct ClassFolder {
    let name: String
    let directory: String
}

var classFolders: [ClassFolder] = []
var evalFolders: [ClassFolder] = []
var trainPath = ""
var evalPath = ""
var evalFraction: Double = 0.1
var numWorkers = 8
var augmentCount = 0
var centerMask: [Float]? = nil

func parseClass(_ spec: String) -> ClassFolder? {
    guard let eq = spec.firstIndex(of: "=") else {
        return nil
    }
    let name = String(spec[spec.startIndex..<eq])
    let dir = String(spec[spec.index(after: eq)...])
    if name.isEmpty || dir.isEmpty {
        return nil
    }
    return ClassFolder(name: name, directory: dir)
}

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
        print("Usage: prepare --class <name>=<dir> ... --train <out.msgo> [--eval <out.msgo>] [options]")
        print("Options:")
        print("  --class <name>=<dir>        Images of one class (repeatable). `other` is class 0; the rest are numbered in order of first appearance")
        print("  --eval-class <name>=<dir>   Images used only for evaluation (repeatable)")
        print("  --train <path>              Output dataset for training [required]")
        print("  --eval <path>               Output dataset for evaluation")
        print("  --eval-fraction <0-1>       Share of --class files sent to --eval when no --eval-class is given (default 0.1)")
        print("  --augment <Int>             Augmented copies per training image (crop, color, bars, cutout, noise; default 0)")
        print("  --center-mask x0,y0,x1,y1   Black out this rectangle (fractions) in every image; train with the same value")
        print("  -p, --parallel <Int>        Decoding threads (default 8)")
        exit(0)
    case "--class":
        guard let v = value(), let folder = parseClass(v) else {
            print("error: --class expects <name>=<dir>")
            exit(1)
        }
        classFolders.append(folder)
    case "--eval-class":
        guard let v = value(), let folder = parseClass(v) else {
            print("error: --eval-class expects <name>=<dir>")
            exit(1)
        }
        evalFolders.append(folder)
    case "--train":
        if let v = value() { trainPath = v }
    case "--eval":
        if let v = value() { evalPath = v }
    case "--eval-fraction":
        if let v = value(), let f = Double(v) { evalFraction = min(1.0, max(0.0, f)) }
    case "-p", "--parallel":
        if let v = value(), let n = Int(v) { numWorkers = max(1, n) }
    case "--augment":
        if let v = value(), let n = Int(v) { augmentCount = max(0, n) }
    case "--center-mask":
        if let v = value() {
            let parts = v.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
            if parts.count != 4 {
                print("error: --center-mask expects x0,y0,x1,y1")
                exit(1)
            }
            centerMask = parts
        }
    default:
        print("error: unknown argument \(arg)")
        exit(1)
    }
    argIdx += 1
}

if trainPath.isEmpty {
    print("error: specify --train <out.msgo>. See prepare --help.")
    exit(1)
}
if classFolders.isEmpty {
    print("error: specify at least one --class <name>=<dir>")
    exit(1)
}

// Class order: `other` first, then the other names in order of first appearance.
var names: [String] = []
for folder in classFolders + evalFolders where folder.name != ClassList.otherName && names.contains(folder.name) != true {
    names.append(folder.name)
}
let classes: ClassList
do {
    classes = try ClassList(others: names)
} catch {
    print("error: invalid class names: \(error)")
    exit(1)
}
print("classes: \(classes.names.enumerated().map { "\($0.offset)=\($0.element)" }.joined(separator: ", "))")

/// FNV-1a of the file name, mapped to [0, 1).
func stableFraction(_ path: String) -> Double {
    var hash: UInt64 = 0xcbf29ce484222325
    for byte in (path as NSString).lastPathComponent.utf8 {
        hash ^= UInt64(byte)
        hash = hash &* 0x100000001b3
    }
    return Double(hash >> 11) / Double(1 << 53)
}

struct Item {
    let path: String
    let label: Int
}

var trainItems: [Item] = []
var evalItems: [Item] = []
let useFraction = evalFolders.isEmpty && evalPath.isEmpty != true
for folder in classFolders {
    let label = classes.index(of: folder.name)!
    let files = listImages(directory: folder.directory)
    if files.isEmpty {
        print("warning: no images under \(folder.directory)")
    }
    for file in files {
        if useFraction && stableFraction(file) < evalFraction {
            evalItems.append(Item(path: file, label: label))
        } else {
            trainItems.append(Item(path: file, label: label))
        }
    }
}
for folder in evalFolders {
    let label = classes.index(of: folder.name)!
    for file in listImages(directory: folder.directory) {
        evalItems.append(Item(path: file, label: label))
    }
}

func summary(_ items: [Item]) -> String {
    var counts = [Int](repeating: 0, count: classes.count)
    for item in items {
        counts[item.label] += 1
    }
    return "\(items.count) images (" + counts.enumerated().map { "\(classes[$0.offset]) \($0.element)" }.joined(separator: ", ") + ")"
}
print("train: \(summary(trainItems))")
print("eval:  \(summary(evalItems))")

final class ResultBox: @unchecked Sendable {
    /// Per item: the original followed by its augmented copies.
    var images: [[ScreenImage]]
    var failures: [String] = []
    let lock = NSLock()
    init(count: Int) {
        images = [[ScreenImage]](repeating: [], count: count)
    }
}

/// Seed for the k-th augmentation of a file, stable across runs.
func augmentationSeed(_ path: String, _ k: Int) -> UInt64 {
    var hash: UInt64 = 0xcbf29ce484222325
    for byte in (path as NSString).lastPathComponent.utf8 {
        hash ^= UInt64(byte)
        hash = hash &* 0x100000001b3
    }
    return hash &+ UInt64(k) &* 0x9E3779B97F4A7C15
}

func write(items: [Item], to path: String, classes: ClassList, workers: Int, augment: Int, mask: [Float]?) {
    if items.isEmpty {
        return
    }
    let box = ResultBox(count: items.count)
    let paths = items.map { $0.path }
    DispatchQueue.concurrentPerform(iterations: max(1, min(workers, items.count))) { worker in
        var i = worker
        while i < paths.count {
            let result: [ScreenImage] = autoreleasepool {
                guard let decoded = try? loadRGB(path: paths[i]),
                      let original = try? ScreenNormalizer.normalize(width: decoded.width, height: decoded.height, rgb: decoded.rgb) else {
                    return []
                }
                var out = [mask.map { original.masked($0) } ?? original]
                var k = 0
                while k < augment {
                    var augmenter = Augmenter(seed: augmentationSeed(paths[i], k))
                    let crop = augmenter.cropRect(width: decoded.width, height: decoded.height)
                    if let cropped = try? ScreenNormalizer.normalize(
                        width: decoded.width, height: decoded.height, rgb: decoded.rgb,
                        cropX: crop.x, cropY: crop.y, cropWidth: crop.width, cropHeight: crop.height
                    ) {
                        let augmented = augmenter.augment(cropped)
                        out.append(mask.map { augmented.masked($0) } ?? augmented)
                    }
                    k += 1
                }
                return out
            }
            box.lock.lock()
            box.images[i] = result
            if result.isEmpty {
                box.failures.append(paths[i])
            }
            box.lock.unlock()
            i += workers
        }
    }
    let url = URL(fileURLWithPath: path)
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    do {
        let writer = try DatasetWriter(url: url, classes: classes)
        var i = 0
        while i < items.count {
            for image in box.images[i] {
                try writer.append(DatasetSample(label: items[i].label, image: image))
            }
            i += 1
        }
        try writer.close()
        print("wrote \(path): \(writer.writtenCount) samples (\(items.count) images, \(augment) augmented copies each)")
        // Sidecar with the source path of every sample, in file order, for inspecting evaluation results.
        var meta: [[String: Any]] = []
        i = 0
        while i < items.count {
            var k = 0
            while k < box.images[i].count {
                meta.append(["path": items[i].path, "label": items[i].label, "augmented": k])
                k += 1
            }
            i += 1
        }
        let metaObject: [String: Any] = ["classes": classes.names, "samples": meta]
        let metaData = try JSONSerialization.data(withJSONObject: metaObject, options: [.prettyPrinted, .sortedKeys])
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
write(items: trainItems, to: trainPath, classes: classes, workers: numWorkers, augment: augmentCount, mask: centerMask)
if evalPath.isEmpty != true {
    write(items: evalItems, to: evalPath, classes: classes, workers: numWorkers, augment: 0, mask: centerMask)
}
print("done (\(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - start)) s)")
