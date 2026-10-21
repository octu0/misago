import Foundation
import Misago

setbuf(stdout, nil)

// Builds dataset files from image folders.
//
// Usage: prepare --class <name>=<dir> ... --train <out.msgo> [--eval <out.msgo>] [options]
// Each --class names a class and a folder of its images (walked recursively). Class names are
// <title>_<screen>, for example <title>_home; the title (before the first underscore) gets its own independent
// output. --none folders hold images of nothing the model should know: they train every title output towards 0.
// Without --eval-class folders, every file is assigned to the eval split with a stable hash of its file name,
// so re-running with more images keeps earlier assignments. Portrait images are turned to landscape first, as
// the reader does at inference.

struct ClassFolder {
    let name: String
    let directory: String
}

var classFolders: [ClassFolder] = []
var evalFolders: [ClassFolder] = []
var noneFolders: [String] = []
var syntheticNoneCount = 0
var noneEvalFraction: Double = 0.0
var trainPath = ""
var evalPath = ""
var evalFraction: Double = 0.1
var numWorkers = 8
var augmentCount = 0
var centerMask: [Float]? = nil
var rotations = [0]

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
        print("  --none <dir>                Images of nothing to recognize: no class, every title output 0 (repeatable)")
        print("  --none-eval-fraction <0-1>  Share of --none files held out for --eval, where the title probabilities are")
        print("                              calibrated; the rest is used for training (default 0: all train)")
        print("  --synthetic-none <Int>      Also generate this many synthetic none images for training (solid, gradients,")
        print("                              noise, stripes, shapes; a tenth as many for --eval)")
        print("  --train <path>              Output dataset for training [required]")
        print("  --eval <path>               Output dataset for evaluation")
        print("  --eval-fraction <0-1>       Share of --class files sent to --eval when no --eval-class is given (default 0.1)")
        print("  --augment <Int>             Augmented copies per training image (crop, color, bars, cutout, noise; default 0)")
        print("  --center-mask x0,y0,x1,y1   Black out this rectangle (fractions) in every image; train with the same value")
        print("  --rotations <list>          Rotations to train, e.g. 0,180: copies turned 180° clockwise are labeled")
        print("                              <class>@180, so the classifier also tells the orientation. Only 0 and 180:")
        print("                              portrait input is turned to landscape before classification (default 0)")
        print("  -p, --parallel <Int>        Decoding threads (default 8)")
        exit(0)
    case "--class":
        guard let v = value(), let folder = parseClass(v) else {
            print("error: --class expects <name>=<dir>")
            exit(1)
        }
        classFolders.append(folder)
    case "--none":
        if let v = value() { noneFolders.append(v) }
    case "--none-eval-fraction":
        if let v = value(), let f = Double(v) { noneEvalFraction = min(1, max(0, f)) }
    case "--synthetic-none":
        if let v = value(), let n = Int(v) { syntheticNoneCount = max(0, n) }
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
    case "--rotations":
        if let v = value() {
            let list = v.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            if list.first != 0 || list.contains(where: { $0 != 0 && $0 != 180 }) || Set(list).count != list.count {
                print("error: --rotations takes 0 first, optionally followed by 180")
                exit(1)
            }
            rotations = list
        }
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
for folder in classFolders + evalFolders where names.contains(folder.name) != true {
    if ClassList.parseRotation(folder.name).rotation != 0 {
        print("error: class \(folder.name) carries a rotation tag; use --rotations instead")
        exit(1)
    }
    names.append(folder.name)
}
// Rotated copies keep their class and add the angle.
for rotation in rotations where rotation != 0 {
    for name in names where names.contains(ClassList.rotated(name, rotation)) != true {
        names.append(ClassList.rotated(name, rotation))
    }
}
let classes: ClassList
do {
    classes = try ClassList(names: names)
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

/// Label of a class turned by `rotation` (the class itself for 0; none stays none).
func rotatedLabel(_ label: Int, _ rotation: Int) -> Int {
    if label < 0 || rotation == 0 {
        return label
    }
    return classes.index(of: ClassList.rotated(classes[label], rotation)) ?? label
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
for folder in noneFolders {
    let files = listImages(directory: folder)
    if files.isEmpty {
        print("warning: no images under \(folder)")
    }
    // A small held-out share lets the title calibration see real images of nothing; synthetic ones alone are
    // too easy to separate and leave the cut-off undetermined.
    for file in files {
        if evalPath.isEmpty != true && stableFraction(file) < noneEvalFraction {
            evalItems.append(Item(path: file, label: DatasetSample.noneLabel))
        } else {
            trainItems.append(Item(path: file, label: DatasetSample.noneLabel))
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
    var none = 0
    for item in items {
        if item.label < 0 {
            none += 1
        } else {
            counts[item.label] += 1
        }
    }
    return "\(items.count) images (" + counts.enumerated().filter { ClassList.parseRotation(classes[$0.offset]).rotation == 0 }.map { "\(classes[$0.offset]) \($0.element)" }.joined(separator: ", ") + ", none \(none))"
}
print("train: \(summary(trainItems))")
print("eval:  \(summary(evalItems))")

/// Synthetic images of nothing: solid colours, gradients, block and pixel noise, stripes, rectangles and circles,
/// and vivid abstract artwork without any UI (soft colour blobs, curved colour bands, glowing lines and dots on a
/// dark ground), drawn directly at the classifier's resolution. They teach the title outputs that colour and
/// shape alone, without a title's UI, are not a screen of any title.
func syntheticNone(index: Int, seed: UInt64) -> ScreenImage {
    var rng = SplitMix64(seed: seed &+ UInt64(index) &* 0x9E3779B97F4A7C15)
    func unit() -> Float {
        return Float(rng.next() >> 40) / Float(1 << 24)
    }
    func color() -> (Float, Float, Float) {
        return (unit(), unit(), unit())
    }
    /// A saturated colour: random hue, high saturation and value.
    func vivid() -> (Float, Float, Float) {
        let hue = unit() * 6
        let s: Float = 0.6 + 0.4 * unit()
        let v: Float = 0.6 + 0.4 * unit()
        let f = hue - Float(Int(hue))
        let p0 = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f))
        switch Int(hue) % 6 {
        case 0: return (v, t, p0)
        case 1: return (q, v, p0)
        case 2: return (p0, v, t)
        case 3: return (p0, q, v)
        case 4: return (t, p0, v)
        default: return (v, p0, q)
        }
    }
    let w = ScreenImage.width
    let h = ScreenImage.height
    var p = [Float](repeating: 0.0, count: ScreenImage.pixelCount)
    func set(_ x: Int, _ y: Int, _ c: (Float, Float, Float)) {
        if 0 <= x && x < w && 0 <= y && y < h {
            let i = (y * w + x) * 3
            p[i] = c.0
            p[i + 1] = c.1
            p[i + 2] = c.2
        }
    }
    func fill(_ c: (Float, Float, Float)) {
        for y in 0..<h {
            for x in 0..<w {
                set(x, y, c)
            }
        }
    }
    func gradient() {
        let a = color(), b = color()
        let angle = unit() * 2 * Float.pi
        let dx = cos(angle), dy = sin(angle)
        for y in 0..<h {
            for x in 0..<w {
                let t = min(1, max(0, 0.5 + (Float(x - w / 2) * dx + Float(y - h / 2) * dy) / Float(w)))
                set(x, y, (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t, a.2 + (b.2 - a.2) * t))
            }
        }
    }
    func blocks() {
        let cell = 1 + Int(unit() * 16)
        var y = 0
        while y < h {
            var x = 0
            while x < w {
                let c = color()
                for yy in y..<min(h, y + cell) {
                    for xx in x..<min(w, x + cell) {
                        set(xx, yy, c)
                    }
                }
                x += cell
            }
            y += cell
        }
    }
    func stripes() {
        let a = color(), b = color()
        let band = 1 + Int(unit() * 12)
        let mode = Int(unit() * 3)
        for y in 0..<h {
            for x in 0..<w {
                let k = mode == 0 ? y : (mode == 1 ? x : x + y)
                set(x, y, (k / band) % 2 == 0 ? a : b)
            }
        }
    }
    func rectangles() {
        for _ in 0..<(3 + Int(unit() * 25)) {
            let c = color()
            let x0 = Int(unit() * Float(w)), y0 = Int(unit() * Float(h))
            let rw = 2 + Int(unit() * Float(w) * 0.5), rh = 2 + Int(unit() * Float(h) * 0.5)
            for y in y0..<min(h, y0 + rh) {
                for x in x0..<min(w, x0 + rw) {
                    set(x, y, c)
                }
            }
        }
    }
    func circles() {
        for _ in 0..<(2 + Int(unit() * 12)) {
            let c = color()
            let cx = unit() * Float(w), cy = unit() * Float(h), r = 2 + unit() * Float(h) * 0.5
            for y in 0..<h {
                for x in 0..<w {
                    let ex = Float(x) - cx, ey = Float(y) - cy
                    if ex * ex + ey * ey <= r * r {
                        set(x, y, c)
                    }
                }
            }
        }
    }
    func pixelNoise() {
        let amplitude = 0.1 + unit() * 0.9
        let base = color()
        for i in stride(from: 0, to: p.count, by: 3) {
            p[i] = min(1, max(0, base.0 + (unit() - 0.5) * amplitude))
            p[i + 1] = min(1, max(0, base.1 + (unit() - 0.5) * amplitude))
            p[i + 2] = min(1, max(0, base.2 + (unit() - 0.5) * amplitude))
        }
    }
    func blobs() {
        let count = 3 + Int(unit() * 5)
        let centers = (0..<count).map { _ in (unit() * Float(w), unit() * Float(h), (0.2 + unit() * 0.6) * Float(w), vivid()) }
        for y in 0..<h {
            for x in 0..<w {
                var total: Float = 1e-6
                var c: (Float, Float, Float) = (0, 0, 0)
                for b in centers {
                    let dx = Float(x) - b.0, dy = Float(y) - b.1
                    let weight = exp(-(dx * dx + dy * dy) / (b.2 * b.2))
                    total += weight
                    c = (c.0 + b.3.0 * weight, c.1 + b.3.1 * weight, c.2 + b.3.2 * weight)
                }
                set(x, y, (c.0 / total, c.1 / total, c.2 / total))
            }
        }
    }
    func bands() {
        let palette = (0..<(2 + Int(unit() * 3))).map { _ in vivid() }
        let fx = 0.02 + unit() * 0.08, fy = 0.02 + unit() * 0.08, warp = unit() * 3, freq = 0.02 + unit() * 0.1
        for y in 0..<h {
            for x in 0..<w {
                let u = sin(Float(x) * fx + warp * sin(Float(y) * freq)) * 0.5 + sin(Float(y) * fy + warp * sin(Float(x) * freq)) * 0.5
                let t = (u * 0.5 + 0.5) * Float(palette.count - 1)
                let i = min(palette.count - 2, max(0, Int(t)))
                let f = min(1, max(0, t - Float(i)))
                let a = palette[i], b = palette[i + 1]
                set(x, y, (a.0 + (b.0 - a.0) * f, a.1 + (b.1 - a.1) * f, a.2 + (b.2 - a.2) * f))
            }
        }
    }
    func glow() {
        let base = (unit() * 0.15, unit() * 0.15, unit() * 0.2)
        fill(base)
        var acc = [Float](repeating: 0, count: p.count)
        for _ in 0..<(2 + Int(unit() * 6)) {
            let c = vivid()
            let width = 0.5 + unit() * 3
            if unit() < 0.5 {
                // A glowing line: distance to the segment, Gaussian falloff.
                let x0 = unit() * Float(w), y0 = unit() * Float(h), x1 = unit() * Float(w), y1 = unit() * Float(h)
                let lx = x1 - x0, ly = y1 - y0, len2 = max(1e-3, lx * lx + ly * ly)
                for y in 0..<h {
                    for x in 0..<w {
                        let t = min(1, max(0, ((Float(x) - x0) * lx + (Float(y) - y0) * ly) / len2))
                        let dx = Float(x) - (x0 + t * lx), dy = Float(y) - (y0 + t * ly)
                        let g = exp(-(dx * dx + dy * dy) / (width * width))
                        let i = (y * w + x) * 3
                        acc[i] += c.0 * g; acc[i + 1] += c.1 * g; acc[i + 2] += c.2 * g
                    }
                }
            } else {
                // A glowing dot.
                let cx = unit() * Float(w), cy = unit() * Float(h), r = 1 + unit() * 10
                for y in 0..<h {
                    for x in 0..<w {
                        let dx = Float(x) - cx, dy = Float(y) - cy
                        let g = exp(-(dx * dx + dy * dy) / (r * r))
                        let i = (y * w + x) * 3
                        acc[i] += c.0 * g; acc[i + 1] += c.1 * g; acc[i + 2] += c.2 * g
                    }
                }
            }
        }
        for i in 0..<p.count {
            p[i] = min(1, p[i] + acc[i])
        }
    }
    switch index % 11 {
    case 8:
        blobs()
    case 9:
        bands()
    case 10:
        glow()
    case 0:
        fill(color())
    case 1:
        gradient()
    case 2:
        blocks()
    case 3:
        stripes()
    case 4:
        fill(color())
        rectangles()
    case 5:
        gradient()
        circles()
    case 6:
        pixelNoise()
    default:
        blocks()
        rectangles()
        circles()
    }
    return (try? ScreenImage(pixels: p)) ?? ScreenImage()
}

final class ResultBox: @unchecked Sendable {
    /// Per item: the original (upright, then turned by each extra rotation) followed by its augmented copies,
    /// each with the rotation applied.
    var images: [[(image: ScreenImage, rotation: Int)]]
    var failures: [String] = []
    let lock = NSLock()
    init(count: Int) {
        images = [[(image: ScreenImage, rotation: Int)]](repeating: [], count: count)
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

func write(items: [Item], to path: String, classes: ClassList, workers: Int, augment: Int, mask: [Float]?, rotations: [Int], synthetic: Int, syntheticSeed: UInt64) {
    // With a center mask, obscuring the middle is redundant and its margin would blur the edge UI the mask keeps.
    let augmentOptions: Augmenter.Options = {
        var options = Augmenter.Options()
        if mask != nil {
            options.centerObscureProbability = 0
        }
        return options
    }()
    if items.isEmpty {
        return
    }
    let box = ResultBox(count: items.count)
    let paths = items.map { $0.path }
    DispatchQueue.concurrentPerform(iterations: max(1, min(workers, items.count))) { worker in
        var i = worker
        while i < paths.count {
            let result: [(image: ScreenImage, rotation: Int)] = autoreleasepool {
                guard let loaded = try? loadRGB(path: paths[i]) else {
                    return []
                }
                // Portrait input is turned to landscape, as the reader does before classifying.
                let decoded = loaded.height <= loaded.width ? loaded : ImageRotation.rotate(width: loaded.width, height: loaded.height, rgb: loaded.rgb, degrees: 90)
                guard let original = try? ScreenNormalizer.normalize(width: decoded.width, height: decoded.height, rgb: decoded.rgb) else {
                    return []
                }
                // Turn first, then mask (the mask is applied in the orientation the classifier sees).
                func finish(_ image: ScreenImage, _ rotation: Int) -> (image: ScreenImage, rotation: Int) {
                    let turned = rotation == 180 ? image.rotated180() : image
                    return (mask.map { turned.masked($0) } ?? turned, rotation)
                }
                var out = rotations.map { finish(original, $0) }
                var k = 0
                while k < augment {
                    var augmenter = Augmenter(seed: augmentationSeed(paths[i], k), options: augmentOptions)
                    let crop = augmenter.cropRect(width: decoded.width, height: decoded.height)
                    if let cropped = try? ScreenNormalizer.normalize(
                        width: decoded.width, height: decoded.height, rgb: decoded.rgb,
                        cropX: crop.x, cropY: crop.y, cropWidth: crop.width, cropHeight: crop.height
                    ) {
                        out.append(finish(augmenter.augment(cropped), rotations[k % rotations.count]))
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
            for sample in box.images[i] {
                try writer.append(DatasetSample(label: rotatedLabel(items[i].label, sample.rotation), image: sample.image))
            }
            i += 1
        }
        var k = 0
        while k < synthetic {
            let image = syntheticNone(index: k, seed: syntheticSeed)
            try writer.append(DatasetSample(label: DatasetSample.noneLabel, image: mask.map { image.masked($0) } ?? image))
            k += 1
        }
        try writer.close()
        print("wrote \(path): \(writer.writtenCount) samples (\(items.count) images, \(augment) augmented copies each, rotations \(rotations.map(String.init).joined(separator: ",")), \(synthetic) synthetic none)")
        // Sidecar with the source path of every sample, in file order, for inspecting evaluation results.
        var meta: [[String: Any]] = []
        i = 0
        while i < items.count {
            var k = 0
            while k < box.images[i].count {
                let rotation = box.images[i][k].rotation
                meta.append(["path": items[i].path, "label": rotatedLabel(items[i].label, rotation), "augmented": k, "rotation": rotation])
                k += 1
            }
            i += 1
        }
        k = 0
        while k < synthetic {
            meta.append(["path": "synthetic:\(k)", "label": DatasetSample.noneLabel, "augmented": 0, "rotation": 0])
            k += 1
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
write(items: trainItems, to: trainPath, classes: classes, workers: numWorkers, augment: augmentCount, mask: centerMask, rotations: rotations, synthetic: syntheticNoneCount, syntheticSeed: 1)
if evalPath.isEmpty != true {
    write(items: evalItems, to: evalPath, classes: classes, workers: numWorkers, augment: 0, mask: centerMask, rotations: rotations, synthetic: syntheticNoneCount / 10, syntheticSeed: 2)
}
print("done (\(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - start)) s)")
