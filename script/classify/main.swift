import Foundation
import Misago

setbuf(stdout, nil)

// Classifies image files with trained weights.
// Usage: classify --weights <model.json> <image>...

var weightsPath = ""
var imagePaths: [String] = []
var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    let arg = args[argIdx]
    switch arg {
    case "-h", "--help":
        print("Usage: classify --weights <model.json> <image>...")
        exit(0)
    case "-w", "--weights":
        if (argIdx + 1) < args.count {
            argIdx += 1
            weightsPath = args[argIdx]
        }
    default:
        if arg.hasPrefix("-") {
            print("error: unknown argument \(arg)")
            exit(1)
        }
        imagePaths.append(arg)
    }
    argIdx += 1
}
if weightsPath.isEmpty || imagePaths.isEmpty {
    print("error: specify --weights <model.json> and at least one image. See classify --help.")
    exit(1)
}
let weights: HawkWeights
do {
    weights = try HawkWeights.load(from: URL(fileURLWithPath: weightsPath))
} catch {
    print("error: cannot read weights \(weightsPath): \(error)")
    exit(1)
}
let classifier: ImageClassifier
do {
    classifier = try ImageClassifier(weights: weights)
} catch {
    print("error: invalid class list in weights: \(error)")
    exit(1)
}
print("weights: \(weightsPath) (\(weights.parameterCount) parameters, classes \(weights.classNames.joined(separator: ", ")), calibration scale \(String(format: "%.3f", weights.calibrationScale)))")
for path in imagePaths {
    do {
        let decoded = try loadRGB(path: path)
        let image = try ScreenNormalizer.normalize(width: decoded.width, height: decoded.height, rgb: decoded.rgb)
        let start = CFAbsoluteTimeGetCurrent()
        let verdict = classifier.classify(image)
        let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        let distribution = verdict.probabilities.enumerated().map { "\(classifier.classes[$0.offset]) \(String(format: "%.3f", $0.element))" }.joined(separator: ", ")
        let titles = verdict.titleProbabilities.sorted { $0.key < $1.key }.map { "\($0.key)=\(String(format: "%.3f", $0.value))" }.joined(separator: " ")
        print("\(verdict.className) p=\(String(format: "%.4f", verdict.probability)) [\(distribution)] titles \(titles.isEmpty ? "-" : titles) \(String(format: "%.2f", ms)) ms  \(path) (\(decoded.width)x\(decoded.height))")
    } catch {
        print("error  \(path): \(error)")
    }
}
