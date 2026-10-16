import Foundation

/// Result of the whole pipeline on one screenshot.
public struct ScreenReading: Sendable, Equatable {
    public let verdict: ImageVerdict
    /// The number read from the class's field, or nil when the class has no field or nothing was read.
    public let reading: DigitReading?

    /// The number for callers that want a single integer: −1 when the screen is not recognized or no number was read.
    public var number: Int {
        return reading?.value ?? -1
    }
}

/// Classifier → crop region of the winning class → digit reader.
public final class ScreenReader: @unchecked Sendable {
    public let classifier: ImageClassifier
    public let digitReader: DigitReader
    public let regions: RegionTable

    public init(classifier: ImageClassifier, digitReader: DigitReader, regions: RegionTable) {
        self.classifier = classifier
        self.digitReader = digitReader
        self.regions = regions
    }

    /// `rgb` is the full-resolution screenshot (interleaved RGB bytes, row 0 at the top).
    public func read(width: Int, height: Int, rgb: [UInt8]) throws -> ScreenReading {
        let image = try ScreenNormalizer.normalize(width: width, height: height, rgb: rgb)
        let verdict = classifier.classify(image)
        guard verdict.isOther != true, let region = regions.regions[verdict.className] else {
            return ScreenReading(verdict: verdict, reading: nil)
        }
        let strip = try StripNormalizer.extract(width: width, height: height, rgb: rgb, region: region)
        return ScreenReading(verdict: verdict, reading: digitReader.read(strip))
    }
}
