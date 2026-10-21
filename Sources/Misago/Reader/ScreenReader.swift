import Foundation

/// Result of the whole pipeline on one screenshot.
public struct ScreenReading: Sendable, Equatable {
    public let verdict: ImageVerdict
    /// The number read from the class's field, or nil when the class has no field or nothing was read.
    public let reading: DigitReading?
    /// Clockwise rotation (0, 90, 180 or 270 degrees) that turned the input upright before classification.
    public let rotation: Int

    public init(verdict: ImageVerdict, reading: DigitReading?, rotation: Int = 0) {
        self.verdict = verdict
        self.reading = reading
        self.rotation = rotation
    }

    /// The number for callers that want a single integer: −1 when the screen is not recognized or no number was read.
    public var number: Int {
        return reading?.value ?? -1
    }

    /// 2 win, 1 loss, 0 draw, −1 not a result screen (see `ImageVerdict.outcome`).
    public var outcome: Int {
        return verdict.outcome
    }

    /// 1 during a match, 0 otherwise (see `ImageVerdict.battle`).
    public var battle: Int {
        return verdict.battle
    }
}

/// The input downscaled for the classifier, turned to landscape when it arrived in portrait. The full-resolution
/// pixels are kept as they came: only the small image and, later, the number field are turned.
public struct LandscapeInput: @unchecked Sendable {
    /// Clockwise rotation applied to make the classifier input landscape (0 for landscape input, 90 for portrait).
    public let baseRotation: Int
    public let width: Int
    public let height: Int
    public let rgb: [UInt8]
    public let image: ScreenImage
}

/// Landscape input → classifier (class and orientation in one pass) → crop region of the winning class in the
/// upright image → digit reader.
///
/// Screens may arrive sideways or upside down (streams send the device's native orientation). Portrait input is
/// turned 90° to landscape; a model trained with rotation tags (`<class>@180`) then tells whether the result is
/// upright or upside down, so a portrait screen turned the other way (270°) is handled too. Rotations touch
/// only small images: the 128 x 72 classifier input and the number field.
public final class ScreenReader: @unchecked Sendable {
    public let classifier: ImageClassifier
    public let digitReader: DigitReader
    public let regions: RegionTable

    public init(classifier: ImageClassifier, digitReader: DigitReader, regions: RegionTable) {
        self.classifier = classifier
        self.digitReader = digitReader
        self.regions = regions
    }

    /// Downscales the input, turning portrait input to landscape.
    public func landscape(width: Int, height: Int, rgb: [UInt8]) throws -> LandscapeInput {
        let base = width < height ? 90 : 0
        let image = try ScreenNormalizer.normalize(width: width, height: height, rgb: rgb, rotation: base)
        return LandscapeInput(baseRotation: base, width: width, height: height, rgb: rgb, image: image)
    }

    /// The verdict and the clockwise rotation that turns the original input upright.
    public func classify(_ input: LandscapeInput) -> (verdict: ImageVerdict, rotation: Int) {
        let verdict = classifier.classify(input.image)
        return (verdict, (input.baseRotation + verdict.rotation) % 360)
    }

    /// The digit strip of the class's field in the upright image, or nil when the class has no field. Only the
    /// field is cut out of the original pixels and turned.
    public func strip(_ input: LandscapeInput, verdict: ImageVerdict, rotation: Int) throws -> DigitStrip? {
        guard verdict.recognized, let region = regions.regions[verdict.className] else {
            return nil
        }
        let quarter = rotation % 180 != 0
        let uprightWidth = quarter ? input.height : input.width
        let uprightHeight = quarter ? input.width : input.height
        let rect = region.pixelRect(width: uprightWidth, height: uprightHeight)
        let field = ImageRotation.cropRotated(width: input.width, height: input.height, rgb: input.rgb, degrees: rotation, rect: rect)
        return try StripNormalizer.extract(width: field.width, height: field.height, rgb: field.rgb, x: 0, y: 0, cropWidth: field.width, cropHeight: field.height)
    }

    /// `rgb` is the full-resolution screenshot (interleaved RGB bytes, row 0 at the top), in any orientation.
    public func read(width: Int, height: Int, rgb: [UInt8]) throws -> ScreenReading {
        let input = try landscape(width: width, height: height, rgb: rgb)
        let result = classify(input)
        guard let strip = try strip(input, verdict: result.verdict, rotation: result.rotation) else {
            return ScreenReading(verdict: result.verdict, reading: nil, rotation: result.rotation)
        }
        return ScreenReading(verdict: result.verdict, reading: digitReader.read(strip), rotation: result.rotation)
    }
}
