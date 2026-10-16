import XCTest
@testable import Misago

final class AugmenterTests: XCTestCase {
    private func sample() -> ScreenImage {
        return try! ScreenImage(bytes: (0..<ScreenImage.pixelCount).map { UInt8(($0 * 13) % 256) })
    }

    func testDeterministicAndBounded() {
        var a = Augmenter(seed: 42)
        var b = Augmenter(seed: 42)
        let image = sample()
        let x = a.augment(image)
        let y = b.augment(image)
        XCTAssertEqual(x, y)
        XCTAssertNotEqual(x, image)
        XCTAssertTrue(x.pixels.allSatisfy { 0.0 <= $0 && $0 <= 1.0 })
        var c = Augmenter(seed: 43)
        XCTAssertNotEqual(c.augment(image), x)
    }

    func testCropRectStaysInside() {
        var a = Augmenter(seed: 1)
        var i = 0
        while i < 200 {
            let r = a.cropRect(width: 2556, height: 1179)
            XCTAssertGreaterThanOrEqual(r.x, 0)
            XCTAssertGreaterThanOrEqual(r.y, 0)
            XCTAssertLessThanOrEqual(r.x + r.width, 2556)
            XCTAssertLessThanOrEqual(r.y + r.height, 1179)
            XCTAssertGreaterThanOrEqual(Float(r.width), 2556 * 0.85 - 1)
            XCTAssertGreaterThanOrEqual(Float(r.height), 1179 * 0.85 - 1)
            i += 1
        }
    }

    func testCropNormalizeMatchesFullImageOnFullRect() throws {
        let w = 256
        let h = 144
        let rgb: [UInt8] = (0..<(w * h * 3)).map { UInt8($0 % 251) }
        let full = try ScreenNormalizer.normalize(width: w, height: h, rgb: rgb)
        let cropped = try ScreenNormalizer.normalize(width: w, height: h, rgb: rgb, cropX: 0, cropY: 0, cropWidth: w, cropHeight: h)
        XCTAssertEqual(full, cropped)
        // A crop of the left half: the first output column comes from input column 0 and the last from column 127.
        let half = try ScreenNormalizer.normalize(width: w, height: h, rgb: rgb, cropX: 0, cropY: 0, cropWidth: 128, cropHeight: 72)
        XCTAssertEqual(half.pixels[0], Float(rgb[0]) / 255.0, accuracy: 1e-5)
        XCTAssertNotEqual(half, full)
    }
}
