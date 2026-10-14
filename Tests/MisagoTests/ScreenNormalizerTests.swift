import XCTest
@testable import Misago

final class ScreenNormalizerTests: XCTestCase {
    private func solid(width: Int, height: Int, r: UInt8, g: UInt8, b: UInt8) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: width * height * 3)
        var i = 0
        while i < width * height {
            out[i * 3] = r
            out[i * 3 + 1] = g
            out[i * 3 + 2] = b
            i += 1
        }
        return out
    }

    func testSolidColorSurvivesShrinking() throws {
        let image = try ScreenNormalizer.normalize(width: 1280, height: 720, rgb: solid(width: 1280, height: 720, r: 255, g: 128, b: 0))
        XCTAssertEqual(image.pixels.count, ScreenImage.pixelCount)
        var i = 0
        while i < ScreenImage.width * ScreenImage.height {
            XCTAssertEqual(image.pixels[i * 3], 1.0, accuracy: 1e-5)
            XCTAssertEqual(image.pixels[i * 3 + 1], 128.0 / 255.0, accuracy: 1e-5)
            XCTAssertEqual(image.pixels[i * 3 + 2], 0.0, accuracy: 1e-5)
            i += 1
        }
    }

    func testSolidColorSurvivesEnlarging() throws {
        let image = try ScreenNormalizer.normalize(width: 64, height: 36, rgb: solid(width: 64, height: 36, r: 10, g: 20, b: 30))
        XCTAssertEqual(image.pixels[0], 10.0 / 255.0, accuracy: 1e-5)
        XCTAssertEqual(image.pixels[ScreenImage.pixelCount - 1], 30.0 / 255.0, accuracy: 1e-5)
    }

    /// A 4:3 input gets black bars on the left and right; the middle keeps its color.
    func testPillarboxForNarrowInput() throws {
        let image = try ScreenNormalizer.normalize(width: 800, height: 600, rgb: solid(width: 800, height: 600, r: 255, g: 255, b: 255))
        // Canvas is 800x600 -> 16:9 canvas 1066.7x600, so 12.5% on each side is black: columns 0..<16 and 112..<128.
        let midRow = ScreenImage.height / 2
        XCTAssertEqual(image.pixels[(midRow * ScreenImage.width + 0) * 3], 0.0, accuracy: 1e-5)
        XCTAssertEqual(image.pixels[(midRow * ScreenImage.width + 64) * 3], 1.0, accuracy: 1e-5)
        XCTAssertEqual(image.pixels[(midRow * ScreenImage.width + 127) * 3], 0.0, accuracy: 1e-5)
        XCTAssertEqual(image.pixels[(midRow * ScreenImage.width + 20) * 3], 1.0, accuracy: 1e-5)
    }

    /// Row 0 of the input lands on row 0 of the output (top stays top).
    func testOrientationIsKept() throws {
        var rgb = solid(width: 256, height: 144, r: 0, g: 0, b: 0)
        var x = 0
        while x < 256 {
            rgb[x * 3] = 255  // first row red
            x += 1
        }
        let image = try ScreenNormalizer.normalize(width: 256, height: 144, rgb: rgb)
        XCTAssertEqual(image.pixels[0], 0.5, accuracy: 1e-4)  // 2x2 block average: one red row of two
        XCTAssertEqual(image.pixels[(ScreenImage.height - 1) * ScreenImage.width * 3], 0.0, accuracy: 1e-5)
    }

    func testInvalidInputThrows() {
        XCTAssertThrowsError(try ScreenNormalizer.normalize(width: 0, height: 10, rgb: [UInt8]()))
        XCTAssertThrowsError(try ScreenNormalizer.normalize(width: 2, height: 2, rgb: [UInt8](repeating: 0, count: 5)))
    }

    func testBytesRoundTrip() throws {
        let original = try ScreenNormalizer.normalize(width: 128, height: 72, rgb: (0..<(128 * 72 * 3)).map { UInt8($0 % 251) } as [UInt8])
        let restored = try ScreenImage(bytes: original.bytes)
        XCTAssertEqual(original, restored)
    }
}
