import XCTest
@testable import Misago

final class ImageRotationTests: XCTestCase {
    /// 3 x 2 image whose pixel (x, y) has red = 10 * y + x.
    private func marked(width: Int, height: Int) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: width * height * 3)
        var y = 0
        while y < height {
            var x = 0
            while x < width {
                out[(y * width + x) * 3] = UInt8(10 * y + x)
                out[(y * width + x) * 3 + 1] = 1
                out[(y * width + x) * 3 + 2] = 2
                x += 1
            }
            y += 1
        }
        return out
    }

    private func red(_ image: (width: Int, height: Int, rgb: [UInt8]), _ x: Int, _ y: Int) -> UInt8 {
        return image.rgb[(y * image.width + x) * 3]
    }

    func testClockwiseQuarterTurnMovesTopLeftToTopRight() {
        let r = ImageRotation.rotate(width: 3, height: 2, rgb: marked(width: 3, height: 2), degrees: 90)
        XCTAssertEqual(r.width, 2)
        XCTAssertEqual(r.height, 3)
        XCTAssertEqual(red(r, 1, 0), 0)    // source top-left
        XCTAssertEqual(red(r, 1, 2), 2)    // source top-right
        XCTAssertEqual(red(r, 0, 0), 10)   // source bottom-left
        XCTAssertEqual(r.rgb[1], 1)
        XCTAssertEqual(r.rgb[2], 2)
    }

    func testHalfTurnMovesTopLeftToBottomRight() {
        let r = ImageRotation.rotate(width: 3, height: 2, rgb: marked(width: 3, height: 2), degrees: 180)
        XCTAssertEqual(r.width, 3)
        XCTAssertEqual(r.height, 2)
        XCTAssertEqual(red(r, 2, 1), 0)
        XCTAssertEqual(red(r, 0, 0), 12)
    }

    func testOppositeTurnsRestoreTheImage() {
        let source = marked(width: 5, height: 3)
        for degrees in [90, 180, 270] {
            let turned = ImageRotation.rotate(width: 5, height: 3, rgb: source, degrees: degrees)
            let back = ImageRotation.rotate(width: turned.width, height: turned.height, rgb: turned.rgb, degrees: 360 - degrees)
            XCTAssertEqual(back.width, 5)
            XCTAssertEqual(back.height, 3)
            XCTAssertEqual(back.rgb, source, "\(degrees)")
        }
        XCTAssertEqual(ImageRotation.rotate(width: 5, height: 3, rgb: source, degrees: -90).rgb,
                       ImageRotation.rotate(width: 5, height: 3, rgb: source, degrees: 270).rgb)
    }

    func testScreenImageHalfTurnMatchesFullResolutionHalfTurn() throws {
        var bytes = [UInt8](repeating: 0, count: ScreenImage.pixelCount)
        var i = 0
        while i < bytes.count {
            bytes[i] = UInt8(i % 251)
            i += 1
        }
        let image = try ScreenImage(bytes: bytes)
        let turned = ImageRotation.rotate(width: ScreenImage.width, height: ScreenImage.height, rgb: bytes, degrees: 180)
        XCTAssertEqual(image.rotated180(), try ScreenImage(bytes: turned.rgb))
        XCTAssertEqual(image.rotated180().rotated180(), image)
    }

    private func noise(width: Int, height: Int) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: width * height * 3)
        var state: UInt64 = 12345
        var i = 0
        while i < out.count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            out[i] = UInt8(truncatingIfNeeded: state >> 56)
            i += 1
        }
        return out
    }

    func testCropRotatedEqualsRotatingTheWholeImage() {
        let w = 37, h = 23
        let rgb = noise(width: w, height: h)
        for degrees in [0, 90, 180, 270] {
            let full = ImageRotation.rotate(width: w, height: h, rgb: rgb, degrees: degrees)
            let rect = (x: 3, y: 5, width: min(11, full.width - 3), height: min(7, full.height - 5))
            var want = [UInt8]()
            var y = rect.y
            while y < rect.y + rect.height {
                let start = (y * full.width + rect.x) * 3
                want.append(contentsOf: full.rgb[start..<(start + rect.width * 3)])
                y += 1
            }
            let got = ImageRotation.cropRotated(width: w, height: h, rgb: rgb, degrees: degrees, rect: rect)
            XCTAssertEqual(got.width, rect.width, "\(degrees)")
            XCTAssertEqual(got.height, rect.height, "\(degrees)")
            XCTAssertEqual(got.rgb, want, "\(degrees)")
        }
    }

    func testNormalizingThenTurningMatchesTurningThenNormalizing() throws {
        let w = 300, h = 640
        let rgb = noise(width: w, height: h)
        for degrees in [90, 270] {
            let turned = ImageRotation.rotate(width: w, height: h, rgb: rgb, degrees: degrees)
            let want = try ScreenNormalizer.normalize(width: turned.width, height: turned.height, rgb: turned.rgb)
            let got = try ScreenNormalizer.normalize(width: w, height: h, rgb: rgb, rotation: degrees)
            var worst: Float = 0
            var i = 0
            while i < want.pixels.count {
                worst = max(worst, abs(want.pixels[i] - got.pixels[i]))
                i += 1
            }
            XCTAssertLessThan(worst, 1e-4, "\(degrees)")
        }
        let landscape = try ScreenNormalizer.normalize(width: h, height: w, rgb: rgb, rotation: 180)
        let reference = try ScreenNormalizer.normalize(width: h, height: w, rgb: rgb).rotated180()
        XCTAssertEqual(landscape, reference)
    }
}
