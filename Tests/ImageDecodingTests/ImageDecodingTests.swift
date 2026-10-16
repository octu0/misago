import XCTest
@testable import ImageDecoding

final class ImageDecodingTests: XCTestCase {
    /// A 3 x 2 RGB PNG: red, green, blue on the first row and (10,20,30), (40,50,60), (70,80,90) on the second.
    private let png = Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAMAAAACCAIAAAASFvFNAAAAF0lEQVR4nGP4z8DAAMFcInIaRjZuAVEAM1kEwOQkmMsAAAAASUVORK5CYII=")!

    func testDecodesPNGTopDown() throws {
        let decoded = try ImageDecoder.decodeRGB(png)
        XCTAssertEqual(decoded.width, 3)
        XCTAssertEqual(decoded.height, 2)
        XCTAssertEqual(decoded.rgb.count, 18)
        XCTAssertEqual(Array(decoded.rgb[0..<9]), [255, 0, 0, 0, 255, 0, 0, 0, 255])
        XCTAssertEqual(Array(decoded.rgb[9..<18]), [10, 20, 30, 40, 50, 60, 70, 80, 90])
    }

    func testRejectsGarbage() {
        XCTAssertThrowsError(try ImageDecoder.decodeRGB(Data([0, 1, 2, 3, 4, 5])))
        XCTAssertThrowsError(try ImageDecoder.decodeRGB(Data()))
    }
}
