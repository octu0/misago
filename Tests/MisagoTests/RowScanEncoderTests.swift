import XCTest
@testable import Misago

final class RowScanEncoderTests: XCTestCase {
    func testShape() {
        XCTAssertEqual(RowScanEncoder.frameCount, 36)
        XCTAssertEqual(RowScanEncoder.frameDim, 768)
        XCTAssertEqual(RowScanEncoder.frameCount * RowScanEncoder.frameDim, ScreenImage.pixelCount)
    }

    func testFramesFollowRows() throws {
        var pixels = [Float](repeating: 0.0, count: ScreenImage.pixelCount)
        // Mark pixel (x = 5, y = 7, channel 1).
        pixels[(7 * ScreenImage.width + 5) * 3 + 1] = 0.75
        let image = try ScreenImage(pixels: pixels)
        let frames = RowScanEncoder.encode(image)
        XCTAssertEqual(frames.count, RowScanEncoder.frameCount)
        XCTAssertEqual(frames[0].count, RowScanEncoder.frameDim)
        // Row 7 is the second row of frame 3.
        let index = (1 * ScreenImage.width + 5) * 3 + 1
        XCTAssertEqual(frames[3][index], 0.75)
        XCTAssertEqual(frames[3].reduce(0.0, +), 0.75, accuracy: 1e-6)
        XCTAssertEqual(frames[2].reduce(0.0, +), 0.0)
        XCTAssertEqual(RowScanEncoder.encodeFlat(image), frames.flatMap { $0 })
    }
}
