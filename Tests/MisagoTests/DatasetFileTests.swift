import XCTest
@testable import Misago

final class DatasetFileTests: XCTestCase {
    private func temporaryURL() -> URL {
        return FileManager.default.temporaryDirectory.appendingPathComponent("misago-\(UUID().uuidString).msgo")
    }

    private let classes = try! ClassList(others: ["a_home", "b_lobby"])

    private func image(seed: Int) -> ScreenImage {
        return try! ScreenImage(bytes: (0..<ScreenImage.pixelCount).map { UInt8(($0 * 31 + seed) % 256) })
    }

    func testWriteAndReadBack() throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let writer = try DatasetWriter(url: url, classes: classes)
        try writer.append(DatasetSample(label: 1, image: image(seed: 1)))
        try writer.append(DatasetSample(label: 0, image: image(seed: 2)))
        try writer.append(DatasetSample(label: 2, image: image(seed: 3)))
        try writer.close()
        let reader = try DatasetReader(url: url)
        XCTAssertEqual(reader.count, 3)
        XCTAssertEqual(reader.classes, classes)
        XCTAssertEqual(reader.labels, [1, 0, 2])
        XCTAssertEqual(reader.classCounts, [1, 1, 1])
        XCTAssertEqual(try reader.sample(at: 1), DatasetSample(label: 0, image: image(seed: 2)))
        XCTAssertEqual(try reader.sample(at: 2).image, image(seed: 3))
        XCTAssertThrowsError(try reader.sample(at: 3))
        let size = try FileManager.default.attributesOfItem(atPath: url.path)[.size] as! Int
        XCTAssertEqual(size, DatasetFormat.fixedHeaderSize + (6 + 7 + 8) + 3 * DatasetFormat.sampleSize)
    }

    func testInvalidLabelIsRejected() throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let writer = try DatasetWriter(url: url, classes: classes)
        XCTAssertThrowsError(try writer.append(DatasetSample(label: 3, image: image(seed: 0))))
        try writer.close()
        XCTAssertEqual(try DatasetReader(url: url).count, 0)
    }

    func testCorruptHeaderAndLengthAreRejected() throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try Data([0x41, 0x42, 0x43, 0x44] + [UInt8](repeating: 0, count: 16)).write(to: url)
        XCTAssertThrowsError(try DatasetReader(url: url)) { error in
            XCTAssertEqual(error as? DatasetFileError, .invalidMagic([0x41, 0x42, 0x43, 0x44]))
        }
        let writer = try DatasetWriter(url: url, classes: classes)
        try writer.append(DatasetSample(label: 0, image: image(seed: 5)))
        try writer.close()
        var data = try Data(contentsOf: url)
        data.removeLast()
        try data.write(to: url)
        let expected = DatasetFormat.fixedHeaderSize + (6 + 7 + 8) + DatasetFormat.sampleSize
        XCTAssertThrowsError(try DatasetReader(url: url)) { error in
            XCTAssertEqual(error as? DatasetFileError, .lengthMismatch(expected: expected, actual: expected - 1))
        }
    }
}
