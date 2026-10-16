import XCTest
@testable import Misago

final class DigitTests: XCTestCase {
    func testVocabulary() {
        XCTAssertEqual(DigitVocabulary.size, 11)
        XCTAssertEqual(DigitVocabulary.ids(of: "1604"), [2, 7, 1, 5])
        XCTAssertEqual(DigitVocabulary.string(of: [2, 7, 1, 5]), "1604")
        XCTAssertEqual(DigitVocabulary.ids(of: ""), [])
        XCTAssertNil(DigitVocabulary.ids(of: "12a"))
        XCTAssertNil(DigitVocabulary.ids(of: "１２"))
    }

    func testGreedyDecodeCollapsesRepeatsAndBlanks() {
        func frame(_ best: Int) -> [Float] {
            var f = [Float](repeating: 0.0, count: DigitVocabulary.size)
            f[best] = 5.0
            return f
        }
        // blank 1 1 blank 1 2 2 blank blank 0 → "112" then "0" → "1120"
        let logits = [frame(0), frame(2), frame(2), frame(0), frame(2), frame(3), frame(3), frame(0), frame(0), frame(1)]
        let reading = DigitReader.greedyDecode(logits: logits)
        XCTAssertEqual(reading.digits, "1120")
        XCTAssertEqual(reading.value, 1120)
        XCTAssertGreaterThan(reading.confidence, 0.9)
        XCTAssertEqual(DigitReader.greedyDecode(logits: [frame(0), frame(0)]).digits, "")
        XCTAssertNil(DigitReader.greedyDecode(logits: [frame(0)]).value)
    }

    func testStemShapes() {
        XCTAssertEqual(ConvStemWeights.outputRows, 28)
        XCTAssertEqual(ConvStemWeights.outputColumns, 160)
        XCTAssertEqual(ConvStemWeights.frameDim, 448)
        // A zero stem maps every strip to GeLU(bias) frames.
        let weights = ConvStemWeights(weight: [Float](repeating: 0.0, count: 3 * 3 * 3 * 16), bias: [Float](repeating: 1.0, count: 16))
        let frames = ConvStem.frames(DigitStrip().pixels, weights: weights)
        XCTAssertEqual(frames.count, 160 * 448)
        XCTAssertEqual(frames[0], 1.0 / (1.0 + expf(-1.702)), accuracy: 1e-5)
    }

    /// A 4:1 crop fills the left 224 columns at full height; the rest stays black.
    func testStripExtractKeepsAspectAndPads() throws {
        let w = 1000
        let h = 500
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        var y = 100
        while y < 200 {
            var x = 100
            while x < 500 {
                let i = (y * w + x) * 3
                rgb[i] = 255
                rgb[i + 1] = 255
                rgb[i + 2] = 255
                x += 1
            }
            y += 1
        }
        let strip = try StripNormalizer.extract(width: w, height: h, rgb: rgb, x: 100, y: 100, cropWidth: 400, cropHeight: 100)
        XCTAssertEqual(strip.pixels[(10 * DigitStrip.width + 10) * 3], 1.0, accuracy: 1e-4)
        XCTAssertEqual(strip.pixels[(55 * DigitStrip.width + 223) * 3], 1.0, accuracy: 1e-4)
        XCTAssertEqual(strip.pixels[(10 * DigitStrip.width + 232) * 3], 0.0)
        // A region in height units lands on the same pixels.
        let region = CropRegion(anchor: .left, x0: 0.2, x1: 1.0, y0: 0.2, y1: 0.4)
        let r = region.pixelRect(width: w, height: h)
        XCTAssertEqual(r.x, 100)
        XCTAssertEqual(r.y, 100)
        XCTAssertEqual(r.width, 400)
        XCTAssertEqual(r.height, 100)
        XCTAssertEqual(try StripNormalizer.extract(width: w, height: h, rgb: rgb, region: region), strip)
        // Center and right anchors.
        XCTAssertEqual(CropRegion(anchor: .center, x0: -0.1, x1: 0.1, y0: 0.0, y1: 0.1).pixelRect(width: w, height: h).x, 450)
        XCTAssertEqual(CropRegion(anchor: .right, x0: -0.2, x1: 0.0, y0: 0.0, y1: 0.1).pixelRect(width: w, height: h).x, 900)
    }

    func testRegionTableRoundTrip() throws {
        let table = RegionTable(regions: ["a_home": CropRegion(anchor: .left, x0: 0.1, x1: 0.5, y0: 0.0, y1: 0.1)])
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("misago-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        try table.save(to: url)
        XCTAssertEqual(try RegionTable.load(from: url), table)
    }

    func testDigitDatasetRoundTrip() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("misago-\(UUID().uuidString).msgd")
        defer { try? FileManager.default.removeItem(at: url) }
        let strip = try DigitStrip(bytes: (0..<DigitStrip.pixelCount).map { UInt8($0 % 253) })
        let writer = try DigitDatasetWriter(url: url)
        try writer.append(DigitSample(digits: "1604", strip: strip))
        try writer.append(DigitSample(digits: "", strip: DigitStrip()))
        XCTAssertThrowsError(try writer.append(DigitSample(digits: "12345678901", strip: strip)))
        XCTAssertThrowsError(try writer.append(DigitSample(digits: "1a", strip: strip)))
        try writer.close()
        let reader = try DigitDatasetReader(url: url)
        XCTAssertEqual(reader.count, 2)
        XCTAssertEqual(reader.allDigits, ["1604", ""])
        XCTAssertEqual(try reader.sample(at: 0), DigitSample(digits: "1604", strip: strip))
        XCTAssertEqual(try reader.sample(at: 1).strip, DigitStrip())
    }

    func testStripAugmentationIsDeterministicAndBounded() throws {
        let strip = try DigitStrip(bytes: (0..<DigitStrip.pixelCount).map { UInt8(($0 * 7) % 256) })
        var a = Augmenter(seed: 5)
        var b = Augmenter(seed: 5)
        let x = a.augment(strip)
        XCTAssertEqual(x, b.augment(strip))
        XCTAssertNotEqual(x, strip)
        XCTAssertTrue(x.pixels.allSatisfy { 0.0 <= $0 && $0 <= 1.0 })
        let r = a.jitter(x: 100, y: 50, width: 400, height: 80)
        XCTAssertGreaterThan(r.width, 300)
        XCTAssertLessThan(abs(r.x - 100), 40)
    }
}
