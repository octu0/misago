import Foundation

/// Binary layout of a digit-strip dataset (little-endian).
///
///     magic       4 bytes  "MSGD"
///     version     u32      1
///     count       u32      number of samples
///     width       u16      320
///     height      u16      56
///     channels    u8       3
///     maxDigits   u8       8 (label field size)
///     reserved    2 bytes  0
///     samples     count times: length u8, digits maxDigits bytes (ASCII, zero padded), pixels width * height * channels bytes
public enum DigitDatasetFormat {
    public static let magic: [UInt8] = [0x4D, 0x53, 0x47, 0x44]  // "MSGD"
    public static let version: UInt32 = 1
    public static let headerSize = 20
    public static let countOffset = 8
    public static let maxDigits = 8
    public static let sampleSize = 1 + maxDigits + DigitStrip.pixelCount

    static func expectedFileSize(count: Int) -> Int {
        return headerSize + count * sampleSize
    }
}

/// The reader's output alphabet: CTC blank first, then the ten digits.
public enum DigitVocabulary {
    public static let blankId = 0
    public static let names: [String] = ["<blank>", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    public static let size = names.count

    /// Class ids of a digit string, or nil when it holds anything but ASCII digits.
    public static func ids(of digits: String) -> [Int]? {
        var out: [Int] = []
        for ch in digits {
            guard let v = ch.wholeNumberValue, ch.isASCII else {
                return nil
            }
            out.append(v + 1)
        }
        return out
    }

    public static func string(of ids: [Int]) -> String {
        var out = ""
        for id in ids where 1 <= id && id <= 10 {
            out.append(Character(String(id - 1)))
        }
        return out
    }
}

public struct DigitSample: Sendable, Equatable {
    public let digits: String
    public let strip: DigitStrip

    public init(digits: String, strip: DigitStrip) {
        self.digits = digits
        self.strip = strip
    }
}

public final class DigitDatasetWriter {
    private let handle: FileHandle
    private var count: Int = 0
    private var closed = false

    public init(url: URL) throws {
        if FileManager.default.createFile(atPath: url.path, contents: nil) != true {
            throw DatasetFileError.ioFailure("cannot create \(url.path)")
        }
        do {
            handle = try FileHandle(forWritingTo: url)
        } catch {
            throw DatasetFileError.ioFailure("cannot open \(url.path): \(error)")
        }
        var header = Data()
        header.append(contentsOf: DigitDatasetFormat.magic)
        header.append(contentsOf: DatasetWriter.le32(DigitDatasetFormat.version))
        header.append(contentsOf: DatasetWriter.le32(0))
        header.append(contentsOf: DatasetWriter.le16(UInt16(DigitStrip.width)))
        header.append(contentsOf: DatasetWriter.le16(UInt16(DigitStrip.height)))
        header.append(UInt8(DigitStrip.channels))
        header.append(UInt8(DigitDatasetFormat.maxDigits))
        header.append(contentsOf: [0, 0])
        handle.write(header)
    }

    /// `digits` may be empty (a field with no number); it must be ASCII digits only and at most `maxDigits` long.
    public func append(_ sample: DigitSample) throws {
        guard DigitVocabulary.ids(of: sample.digits) != nil, sample.digits.count <= DigitDatasetFormat.maxDigits else {
            throw DatasetFileError.invalidLabel(sample.digits.count)
        }
        var data = Data(capacity: DigitDatasetFormat.sampleSize)
        data.append(UInt8(sample.digits.count))
        var field = [UInt8](repeating: 0, count: DigitDatasetFormat.maxDigits)
        var i = 0
        for byte in sample.digits.utf8 {
            field[i] = byte
            i += 1
        }
        data.append(contentsOf: field)
        data.append(contentsOf: sample.strip.bytes)
        handle.write(data)
        count += 1
    }

    public var writtenCount: Int {
        return count
    }

    public func close() throws {
        if closed {
            return
        }
        closed = true
        handle.seek(toFileOffset: UInt64(DigitDatasetFormat.countOffset))
        handle.write(Data(DatasetWriter.le32(UInt32(count))))
        do {
            try handle.close()
        } catch {
            throw DatasetFileError.ioFailure("cannot close: \(error)")
        }
    }
}

public final class DigitDatasetReader: @unchecked Sendable {
    private let data: Data
    public let count: Int

    public init(url: URL) throws {
        do {
            data = try Data(contentsOf: url, options: .mappedIfSafe)
        } catch {
            throw DatasetFileError.ioFailure("cannot read \(url.path): \(error)")
        }
        if data.count < DigitDatasetFormat.headerSize {
            throw DatasetFileError.truncatedHeader(actualBytes: data.count)
        }
        let magic = Array(data[0..<4])
        if magic != DigitDatasetFormat.magic {
            throw DatasetFileError.invalidMagic(magic)
        }
        let version = UInt32(data[4]) | (UInt32(data[5]) << 8) | (UInt32(data[6]) << 16) | (UInt32(data[7]) << 24)
        if version != DigitDatasetFormat.version {
            throw DatasetFileError.unsupportedVersion(version)
        }
        let count = Int(UInt32(data[8]) | (UInt32(data[9]) << 8) | (UInt32(data[10]) << 16) | (UInt32(data[11]) << 24))
        let width = Int(UInt16(data[12]) | (UInt16(data[13]) << 8))
        let height = Int(UInt16(data[14]) | (UInt16(data[15]) << 8))
        let channels = Int(data[16])
        let maxDigits = Int(data[17])
        if width != DigitStrip.width || height != DigitStrip.height || channels != DigitStrip.channels || maxDigits != DigitDatasetFormat.maxDigits {
            throw DatasetFileError.unsupportedShape(width: width, height: height, channels: channels)
        }
        let expected = DigitDatasetFormat.expectedFileSize(count: count)
        if data.count != expected {
            throw DatasetFileError.lengthMismatch(expected: expected, actual: data.count)
        }
        var i = 0
        while i < count {
            let offset = DigitDatasetFormat.headerSize + i * DigitDatasetFormat.sampleSize
            let length = Int(data[offset])
            if DigitDatasetFormat.maxDigits < length {
                throw DatasetFileError.invalidLabel(length)
            }
            var k = 0
            while k < length {
                let byte = data[offset + 1 + k]
                if byte < 0x30 || 0x39 < byte {
                    throw DatasetFileError.invalidLabel(Int(byte))
                }
                k += 1
            }
            i += 1
        }
        self.count = count
    }

    public func digits(at index: Int) throws -> String {
        if index < 0 || count <= index {
            throw DatasetFileError.indexOutOfRange(index: index, count: count)
        }
        let offset = DigitDatasetFormat.headerSize + index * DigitDatasetFormat.sampleSize
        let length = Int(data[offset])
        return String(decoding: data[(offset + 1)..<(offset + 1 + length)], as: UTF8.self)
    }

    public func sample(at index: Int) throws -> DigitSample {
        let digits = try digits(at: index)
        let offset = DigitDatasetFormat.headerSize + index * DigitDatasetFormat.sampleSize + 1 + DigitDatasetFormat.maxDigits
        let bytes = Array(data[offset..<(offset + DigitStrip.pixelCount)])
        return DigitSample(digits: digits, strip: try DigitStrip(bytes: bytes))
    }

    public var allDigits: [String] {
        var out: [String] = []
        out.reserveCapacity(count)
        var i = 0
        while i < count {
            out.append((try? digits(at: i)) ?? "")
            i += 1
        }
        return out
    }
}
