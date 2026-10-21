import Foundation

public enum DatasetFileError: Error, Equatable {
    case invalidMagic([UInt8])
    case unsupportedVersion(UInt32)
    case unsupportedShape(width: Int, height: Int, channels: Int)
    case truncatedHeader(actualBytes: Int)
    case lengthMismatch(expected: Int, actual: Int)
    case indexOutOfRange(index: Int, count: Int)
    case invalidLabel(Int)
    case invalidClassList(String)
    case ioFailure(String)
}

/// Binary layout of a dataset file (little-endian).
///
///     magic       4 bytes  "MSGO"
///     version     u32      1
///     count       u32      number of samples
///     width       u16      128
///     height      u16      72
///     channels    u8       3
///     classCount  u8       number of classes (class 0 is "other")
///     reserved    2 bytes  0
///     classes     classCount times: length u8, name (UTF-8)
///     samples     count times: label u16, pixels width * height * channels bytes (row-major, interleaved RGB)
public enum DatasetFormat {
    public static let magic: [UInt8] = [0x4D, 0x53, 0x47, 0x4F]  // "MSGO"
    public static let version: UInt32 = 1
    /// magic 4 + version 4 + count 4 + width 2 + height 2 + channels 1 + classCount 1 + reserved 2
    public static let fixedHeaderSize = 20
    public static let countOffset = 8
    public static let sampleSize = 2 + ScreenImage.pixelCount
    /// Stored label of a sample that belongs to no class.
    public static let noneLabelValue: UInt16 = 0xFFFF

    static func classTableSize(_ classes: ClassList) -> Int {
        var n = 0
        for name in classes.names {
            n += 1 + name.utf8.count
        }
        return n
    }

    static func headerSize(_ classes: ClassList) -> Int {
        return fixedHeaderSize + classTableSize(classes)
    }

    static func expectedFileSize(classes: ClassList, count: Int) -> Int {
        return headerSize(classes) + count * sampleSize
    }
}

public struct DatasetSample: Sendable, Equatable {
    /// Label of a sample that belongs to no class and no title (an image of something else entirely).
    public static let noneLabel = -1

    public let label: Int
    public let image: ScreenImage

    public init(label: Int, image: ScreenImage) {
        self.label = label
        self.image = image
    }
}

/// Appending writer. The header is written with count = 0 and the real count is written back on `close()`,
/// so the prepare tool can stream samples without holding them all.
public final class DatasetWriter {
    public let classes: ClassList
    private let handle: FileHandle
    private var count: Int = 0
    private var closed = false

    public init(url: URL, classes: ClassList) throws {
        self.classes = classes
        let manager = FileManager.default
        if manager.createFile(atPath: url.path, contents: nil) != true {
            throw DatasetFileError.ioFailure("cannot create \(url.path)")
        }
        do {
            handle = try FileHandle(forWritingTo: url)
        } catch {
            throw DatasetFileError.ioFailure("cannot open \(url.path): \(error)")
        }
        var header = Data()
        header.append(contentsOf: DatasetFormat.magic)
        header.append(contentsOf: Self.le32(DatasetFormat.version))
        header.append(contentsOf: Self.le32(0))
        header.append(contentsOf: Self.le16(UInt16(ScreenImage.width)))
        header.append(contentsOf: Self.le16(UInt16(ScreenImage.height)))
        header.append(UInt8(ScreenImage.channels))
        header.append(UInt8(classes.count))
        header.append(contentsOf: [0, 0])
        for name in classes.names {
            let bytes = Array(name.utf8)
            header.append(UInt8(bytes.count))
            header.append(contentsOf: bytes)
        }
        handle.write(header)
    }

    public func append(_ sample: DatasetSample) throws {
        if sample.label < DatasetSample.noneLabel || classes.count <= sample.label {
            throw DatasetFileError.invalidLabel(sample.label)
        }
        var data = Data(capacity: DatasetFormat.sampleSize)
        data.append(contentsOf: Self.le16(sample.label == DatasetSample.noneLabel ? DatasetFormat.noneLabelValue : UInt16(sample.label)))
        data.append(contentsOf: sample.image.bytes)
        handle.write(data)
        count += 1
    }

    public var writtenCount: Int {
        return count
    }

    /// Writes the sample count into the header and closes the file.
    public func close() throws {
        if closed {
            return
        }
        closed = true
        handle.seek(toFileOffset: UInt64(DatasetFormat.countOffset))
        handle.write(Data(Self.le32(UInt32(count))))
        do {
            try handle.close()
        } catch {
            throw DatasetFileError.ioFailure("cannot close: \(error)")
        }
    }

    static func le32(_ v: UInt32) -> [UInt8] {
        return [UInt8(v & 0xFF), UInt8((v >> 8) & 0xFF), UInt8((v >> 16) & 0xFF), UInt8((v >> 24) & 0xFF)]
    }

    static func le16(_ v: UInt16) -> [UInt8] {
        return [UInt8(v & 0xFF), UInt8((v >> 8) & 0xFF)]
    }
}

/// Whole-file reader. The header, the class table, the file length and every label are validated up front.
public final class DatasetReader: @unchecked Sendable {
    private let data: Data
    private let samplesOffset: Int
    public let classes: ClassList
    public let count: Int

    public init(url: URL) throws {
        do {
            data = try Data(contentsOf: url, options: .mappedIfSafe)
        } catch {
            throw DatasetFileError.ioFailure("cannot read \(url.path): \(error)")
        }
        if data.count < DatasetFormat.fixedHeaderSize {
            throw DatasetFileError.truncatedHeader(actualBytes: data.count)
        }
        let magic = Array(data[0..<4])
        if magic != DatasetFormat.magic {
            throw DatasetFileError.invalidMagic(magic)
        }
        let version = Self.read32(data, 4)
        if version != DatasetFormat.version {
            throw DatasetFileError.unsupportedVersion(version)
        }
        let count = Int(Self.read32(data, DatasetFormat.countOffset))
        let width = Int(Self.read16(data, 12))
        let height = Int(Self.read16(data, 14))
        let channels = Int(data[16])
        let classCount = Int(data[17])
        if width != ScreenImage.width || height != ScreenImage.height || channels != ScreenImage.channels {
            throw DatasetFileError.unsupportedShape(width: width, height: height, channels: channels)
        }
        var names: [String] = []
        var offset = DatasetFormat.fixedHeaderSize
        var c = 0
        while c < classCount {
            if data.count <= offset {
                throw DatasetFileError.truncatedHeader(actualBytes: data.count)
            }
            let length = Int(data[offset])
            offset += 1
            if data.count < offset + length {
                throw DatasetFileError.truncatedHeader(actualBytes: data.count)
            }
            guard let name = String(data: data[offset..<(offset + length)], encoding: .utf8) else {
                throw DatasetFileError.invalidClassList("class \(c) is not UTF-8")
            }
            names.append(name)
            offset += length
            c += 1
        }
        do {
            classes = try ClassList(names: names)
        } catch {
            throw DatasetFileError.invalidClassList("\(error)")
        }
        samplesOffset = offset
        let expected = DatasetFormat.expectedFileSize(classes: classes, count: count)
        if data.count != expected {
            throw DatasetFileError.lengthMismatch(expected: expected, actual: data.count)
        }
        var i = 0
        while i < count {
            let label = Self.decodeLabel(Self.read16(data, offset + i * DatasetFormat.sampleSize))
            if classes.count <= label {
                throw DatasetFileError.invalidLabel(label)
            }
            i += 1
        }
        self.count = count
    }

    public func label(at index: Int) throws -> Int {
        if index < 0 || count <= index {
            throw DatasetFileError.indexOutOfRange(index: index, count: count)
        }
        return Self.decodeLabel(Self.read16(data, samplesOffset + index * DatasetFormat.sampleSize))
    }

    public func sample(at index: Int) throws -> DatasetSample {
        if index < 0 || count <= index {
            throw DatasetFileError.indexOutOfRange(index: index, count: count)
        }
        let offset = samplesOffset + index * DatasetFormat.sampleSize
        let label = Self.decodeLabel(Self.read16(data, offset))
        let bytes = Array(data[(offset + 2)..<(offset + DatasetFormat.sampleSize)])
        return DatasetSample(label: label, image: try ScreenImage(bytes: bytes))
    }

    /// All labels, in file order.
    public var labels: [Int] {
        var out: [Int] = []
        out.reserveCapacity(count)
        var i = 0
        while i < count {
            out.append(Self.decodeLabel(Self.read16(data, samplesOffset + i * DatasetFormat.sampleSize)))
            i += 1
        }
        return out
    }

    /// Number of samples per class (samples labeled none are not counted; see `noneCount`).
    public var classCounts: [Int] {
        var counts = [Int](repeating: 0, count: classes.count)
        for label in labels where 0 <= label {
            counts[label] += 1
        }
        return counts
    }

    /// Number of samples that belong to no class.
    public var noneCount: Int {
        return labels.filter { $0 == DatasetSample.noneLabel }.count
    }

    private static func decodeLabel(_ value: UInt16) -> Int {
        return value == DatasetFormat.noneLabelValue ? DatasetSample.noneLabel : Int(value)
    }

    private static func read32(_ d: Data, _ offset: Int) -> UInt32 {
        return UInt32(d[offset]) | (UInt32(d[offset + 1]) << 8) | (UInt32(d[offset + 2]) << 16) | (UInt32(d[offset + 3]) << 24)
    }

    private static func read16(_ d: Data, _ offset: Int) -> UInt16 {
        return UInt16(d[offset]) | (UInt16(d[offset + 1]) << 8)
    }
}
