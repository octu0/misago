import Foundation
import CStbImage
#if canImport(CTurboJPEG)
import CTurboJPEG
#endif
#if canImport(CLibPNG)
import CLibPNG
#endif

public enum ImageDecodingError: Error, Equatable {
    case undecodable(String)
    case tooLarge(width: Int, height: Int)
}

/// Decodes screenshots into interleaved RGB bytes (row 0 at the top). On Linux, JPEG goes through
/// libjpeg-turbo and PNG through libpng (both SIMD-accelerated system libraries); everything else, and all
/// formats on platforms without those libraries, goes through the bundled stb_image.
public enum ImageDecoder {
    /// Images above this many pixels are refused (a 4K screenshot is 8.3M).
    public static let maxPixels = 40_000_000

    public static func decodeRGB(_ data: Data) throws -> (width: Int, height: Int, rgb: [UInt8]) {
        #if canImport(CTurboJPEG)
        if isJPEG(data) {
            return try decodeJPEGTurbo(data)
        }
        #endif
        #if canImport(CLibPNG)
        if isPNG(data) {
            return try decodePNG(data)
        }
        #endif
        return try decodeSTB(data)
    }

    static func isJPEG(_ data: Data) -> Bool {
        return 3 <= data.count && data[data.startIndex] == 0xFF && data[data.startIndex + 1] == 0xD8 && data[data.startIndex + 2] == 0xFF
    }

    static func isPNG(_ data: Data) -> Bool {
        return 8 <= data.count && data[data.startIndex] == 0x89 && data[data.startIndex + 1] == 0x50 && data[data.startIndex + 2] == 0x4E && data[data.startIndex + 3] == 0x47
    }

    // MARK: - stb_image (all platforms)

    static func decodeSTB(_ data: Data) throws -> (width: Int, height: Int, rgb: [UInt8]) {
        var width: Int32 = 0
        var height: Int32 = 0
        var channels: Int32 = 0
        let pixels: UnsafeMutablePointer<UInt8>? = data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else {
                return nil
            }
            if stbi_info_from_memory(base.assumingMemoryBound(to: UInt8.self), Int32(raw.count), &width, &height, &channels) == 0 {
                return nil
            }
            if Int(width) * Int(height) > maxPixels {
                return nil
            }
            return stbi_load_from_memory(base.assumingMemoryBound(to: UInt8.self), Int32(raw.count), &width, &height, &channels, 3)
        }
        if Int(width) * Int(height) > maxPixels {
            throw ImageDecodingError.tooLarge(width: Int(width), height: Int(height))
        }
        guard let p = pixels else {
            var reason = "unknown image type"
            if let message = stbi_failure_reason() {
                reason = String(cString: message)
            }
            throw ImageDecodingError.undecodable(reason)
        }
        defer {
            stbi_image_free(p)
        }
        let count = Int(width) * Int(height) * 3
        return (Int(width), Int(height), Array(UnsafeBufferPointer(start: p, count: count)))
    }

    // MARK: - libjpeg-turbo (Linux)

    #if canImport(CTurboJPEG)
    static func decodeJPEGTurbo(_ data: Data) throws -> (width: Int, height: Int, rgb: [UInt8]) {
        guard let handle = tjInitDecompress() else {
            throw ImageDecodingError.undecodable("tjInitDecompress failed")
        }
        defer {
            tjDestroy(handle)
        }
        var width: Int32 = 0
        var height: Int32 = 0
        var subsampling: Int32 = 0
        var colorspace: Int32 = 0
        let headerOK = data.withUnsafeBytes { raw -> Bool in
            guard let base = raw.baseAddress else {
                return false
            }
            let p = UnsafeMutablePointer(mutating: base.assumingMemoryBound(to: UInt8.self))
            return tjDecompressHeader3(handle, p, UInt(raw.count), &width, &height, &subsampling, &colorspace) == 0
        }
        if headerOK != true {
            throw ImageDecodingError.undecodable(turboMessage(handle))
        }
        if Int(width) * Int(height) > maxPixels {
            throw ImageDecodingError.tooLarge(width: Int(width), height: Int(height))
        }
        var rgb = [UInt8](repeating: 0, count: Int(width) * Int(height) * 3)
        let ok = data.withUnsafeBytes { raw -> Bool in
            rgb.withUnsafeMutableBufferPointer { out -> Bool in
                let p = UnsafeMutablePointer(mutating: raw.baseAddress!.assumingMemoryBound(to: UInt8.self))
                return tjDecompress2(handle, p, UInt(raw.count), out.baseAddress!, width, 0, height, Int32(TJPF_RGB.rawValue), 0) == 0
            }
        }
        if ok != true {
            throw ImageDecodingError.undecodable(turboMessage(handle))
        }
        return (Int(width), Int(height), rgb)
    }

    static func turboMessage(_ handle: tjhandle) -> String {
        if let message = tjGetErrorStr2(handle) {
            return String(cString: message)
        }
        return "libjpeg-turbo error"
    }
    #endif

    // MARK: - libpng simplified API (Linux)

    #if canImport(CLibPNG)
    static func decodePNG(_ data: Data) throws -> (width: Int, height: Int, rgb: [UInt8]) {
        var image = png_image()
        image.version = UInt32(PNG_IMAGE_VERSION)
        let began = data.withUnsafeBytes { raw -> Bool in
            guard let base = raw.baseAddress else {
                return false
            }
            return png_image_begin_read_from_memory(&image, base, raw.count) != 0
        }
        if began != true {
            throw ImageDecodingError.undecodable(pngMessage(&image))
        }
        let width = Int(image.width)
        let height = Int(image.height)
        if width * height > maxPixels {
            png_image_free(&image)
            throw ImageDecodingError.tooLarge(width: width, height: height)
        }
        // PNG_FORMAT_RGB: 8-bit, three channels, no alpha; libpng converts palette, gray and 16-bit input.
        image.format = 0x02
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        let finished = rgb.withUnsafeMutableBufferPointer { out -> Bool in
            return png_image_finish_read(&image, nil, out.baseAddress!, 0, nil) != 0
        }
        if finished != true {
            let message = pngMessage(&image)
            png_image_free(&image)
            throw ImageDecodingError.undecodable(message)
        }
        return (width, height, rgb)
    }

    static func pngMessage(_ image: inout png_image) -> String {
        return withUnsafePointer(to: &image.message) { p in
            p.withMemoryRebound(to: CChar.self, capacity: 64) { String(cString: $0) }
        }
    }
    #endif
}
