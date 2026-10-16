import Foundation
import CStbImage

public enum ImageDecodingError: Error, Equatable {
    case undecodable(String)
    case tooLarge(width: Int, height: Int)
}

/// Decodes PNG, JPEG, BMP and WebP-less common screenshot formats into interleaved RGB bytes (row 0 at the top)
/// with stb_image, so the same code runs on macOS and Linux without ImageIO.
public enum ImageDecoder {
    /// Images above this many pixels are refused (a 4K screenshot is 8.3M).
    public static let maxPixels = 40_000_000

    public static func decodeRGB(_ data: Data) throws -> (width: Int, height: Int, rgb: [UInt8]) {
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
            let reason = String(cString: stbi_failure_reason() ?? UnsafePointer(bitPattern: 0)!)
            throw ImageDecodingError.undecodable(reason)
        }
        defer {
            stbi_image_free(p)
        }
        let count = Int(width) * Int(height) * 3
        return (Int(width), Int(height), Array(UnsafeBufferPointer(start: p, count: count)))
    }
}
