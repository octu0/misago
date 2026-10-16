import Foundation
import ImageIO
import CoreGraphics

/// Decodes an image file into interleaved RGB bytes, row 0 at the top. ImageIO is used only by the tools.
func loadRGB(path: String) throws -> (width: Int, height: Int, rgb: [UInt8]) {
    let url = URL(fileURLWithPath: path)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil), 0 < CGImageSourceGetCount(source),
          let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw NSError(domain: "misago", code: 1, userInfo: [NSLocalizedDescriptionKey: "cannot decode \(path)"])
    }
    let width = cgImage.width
    let height = cgImage.height
    var rgba = [UInt8](repeating: 0, count: width * height * 4)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let ok = rgba.withUnsafeMutableBytes { buffer -> Bool in
        guard let context = CGContext(
            data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
            space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        ) else {
            return false
        }
        // A bitmap context's first row in memory is the top of the picture, so drawing the image unflipped
        // leaves row 0 at the top (flipping here would turn the picture upside down).
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return true
    }
    if ok != true {
        throw NSError(domain: "misago", code: 2, userInfo: [NSLocalizedDescriptionKey: "cannot draw \(path)"])
    }
    var rgb = [UInt8](repeating: 0, count: width * height * 3)
    var i = 0
    while i < width * height {
        rgb[i * 3] = rgba[i * 4]
        rgb[i * 3 + 1] = rgba[i * 4 + 1]
        rgb[i * 3 + 2] = rgba[i * 4 + 2]
        i += 1
    }
    return (width, height, rgb)
}

let imageExtensions: Set<String> = ["png", "jpg", "jpeg", "webp", "heic", "bmp", "tiff", "tif", "gif"]

/// Image files under `directory`, recursively, sorted by path.
func listImages(directory: String) -> [String] {
    var out: [String] = []
    let manager = FileManager.default
    guard let enumerator = manager.enumerator(atPath: directory) else {
        return out
    }
    while let relative = enumerator.nextObject() as? String {
        let ext = (relative as NSString).pathExtension.lowercased()
        if imageExtensions.contains(ext) && (relative as NSString).lastPathComponent.hasPrefix(".") != true {
            out.append((directory as NSString).appendingPathComponent(relative))
        }
    }
    return out.sorted()
}
