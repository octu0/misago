import Foundation

/// Quarter-turn rotations of interleaved RGB pixels. Rotating never mirrors: turning an image back by the
/// opposite angle restores it exactly.
public enum ImageRotation {
    /// Rotates clockwise by `degrees` (a multiple of 90). The result has width and height swapped for 90 and 270.
    public static func rotate(width: Int, height: Int, rgb: [UInt8], degrees: Int) -> (width: Int, height: Int, rgb: [UInt8]) {
        let r = rotate(width: width, height: height, pixels: rgb, degrees: degrees)
        return (r.width, r.height, r.pixels)
    }

    /// Generic form for byte or float samples (3 channels).
    public static func rotate<T>(width: Int, height: Int, pixels: [T], degrees: Int) -> (width: Int, height: Int, pixels: [T]) {
        let turns = quarterTurns(degrees)
        if turns == 0 || pixels.isEmpty {
            return (width, height, pixels)
        }
        let outW = turns == 2 ? width : height
        let outH = turns == 2 ? height : width
        var out = pixels
        pixels.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                var y = 0
                while y < height {
                    var x = 0
                    while x < width {
                        // Destination of source pixel (x, y) under a clockwise rotation.
                        let dx: Int
                        let dy: Int
                        switch turns {
                        case 1:
                            dx = height - 1 - y
                            dy = x
                        case 2:
                            dx = width - 1 - x
                            dy = height - 1 - y
                        default:
                            dx = y
                            dy = width - 1 - x
                        }
                        let s = (y * width + x) * 3
                        let d = (dy * outW + dx) * 3
                        dst[d] = src[s]
                        dst[d + 1] = src[s + 1]
                        dst[d + 2] = src[s + 2]
                        x += 1
                    }
                    y += 1
                }
            }
        }
        return (outW, outH, out)
    }

    /// The part of the image that becomes `rect` once the image is turned clockwise by `degrees`, cut out and
    /// turned the same way. Equal to rotating the whole image and cropping `rect`, at the cost of the crop only.
    /// `rect` must lie inside the rotated image.
    public static func cropRotated(width: Int, height: Int, rgb: [UInt8], degrees: Int, rect: (x: Int, y: Int, width: Int, height: Int)) -> (width: Int, height: Int, rgb: [UInt8]) {
        let turns = quarterTurns(degrees)
        // The rectangle in source coordinates (inverse of the rotation in `rotate`).
        let sx: Int
        let sy: Int
        let sw: Int
        let sh: Int
        switch turns {
        case 1:
            sx = rect.y
            sy = height - rect.x - rect.width
            sw = rect.height
            sh = rect.width
        case 2:
            sx = width - rect.x - rect.width
            sy = height - rect.y - rect.height
            sw = rect.width
            sh = rect.height
        case 3:
            sx = width - rect.y - rect.height
            sy = rect.x
            sw = rect.height
            sh = rect.width
        default:
            sx = rect.x
            sy = rect.y
            sw = rect.width
            sh = rect.height
        }
        var crop = [UInt8](repeating: 0, count: sw * sh * 3)
        var y = 0
        while y < sh {
            let src = ((sy + y) * width + sx) * 3
            crop.replaceSubrange((y * sw * 3)..<((y + 1) * sw * 3), with: rgb[src..<(src + sw * 3)])
            y += 1
        }
        return rotate(width: sw, height: sh, rgb: crop, degrees: degrees)
    }

    private static func quarterTurns(_ degrees: Int) -> Int {
        return ((degrees / 90) % 4 + 4) % 4
    }
}

extension ScreenImage {
    /// The image turned upside down (a 180° rotation, not a mirror).
    public func rotated180() -> ScreenImage {
        var out = self
        out.pixels = ImageRotation.rotate(width: Self.width, height: Self.height, pixels: pixels, degrees: 180).pixels
        return out
    }
}
