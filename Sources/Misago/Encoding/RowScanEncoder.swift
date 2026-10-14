import Foundation

/// Turns a `ScreenImage` into the sequence the Hawk model reads: rows from top to bottom, two rows per frame.
///
/// Frame t = rows 2t and 2t + 1, each as 128 interleaved RGB triples, so a frame holds the full width of the
/// screen and the sequence index follows the vertical position. 72 rows give 36 frames of 768 values.
public enum RowScanEncoder {
    public static let rowsPerFrame = 2
    public static let frameCount = ScreenImage.height / rowsPerFrame
    public static let frameDim = ScreenImage.width * ScreenImage.channels * rowsPerFrame

    /// [frameCount][frameDim]
    public static func encode(_ image: ScreenImage) -> [[Float]] {
        var frames: [[Float]] = []
        frames.reserveCapacity(frameCount)
        var t = 0
        while t < frameCount {
            let start = t * frameDim
            frames.append(Array(image.pixels[start..<(start + frameDim)]))
            t += 1
        }
        return frames
    }

    /// The frames concatenated: `frameCount * frameDim` values, for handing to MLX in one buffer.
    /// With two consecutive rows per frame this is the pixel buffer itself.
    public static func encodeFlat(_ image: ScreenImage) -> [Float] {
        return image.pixels
    }
}
