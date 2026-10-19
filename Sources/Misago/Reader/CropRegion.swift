import Foundation

/// A rectangle on a screenshot expressed in units of the UI scale, so that it lands on the same UI element
/// across devices. The unit is the image height, or, when `maxAspect` is set, the smaller of the height and
/// width / maxAspect: games lay their HUD out for a wide screen and shrink it on squarer tablets, where the width
/// becomes the limiting dimension. `x0`/`x1` are measured from the anchor (left edge, horizontal center or right
/// edge) and `y0`/`y1` from the top.
public struct CropRegion: Sendable, Codable, Equatable {
    public enum Anchor: String, Sendable, Codable {
        case left
        case center
        case right
    }

    public var anchor: Anchor
    public var x0: Float
    public var x1: Float
    public var y0: Float
    public var y1: Float
    /// Aspect ratio (width / height) below which the unit follows the width instead of the height; nil = height only.
    public var maxAspect: Float?

    public init(anchor: Anchor, x0: Float, x1: Float, y0: Float, y1: Float, maxAspect: Float? = nil) {
        self.anchor = anchor
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.maxAspect = maxAspect
    }

    /// The length in pixels that the region coordinates are multiplied by.
    public func unit(width: Int, height: Int) -> Float {
        if let aspect = maxAspect, 0.0 < aspect {
            return min(Float(height), Float(width) / aspect)
        }
        return Float(height)
    }

    /// The rectangle in pixels for an image of the given size, clipped to the image.
    public func pixelRect(width: Int, height: Int) -> (x: Int, y: Int, width: Int, height: Int) {
        let h = unit(width: width, height: height)
        var base: Float = 0.0
        switch anchor {
        case .left:
            base = 0.0
        case .center:
            base = Float(width) * 0.5
        case .right:
            base = Float(width)
        }
        let px0 = min(max(0, Int((base + x0 * h).rounded())), width - 1)
        let px1 = min(width, max(px0 + 1, Int((base + x1 * h).rounded())))
        let py0 = min(max(0, Int((y0 * h).rounded())), height - 1)
        let py1 = min(height, max(py0 + 1, Int((y1 * h).rounded())))
        return (px0, py0, px1 - px0, py1 - py0)
    }
}

/// Crop regions keyed by class name, loaded from a JSON file supplied with the weights.
public struct RegionTable: Sendable, Codable, Equatable {
    public var regions: [String: CropRegion]

    public init(regions: [String: CropRegion]) {
        self.regions = regions
    }

    public static func load(from url: URL) throws -> RegionTable {
        let data = try Data(contentsOf: url)
        return RegionTable(regions: try JSONDecoder().decode([String: CropRegion].self, from: data))
    }

    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(regions).write(to: url, options: .atomic)
    }
}
