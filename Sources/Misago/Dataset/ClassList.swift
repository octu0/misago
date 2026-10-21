import Foundation

public enum ClassListError: Error, Equatable {
    case empty
    case invalidName(String)
    case duplicateName(String)
    case tooMany(Int)
}

/// Named classes of a classifier: `<title>_<screen>` named by the dataset (for example `<title>_home`), the
/// title being the part before the first underscore. Names are stored in dataset files and weights so that the
/// code never has to know them.
public struct ClassList: Sendable, Equatable, Codable {
    /// Names are stored with a one-byte length.
    public static let maxNameBytes = 255
    public static let maxCount = 255

    public let names: [String]

    public init(names: [String]) throws {
        if names.isEmpty {
            throw ClassListError.empty
        }
        if Self.maxCount < names.count {
            throw ClassListError.tooMany(names.count)
        }
        var seen = Set<String>()
        for name in names {
            if name.isEmpty || Self.maxNameBytes < name.utf8.count || name.contains(where: { $0.isWhitespace || $0 == "=" }) {
                throw ClassListError.invalidName(name)
            }
            if seen.insert(name).inserted != true {
                throw ClassListError.duplicateName(name)
            }
        }
        self.names = names
    }

    public var count: Int {
        return names.count
    }

    public func index(of name: String) -> Int? {
        return names.firstIndex(of: name)
    }

    public subscript(_ index: Int) -> String {
        return names[index]
    }

    /// Separates a class from the rotation of its training images: `<class>@180` is `<class>` seen turned 180°
    /// clockwise. Rotated copies keep their class and only add the angle, so one pass also tells the orientation.
    public static let rotationSeparator: Character = "@"

    /// The class and the clockwise rotation (degrees) encoded in a name; names without a tag have rotation 0.
    public static func parseRotation(_ name: String) -> (base: String, rotation: Int) {
        guard let at = name.lastIndex(of: rotationSeparator), let degrees = Int(name[name.index(after: at)...]) else {
            return (name, 0)
        }
        return (String(name[..<at]), ((degrees % 360) + 360) % 360)
    }

    /// The title of a class: its name without rotation tag, up to the first underscore.
    public static func title(of name: String) -> String {
        let base = parseRotation(name).base
        guard let underscore = base.firstIndex(of: "_") else {
            return base
        }
        return String(base[..<underscore])
    }

    /// Titles of the classes, in order of first appearance.
    public var titles: [String] {
        var out: [String] = []
        for name in names where out.contains(Self.title(of: name)) != true {
            out.append(Self.title(of: name))
        }
        return out
    }

    /// `name` tagged with `rotation` (the name itself for 0).
    public static func rotated(_ name: String, _ rotation: Int) -> String {
        let degrees = ((rotation % 360) + 360) % 360
        return degrees == 0 ? name : "\(name)\(rotationSeparator)\(degrees)"
    }
}
