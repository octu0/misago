import Foundation

public enum ClassListError: Error, Equatable {
    case empty
    case firstClassMustBeOther(String)
    case invalidName(String)
    case duplicateName(String)
    case tooMany(Int)
}

/// Named classes of a classifier. Class 0 is always `other` (not a recognized screen); the rest are
/// title-and-screen classes named by the dataset (for example `<title>_home`). Names are stored in
/// dataset files and weights so that the code never has to know them.
public struct ClassList: Sendable, Equatable, Codable {
    public static let otherName = "other"
    /// Names are stored with a one-byte length.
    public static let maxNameBytes = 255
    public static let maxCount = 255

    public let names: [String]

    public init(names: [String]) throws {
        if names.isEmpty {
            throw ClassListError.empty
        }
        if names[0] != Self.otherName {
            throw ClassListError.firstClassMustBeOther(names[0])
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

    /// `other` plus the given class names, in order.
    public init(others: [String]) throws {
        try self.init(names: [Self.otherName] + others)
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
}
