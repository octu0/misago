import XCTest
@testable import Misago

final class ClassListTests: XCTestCase {
    func testOtherIsFirst() throws {
        let classes = try ClassList(others: ["a_home", "b_home"])
        XCTAssertEqual(classes.names, ["other", "a_home", "b_home"])
        XCTAssertEqual(classes.index(of: "b_home"), 2)
        XCTAssertEqual(classes[0], "other")
        XCTAssertThrowsError(try ClassList(names: ["a_home", "other"])) { error in
            XCTAssertEqual(error as? ClassListError, .firstClassMustBeOther("a_home"))
        }
    }

    func testInvalidNamesAreRejected() {
        XCTAssertThrowsError(try ClassList(names: []))
        XCTAssertThrowsError(try ClassList(others: ["a", "a"]))
        XCTAssertThrowsError(try ClassList(others: ["has space"]))
        XCTAssertThrowsError(try ClassList(others: ["a=b"]))
        XCTAssertThrowsError(try ClassList(others: [""]))
        XCTAssertThrowsError(try ClassList(others: (0..<255).map { "c\($0)" }))
        XCTAssertNoThrow(try ClassList(others: (0..<254).map { "c\($0)" }))
    }
}
