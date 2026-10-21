import XCTest
@testable import Misago

final class ClassListTests: XCTestCase {
    func testNamesKeepTheirOrderAndGiveTitles() throws {
        let classes = try ClassList(names: ["a_home", "b_home", "a_battle_knockout", "a_home@180"])
        XCTAssertEqual(classes.index(of: "b_home"), 1)
        XCTAssertEqual(classes.titles, ["a", "b"])
        XCTAssertEqual(ClassList.title(of: "a_battle_knockout"), "a")
        XCTAssertEqual(ClassList.title(of: "a_home@180"), "a")
        XCTAssertEqual(ClassList.title(of: "plain"), "plain")
    }

    func testInvalidNamesAreRejected() {
        XCTAssertThrowsError(try ClassList(names: []))
        XCTAssertThrowsError(try ClassList(names: ["other"] + ["a", "a"]))
        XCTAssertThrowsError(try ClassList(names: ["other"] + ["has space"]))
        XCTAssertThrowsError(try ClassList(names: ["other"] + ["a=b"]))
        XCTAssertThrowsError(try ClassList(names: ["other"] + [""]))
        XCTAssertThrowsError(try ClassList(names: ["other"] + (0..<255).map { "c\($0)" }))
        XCTAssertNoThrow(try ClassList(names: ["other"] + (0..<254).map { "c\($0)" }))
    }
}
