import SwiftUI
import Combine

// MARK: - Publisher Extension
extension AnyCancellable {
    func cancel(after interval: TimeInterval) {
        DispatchQueue.main.asyncAfter(deadline: .now() + interval) {
            self.cancel()
        }
    }
}
