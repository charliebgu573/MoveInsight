import SwiftUI

@main
struct MoveInsightApp: App {
    init() {
        // Set the accent color for the entire app
        UINavigationBar.appearance().tintColor = UIColor(Color.accentColor)
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .accentColor(ColorManager.accentColor)
        }
    }
}

// For iOS 17+, we can use the newer API too
#if swift(>=5.9)
extension MoveInsightApp {
    @ViewBuilder
    private func contentWithTint() -> some View {
        if #available(iOS 17.0, *) {
            ContentView()
                .preferredColorScheme(.dark)
                .tint(ColorManager.accentColor)
        } else {
            ContentView()
                .preferredColorScheme(.dark)
                .accentColor(ColorManager.accentColor)
        }
    }
}
#endif
