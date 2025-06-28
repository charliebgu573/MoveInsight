// MoveInsight/MoveInsightApp.swift
import SwiftUI

@main
struct MoveInsightApp: App {
    // This state will now always start as false, forcing ServerSetupView to show.
    @State private var isServerConfiguredThisSession: Bool = false

    init() {
        // Set the accent color for the entire app (existing code)
        UINavigationBar.appearance().tintColor = UIColor(ColorManager.accentColor)
        
        // Optional: Clear the stored IP on app launch if you truly want a fresh entry every single time
        // and don't want the previous session's IP to be even pre-filled in the text field.
        // UserDefaults.standard.removeObject(forKey: "serverBaseIP")
    }
    
    var body: some Scene {
        WindowGroup {
            if isServerConfiguredThisSession {
                contentWithTint()
            } else {
                // Always show ServerSetupView first.
                // It will set isServerConfiguredThisSession to true when IP is saved.
                ServerSetupView(isServerConfigured: $isServerConfiguredThisSession)
            }
        }
    }
    
    // For iOS 17+, we can use the newer API too (existing code)
    #if swift(>=5.9)
    @ViewBuilder
    private func contentWithTint() -> some View {
        if #available(iOS 17.0, *) {
            ContentView()
                .preferredColorScheme(.dark) // Or remove for system default
                .tint(ColorManager.accentColor)
        } else {
            ContentView()
                .preferredColorScheme(.dark) // Or remove for system default
                .accentColor(ColorManager.accentColor)
        }
    }
    #else
    // Fallback for older Swift versions
    @ViewBuilder
    private func contentWithTint() -> some View {
        ContentView()
            .preferredColorScheme(.dark) // Or remove for system default
            .accentColor(ColorManager.accentColor)
    }
    #endif
}
