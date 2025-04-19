import SwiftUI

struct ContentView: View {
    @State private var selectedTab = 0

    var body: some View {
        NavigationView {
            ZStack {
                // Main background
                ColorManager.background.ignoresSafeArea()

                // Content area based on selected tab
                VStack {
                    TabView(selection: $selectedTab) {
                        HomeView()
                            .tag(0)
                        
                        Text(LocalizedStringKey("Training Screen"))
                            .foregroundColor(ColorManager.textPrimary)
                            .tag(1)
                        
                        // Use the new UploadTabView directly in the Upload tab
                        UploadTabView()
                            .tag(2)
                        
                        Text(LocalizedStringKey("Videos Screen"))
                            .foregroundColor(ColorManager.textPrimary)
                            .tag(3)
                        
                        Text(LocalizedStringKey("Messages Screen"))
                            .foregroundColor(ColorManager.textPrimary)
                            .tag(4)
                    }
                    .tabViewStyle(PageTabViewStyle(indexDisplayMode: .never))
                    
                    // Custom tab bar now using the evenly-spaced implementation
                    CustomTabBar(selectedTab: $selectedTab)
                }
            }
            .navigationBarHidden(true)
        }
    }
}

// MARK: - Previews
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .preferredColorScheme(.dark)
    }
}
