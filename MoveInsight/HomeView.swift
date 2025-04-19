import SwiftUI

struct HomeView: View {
    // Sample user data – name typically wouldn’t be localized
    let username = "Zhang Wei"
    
    var body: some View {
        ZStack {
            // Main background
            ColorManager.background.ignoresSafeArea()
            
            // Content
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    // User greeting
                    HStack {
                        VStack(alignment: .leading) {
                            Text(LocalizedStringKey("Good morning,"))
                                .font(.system(size: 16))
                                .foregroundColor(ColorManager.textSecondary)
                            Text(username)
                                .font(.system(size: 24, weight: .bold))
                                .foregroundColor(ColorManager.textPrimary)
                        }
                        
                        Spacer()
                        
                        // Profile image
                        Image(systemName: "person.crop.circle.fill")
                            .resizable()
                            .frame(width: 40, height: 40)
                            .foregroundColor(ColorManager.accentColor)
                            .background(ColorManager.cardBackground)
                            .clipShape(Circle())
                    }
                    .padding(.top, 12)
                    
                    // Match Performance Card
                    PerformanceCard()
                    
                    // Technicals Section
                    NavigationLink(destination: Text(LocalizedStringKey("Technicals Detail"))) {
                        SectionCard(title: LocalizedStringKey("Technicals"))
                    }
                    
                    // Training Goals
                    NavigationLink(destination: Text(LocalizedStringKey("Training Goals Detail"))) {
                        GoalsCard()
                    }
                    
                    // Tutorials Section
                    Text(LocalizedStringKey("Tutorials Specifically For You"))
                        .font(.headline)
                        .foregroundColor(ColorManager.textPrimary)
                        .padding(.top, 5)
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 20)
            }
        }
    }
}

// MARK: - Performance Card
struct PerformanceCard: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                Text(LocalizedStringKey("Match Performance"))
                    .font(.headline)
                    .foregroundColor(ColorManager.textPrimary)
                
                Spacer()
                
                Image(systemName: "arrow.up.right.square")
                    .foregroundColor(ColorManager.textSecondary)
            }
            
            HStack(alignment: .top, spacing: 15) {
                // Improvement stat
                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Image(systemName: "chart.line.uptrend.xyaxis")
                            .foregroundColor(ColorManager.accentColor)
                        
                        Text(LocalizedStringKey("2.3%"))
                            .fontWeight(.bold)
                            .foregroundColor(ColorManager.textPrimary)
                    }
                    
                    Text(LocalizedStringKey("/ last week"))
                        .font(.caption)
                        .foregroundColor(ColorManager.textSecondary)
                    
                    Text(LocalizedStringKey("Well done on swing path!"))
                        .font(.caption)
                        .foregroundColor(ColorManager.textSecondary)
                        .padding(.top, 5)
                }
                
                Spacer()
                
                // Rating gauge using only purple with a gradient
                ZStack {
                    Circle()
                        .trim(from: 0, to: 0.75)
                        .stroke(
                            AngularGradient(
                                gradient: Gradient(colors: [ColorManager.accentColor.opacity(0.6), ColorManager.accentColor]),
                                center: .center,
                                startAngle: .degrees(0),
                                endAngle: .degrees(270)
                            ),
                            style: StrokeStyle(lineWidth: 8, lineCap: .round)
                        )
                        .frame(width: 80, height: 80)
                        .rotationEffect(.degrees(135))
                    
                    Text(LocalizedStringKey("4.3"))
                        .font(.system(size: 24, weight: .bold))
                        .foregroundColor(ColorManager.textPrimary)
                }
            }
            
            // Progress chart
            Chart()
                .frame(height: 120)
                .padding(.vertical, 8)
            
            // Match history link
            HStack {
                Text(LocalizedStringKey("Match History"))
                    .foregroundColor(ColorManager.textSecondary)
                    .font(.subheadline)
                
                Spacer()
                
                Image(systemName: "chevron.right")
                    .foregroundColor(ColorManager.textSecondary)
                    .font(.caption)
            }
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(ColorManager.cardBackground.opacity(0.8))
        )
    }
}

// MARK: - Simple Chart Component
struct Chart: View {
    let months = ["Jan", "Feb", "Mar", "Apr", "May"]
    
    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let height = geometry.size.height
            
            HStack(spacing: 0) {
                ForEach(months, id: \.self) { month in
                    Text(LocalizedStringKey(month))
                        .font(.system(size: 8))
                        .foregroundColor(ColorManager.textSecondary)
                        .frame(width: width / CGFloat(months.count))
                }
            }
            .position(x: width / 2, y: height - 5)
            
            VStack(spacing: 8) {
                ForEach(["4", "3", "2", "1", "0"], id: \.self) { value in
                    Text(value)
                        .font(.system(size: 8))
                        .foregroundColor(ColorManager.textSecondary)
                }
            }
            .position(x: 8, y: height / 2)
            
            Path { path in
                let points = [
                    CGPoint(x: width * 0.1, y: height * 0.7),
                    CGPoint(x: width * 0.3, y: height * 0.5),
                    CGPoint(x: width * 0.5, y: height * 0.4),
                    CGPoint(x: width * 0.7, y: height * 0.35),
                    CGPoint(x: width * 0.9, y: height * 0.3)
                ]
                
                path.move(to: points[0])
                for point in points.dropFirst() {
                    path.addLine(to: point)
                }
            }
            .stroke(ColorManager.textPrimary, lineWidth: 1.5)
        }
    }
}

// MARK: - Section Card
struct SectionCard: View {
    let title: LocalizedStringKey
    
    var body: some View {
        HStack {
            Text(title)
                .font(.headline)
                .foregroundColor(ColorManager.textPrimary)
            
            Spacer()
            
            Image(systemName: "chevron.right")
                .foregroundColor(ColorManager.textSecondary)
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(ColorManager.cardBackground.opacity(0.8))
        )
    }
}

// MARK: - Goals Card
struct GoalsCard: View {
    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text(LocalizedStringKey("Training Goals"))
                    .font(.headline)
                    .foregroundColor(ColorManager.textPrimary)
                
                HStack(spacing: 8) {
                    ForEach(0..<4) { _ in
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                    }
                    
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(ColorManager.textSecondary.opacity(0.5))
                }
            }
            
            Spacer()
            
            Image(systemName: "chevron.right")
                .foregroundColor(ColorManager.textSecondary)
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(ColorManager.cardBackground.opacity(0.8))
        )
    }
}
