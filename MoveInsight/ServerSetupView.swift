// MoveInsight/ServerSetupView.swift
import SwiftUI

struct ServerSetupView: View {
    @State private var serverIP: String = "" // For the TextField
    @AppStorage("serverBaseIP") private var storedServerIP: String? // To save/load the IP
    @Binding var isServerConfigured: Bool

    @State private var lastUsedIP: String? = nil // To display the last used IP, if available

    var body: some View {
        VStack(spacing: 20) {
            Text("Server Configuration")
                .font(.largeTitle)
                .padding(.bottom, 30)

            Text("Enter the base IP address of the MoveInsight server (e.g., 192.168.1.100), or use the last known IP.")
                .font(.subheadline)
                .foregroundColor(.gray)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            // Display last used IP and a button to use it
            if let previousIP = lastUsedIP, !previousIP.isEmpty {
                VStack(spacing: 10) {
                    Text("Last used IP: \(previousIP)")
                        .font(.caption)
                        .foregroundColor(ColorManager.textSecondary)
                    Button("Use Last IP: \(previousIP)") {
                        serverIP = previousIP // Populate TextField with the last used IP
                        // Optionally, you could directly attempt to save and continue here
                        // if you want this button to be a one-click action.
                        // For now, it just fills the TextField.
                    }
                    .padding(.horizontal, 15)
                    .padding(.vertical, 8)
                    .background(ColorManager.accentColor.opacity(0.2))
                    .foregroundColor(ColorManager.accentColor)
                    .cornerRadius(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(ColorManager.accentColor, lineWidth: 1)
                    )
                }
                .padding(.bottom, 10)
            }

            TextField("Enter New Server IP Address", text: $serverIP)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .keyboardType(.decimalPad)
                .autocapitalization(.none)
                .disableAutocorrection(true)
                .padding(.horizontal, 40)
                .overlay(
                    HStack {
                        Spacer()
                        if !serverIP.isEmpty {
                            Button(action: { serverIP = "" }) { // Clear button
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundColor(.gray)
                            }
                            .padding(.trailing, 50) // Adjust padding to avoid overlap
                        }
                    }
                )


            Button("Save and Continue") {
                let trimmedIP = serverIP.trimmingCharacters(in: .whitespacesAndNewlines)
                if isValidIP(trimmedIP) {
                    storedServerIP = trimmedIP // Save the successfully validated IP
                    isServerConfigured = true
                } else {
                    // Handle invalid IP (e.g., show an alert)
                    print("Invalid IP address format entered: \(trimmedIP)")
                    // You might want to add a @State variable to show an error message to the user
                }
            }
            .padding()
            .foregroundColor(.white)
            .background(ColorManager.accentColor)
            .cornerRadius(10)
            .disabled(serverIP.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty) // Disable if TextField is empty or only whitespace
        }
        .padding()
        .onAppear {
            // Load the last successfully stored IP to display it and potentially pre-fill
            if let currentStoredIP = storedServerIP {
                lastUsedIP = currentStoredIP
                // Pre-fill the TextField with the last used IP if it's currently empty.
                // User can then choose to edit it or use the "Use Last IP" button which also populates it.
                if serverIP.isEmpty {
                     serverIP = currentStoredIP
                }
            }
        }
    }

    // Basic IP validation
    private func isValidIP(_ ip: String) -> Bool {
        let parts = ip.split(separator: ".")
        guard parts.count == 4 else { return false }
        return parts.allSatisfy { part in
            if let num = Int(part), num >= 0 && num <= 255 {
                return true
            }
            return false
        }
    }
}

struct ServerSetupView_Previews: PreviewProvider {
    static var previews: some View {
        // Example with a previously stored IP for preview
        let _ = UserDefaults.standard.setValue("192.168.0.10", forKey: "serverBaseIP")
        return ServerSetupView(isServerConfigured: .constant(false))
    }
}
