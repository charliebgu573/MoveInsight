#!/usr/bin/env python3
"""
Simple shuttlecock Y-coordinate visualization over time
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_y_over_time(csv_file):
    """Plot Y coordinate vs Frame number"""
    # Load data
    df = pd.read_csv(csv_file)
    
    # Filter for visible detections only
    df = df[df['Visibility'] == 1]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['Frame'], df['Y'], 'b-', linewidth=2, marker='o', markersize=3)
    
    plt.xlabel('Frame Number (Time)')
    plt.ylabel('Y Coordinate (pixels)')
    plt.title('Shuttlecock Height Over Time')
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()  # Invert Y to match video coordinates (0 at top)
    
    plt.tight_layout()
    plt.savefig('shuttlecock_height_over_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved as 'shuttlecock_height_over_time.png'")
    print(f"Total data points: {len(df)}")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'test_shuttlecock_tracking.csv'
    plot_y_over_time(csv_file)