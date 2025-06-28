## Testing Guidance

To test the improved match_analysis endpoint, you need a video that contains:

1. **Visible Badminton Court**: Clear view of a badminton court with court lines
2. **Players on Court**: 1-2 people playing badminton within the court area  
3. **Good Lighting**: Court lines should be clearly visible
4. **Stable Camera**: Preferably fixed camera position

### Example Test Command:
```bash
curl -X POST "http://localhost:8000/analyze/match_analysis/" \
  -F "file=@your_badminton_video.mp4" \
  -F "num_people=2" \
  -F "court_type=doubles" \
  -o match_analysis_result.zip
```

### Expected Output:
- `*_match_analysis_combined.mp4` - Main video with all overlays
- `*_top_down_movement.mp4` - Bird's-eye court view
- `*_shuttlecock_tracking.csv` - Shuttlecock coordinates
- `*_pose_data.json` - Player pose data
- `*_movement_data.csv` - Movement analysis
- `*_court_coordinates.txt` - Court reference points

### If Court Detection Fails:
The video likely doesn't contain a clearly visible badminton court. Try with:
- Real badminton match videos
- Videos with good court line visibility
- Proper lighting and camera angle

