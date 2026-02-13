#!/usr/bin/env python3
"""
Test Episode Detection with NA Gap Tolerance

This script validates that the new identify_outbreak_episodes_with_gaps()
function correctly bridges 1-week NA gaps while splitting on longer gaps.

Expected Behavior:
------------------
Test Case:
    Week:     1  2  3  4  5  6  7  8
    Above:    T  T  NA T  T  F  T  T
    Filled:   T  T  T  T  T  F  T  T
                └─episode 1─┘    └ep2┘

Expected Result:
    - 2 episodes detected
    - Episode 1: weeks 1-5 (bridging week 3 NA gap)
    - Episode 2: weeks 7-8
    - Episode 1 should have 4 actual outbreak weeks (1,2,4,5)
    - Episode 2 should have 2 actual outbreak weeks (7,8)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.evaluation.lead_time import identify_outbreak_episodes_with_gaps

def test_episode_detection():
    """Test episode detection with NA gap tolerance."""
    
    print("="*70)
    print("TESTING EPISODE DETECTION WITH NA GAP TOLERANCE")
    print("="*70)
    
    # Create test case
    test_data = pd.DataFrame({
        'state': ['TestState'] * 8,
        'district': ['TestDist'] * 8,
        'week': range(1, 9),
        'year': [2020] * 8,
        'cases': [15, 16, np.nan, 15, 17, 8, 12, 13]  # NA at week 3, below threshold at week 6
    })
    
    # Set threshold
    outbreak_thresholds = {
        ('TestState', 'TestDist'): 10.0
    }
    
    print("\nTest Data:")
    print("-"*70)
    print(test_data[['week', 'cases']])
    print(f"\nOutbreak Threshold: {outbreak_thresholds[('TestState', 'TestDist')]}")
    print(f"Weeks above threshold: 1, 2, 4, 5, 7, 8")
    print(f"Week 3: NA (should be bridged)")
    print(f"Week 6: 8 (below threshold, should split episodes)")
    
    # Run episode detection
    print("\n" + "="*70)
    print("RUNNING EPISODE DETECTION")
    print("="*70)
    
    episodes = identify_outbreak_episodes_with_gaps(
        test_df=test_data,
        outbreak_thresholds=outbreak_thresholds,
        fold_name='test_fold',
        case_col='cases',
        max_gap_weeks=1,
        min_episode_length=2
    )
    
    # Validate results
    print(f"\n{'='*70}")
    print("VALIDATION RESULTS")
    print("="*70)
    
    print(f"\nNumber of episodes detected: {len(episodes)}")
    
    if len(episodes) == 0:
        print("\n❌ FAILED: No episodes detected!")
        return False
    
    for i, ep in enumerate(episodes, 1):
        print(f"\nEpisode {i}:")
        print(f"  First week: {ep.first_outbreak_week}")
        print(f"  Peak week: {ep.peak_week}")
        print(f"  Peak cases: {ep.peak_cases}")
        print(f"  Total outbreak weeks: {ep.total_outbreak_weeks}")
    
    # Check expected results
    success = True
    
    if len(episodes) != 2:
        print(f"\n❌ FAILED: Expected 2 episodes, got {len(episodes)}")
        success = False
    else:
        print(f"\n✓ Correct number of episodes: 2")
        
        # Episode 1: weeks 1-5 (bridging NA at week 3)
        ep1 = episodes[0]
        if ep1.first_outbreak_week == 1 and ep1.total_outbreak_weeks == 4:
            print(f"✓ Episode 1: weeks 1-5 with 4 outbreak weeks (bridged NA gap)")
        else:
            print(f"❌ Episode 1 incorrect: first_week={ep1.first_outbreak_week}, outbreak_weeks={ep1.total_outbreak_weeks}")
            print(f"   Expected: first_week=1, outbreak_weeks=4")
            success = False
        
        # Episode 2: weeks 7-8
        ep2 = episodes[1]
        if ep2.first_outbreak_week == 7 and ep2.total_outbreak_weeks == 2:
            print(f"✓ Episode 2: weeks 7-8 with 2 outbreak weeks")
        else:
            print(f"❌ Episode 2 incorrect: first_week={ep2.first_outbreak_week}, outbreak_weeks={ep2.total_outbreak_weeks}")
            print(f"   Expected: first_week=7, outbreak_weeks=2")
            success = False
    
    print("\n" + "="*70)
    if success:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ TESTS FAILED!")
    print("="*70)
    
    return success


def test_no_gap_tolerance():
    """Test that episodes are correctly split when gap is too large."""
    
    print("\n\n" + "="*70)
    print("TESTING EPISODE SPLITTING (NO GAP TOLERANCE FOR 2-WEEK GAP)")
    print("="*70)
    
    # Create test case with 2-week gap
    test_data = pd.DataFrame({
        'state': ['TestState'] * 10,
        'district': ['TestDist'] * 10,
        'week': range(1, 11),
        'year': [2020] * 10,
        'cases': [15, 16, np.nan, np.nan, 17, 8, 12, 13, 14, 15]  # 2-week NA gap (weeks 3-4)
    })
    
    outbreak_thresholds = {
        ('TestState', 'TestDist'): 10.0
    }
    
    print("\nTest Data:")
    print("-"*70)
    print(test_data[['week', 'cases']])
    print(f"\nOutbreak Threshold: {outbreak_thresholds[('TestState', 'TestDist')]}")
    print(f"Weeks above threshold: 1, 2, 5, 7, 8, 9, 10")
    print(f"Weeks 3-4: NA (2-week gap, should NOT be bridged)")
    print(f"Week 6: 8 (below threshold)")
    
    episodes = identify_outbreak_episodes_with_gaps(
        test_df=test_data,
        outbreak_thresholds=outbreak_thresholds,
        fold_name='test_fold',
        case_col='cases',
        max_gap_weeks=1,  # Only bridge 1-week gaps
        min_episode_length=2
    )
    
    print(f"\nNumber of episodes detected: {len(episodes)}")
    
    for i, ep in enumerate(episodes, 1):
        print(f"\nEpisode {i}:")
        print(f"  First week: {ep.first_outbreak_week}")
        print(f"  Total outbreak weeks: {ep.total_outbreak_weeks}")
    
    # Should have at least 2 separate episodes (week 1-2 group, and week 7+ group)
    if len(episodes) >= 2:
        print(f"\n✓ Correctly split episodes on 2-week gap")
        return True
    else:
        print(f"\n❌ Failed to split on 2-week gap")
        return False


def test_min_episode_length():
    """Test that single-week episodes are filtered out."""
    
    print("\n\n" + "="*70)
    print("TESTING MINIMUM EPISODE LENGTH FILTER")
    print("="*70)
    
    test_data = pd.DataFrame({
        'state': ['TestState'] * 8,
        'district': ['TestDist'] * 8,
        'week': range(1, 9),
        'year': [2020] * 8,
        'cases': [15, 8, 8, 8, 16, 17, 8, 8]  # Only weeks 1, 5, 6 above threshold
    })
    
    outbreak_thresholds = {
        ('TestState', 'TestDist'): 10.0
    }
    
    print("\nTest Data:")
    print("-"*70)
    print(test_data[['week', 'cases']])
    print(f"\nOutbreak Threshold: {outbreak_thresholds[('TestState', 'TestDist')]}")
    print(f"Weeks above threshold: 1 (isolated), 5-6 (consecutive)")
    print(f"Expected: 1 episode (weeks 5-6), week 1 should be filtered out")
    
    episodes = identify_outbreak_episodes_with_gaps(
        test_df=test_data,
        outbreak_thresholds=outbreak_thresholds,
        fold_name='test_fold',
        case_col='cases',
        max_gap_weeks=1,
        min_episode_length=2  # Require at least 2 weeks
    )
    
    print(f"\nNumber of episodes detected: {len(episodes)}")
    
    for i, ep in enumerate(episodes, 1):
        print(f"\nEpisode {i}:")
        print(f"  First week: {ep.first_outbreak_week}")
        print(f"  Total outbreak weeks: {ep.total_outbreak_weeks}")
    
    if len(episodes) == 1 and episodes[0].first_outbreak_week == 5:
        print(f"\n✓ Correctly filtered out single-week episode")
        return True
    else:
        print(f"\n❌ Failed to filter single-week episode or detected wrong episode")
        return False


if __name__ == '__main__':
    print("\n" + "="*70)
    print("EPISODE DETECTION VALIDATION SUITE")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("NA Gap Bridging", test_episode_detection()))
    results.append(("2-Week Gap Splitting", test_no_gap_tolerance()))
    results.append(("Min Episode Length", test_min_episode_length()))
    
    # Summary
    print("\n\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL VALIDATION TESTS PASSED!")
        print("Episode detection is working correctly.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the implementation.")
    print("="*70)
    
    sys.exit(0 if all_passed else 1)
