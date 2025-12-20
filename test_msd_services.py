"""
Test script for MSD services
Verifies database, profiler, and importer functionality
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from services.msd_database_service import MSDDatabaseService
from services.genre_profiler import GenreProfiler
from services.msd_importer import MSDSubsetImporter

def test_msd_services():
    """Test MSD services initialization and basic operations"""
    
    print("=" * 60)
    print("MSD Services Test")
    print("=" * 60)
    
    # Initialize services
    print("\n1. Initializing services...")
    try:
        db_service = MSDDatabaseService("data/test_msd.db")
        profiler = GenreProfiler("data/test_msd.db")
        importer = MSDSubsetImporter()
        importer.db_service.db_path = "data/test_msd.db"
        importer.profiler.db_path = "data/test_msd.db"
        print("✅ Services initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize services: {e}")
        return False
    
    # Test database creation
    print("\n2. Testing database operations...")
    try:
        stats = db_service.get_database_stats()
        print(f"✅ Database stats: {stats}")
    except Exception as e:
        print(f"❌ Database operation failed: {e}")
        return False
    
    # Test sample data generation
    print("\n3. Generating sample data...")
    try:
        samples = importer.generate_sample_data(count=50)
        print(f"✅ Generated {len(samples)} sample songs")
        print(f"   Sample: {samples[0]['title']} by {samples[0]['artist_name']} ({samples[0]['genre']})")
    except Exception as e:
        print(f"❌ Sample generation failed: {e}")
        return False
    
    # Test batch import
    print("\n4. Testing batch import...")
    try:
        successful, failed = db_service.import_songs_batch(samples)
        print(f"✅ Import results: {successful} successful, {failed} failed")
    except Exception as e:
        print(f"❌ Batch import failed: {e}")
        return False
    
    # Test genre analysis
    print("\n5. Testing genre profiling...")
    try:
        # Get unique genres from samples
        genres = set(s['genre'] for s in samples)
        print(f"   Analyzing {len(genres)} genres: {', '.join(genres)}")
        
        for genre in list(genres)[:3]:  # Test first 3 genres
            profile = profiler.analyze_genre(genre)
            if profile:
                print(f"   ✅ {genre}: {profile['song_count']} songs, avg tempo {profile['tempo']['avg']} BPM")
            else:
                print(f"   ⚠️ {genre}: No profile generated")
                
    except Exception as e:
        print(f"❌ Genre profiling failed: {e}")
        return False
    
    # Test saving profiles
    print("\n6. Testing profile persistence...")
    try:
        analyzed = profiler.analyze_all_genres()
        print(f"✅ Analyzed and saved {analyzed} genre profiles")
    except Exception as e:
        print(f"❌ Profile saving failed: {e}")
        return False
    
    # Test parameter suggestions
    print("\n7. Testing parameter suggestions...")
    try:
        genres = profiler.get_all_genre_names()
        if genres:
            test_genre = genres[0]
            suggestions = profiler.suggest_parameters_for_genre(test_genre)
            
            if suggestions:
                print(f"✅ Suggestions for {test_genre}:")
                print(f"   Tempo: {suggestions['tempo_bpm']} BPM")
                print(f"   Range: {suggestions['tempo_range']['min']:.0f} - {suggestions['tempo_range']['max']:.0f} BPM")
                print(f"   Keys: {suggestions['recommended_keys']}")
                print(f"   Mode: {suggestions['recommended_mode']}")
            else:
                print(f"⚠️ No suggestions generated for {test_genre}")
        else:
            print("⚠️ No genres available")
            
    except Exception as e:
        print(f"❌ Parameter suggestions failed: {e}")
        return False
    
    # Test key-tempo patterns
    print("\n8. Testing key-tempo pattern analysis...")
    try:
        patterns = profiler.analyze_key_tempo_patterns()
        print(f"✅ Found {len(patterns)} key-tempo patterns")
        
        if patterns:
            pattern = patterns[0]
            print(f"   Example: {pattern['key_name']} {pattern['mode_name']}: {pattern['tempo_avg']:.0f} BPM avg")
        
        saved = profiler.save_key_tempo_patterns(patterns)
        print(f"✅ Saved {saved} patterns to database")
        
    except Exception as e:
        print(f"❌ Key-tempo analysis failed: {e}")
        return False
    
    # Final database stats
    print("\n9. Final database statistics...")
    try:
        stats = db_service.get_database_stats()
        print(f"✅ Total songs: {stats['total_songs']}")
        print(f"   Unique genres: {stats['genres_count']}")
        print(f"   Tempo range: {stats['tempo_min']:.0f} - {stats['tempo_max']:.0f} BPM")
        print(f"   Average tempo: {stats['tempo_avg']:.0f} BPM")
        
        if stats.get('top_genres'):
            print(f"   Top genres:")
            for genre, count in list(stats['top_genres'].items())[:5]:
                print(f"      - {genre}: {count} songs")
    except Exception as e:
        print(f"❌ Stats retrieval failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    
    # Cleanup test database
    print("\nCleaning up test database...")
    try:
        import os
        if os.path.exists("data/test_msd.db"):
            os.remove("data/test_msd.db")
            print("✅ Test database removed")
    except Exception as e:
        print(f"⚠️ Could not remove test database: {e}")
    
    return True

if __name__ == "__main__":
    success = test_msd_services()
    sys.exit(0 if success else 1)
