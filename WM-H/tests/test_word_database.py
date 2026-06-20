"""
Unit tests for WordDatabase
"""

import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.word_database import WordDatabase


def test_create_database():
    """Test database creation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = WordDatabase(db_path)
        
        # Check if database file was created
        assert os.path.exists(db_path)
        
        db.close()


def test_initialize_with_seeds():
    """Test initialization with seed words"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        
        with WordDatabase(db_path) as db:
            seed_verbs = ["pick", "grab", "move"]
            seed_nouns = ["apple", "ball", "box"]
            seed_adjectives = ["red", "blue", "small"]
            
            db.initialize_with_seeds(seed_verbs, seed_nouns, seed_adjectives)
            
            # Check statistics
            stats = db.get_statistics()
            assert stats['verb']['count'] == 3
            assert stats['noun']['count'] == 3
            assert stats['adjective']['count'] == 3


def test_add_instruction():
    """Test adding instruction and updating frequency"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        
        with WordDatabase(db_path) as db:
            # Initialize
            db.initialize_with_seeds(
                seed_verbs=["pick"],
                seed_nouns=["apple"],
                seed_adjectives=["red"]
            )
            
            # Add instruction
            instruction = "pick red apple"
            words = db.add_instruction(instruction, mode="test")
            
            # Check that instruction was recorded
            stats = db.get_statistics()
            assert stats['total_instructions'] == 1


def test_get_words_by_type():
    """Test getting words by type"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        
        with WordDatabase(db_path) as db:
            db.initialize_with_seeds(
                seed_verbs=["pick", "grab"],
                seed_nouns=["apple"],
                seed_adjectives=["red"]
            )
            
            # Get verbs
            verbs = db.get_words_by_type('verb')
            assert len(verbs) == 2
            
            # Check format (word, frequency)
            assert all(len(item) == 2 for item in verbs)


def test_get_low_frequency_words():
    """Test getting low frequency words"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        
        with WordDatabase(db_path) as db:
            db.initialize_with_seeds(
                seed_verbs=["pick", "grab", "move"],
                seed_nouns=["apple"],
                seed_adjectives=["red"]
            )
            
            # All words start with 0 frequency, so all are low frequency
            low_freq = db.get_low_frequency_words(threshold=3)
            assert len(low_freq) > 0


def test_instruction_exists():
    """Test checking if instruction already exists"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        
        with WordDatabase(db_path) as db:
            db.initialize_with_seeds(
                seed_verbs=["pick"],
                seed_nouns=["apple"],
                seed_adjectives=["red"]
            )
            
            instruction = "pick red apple"
            
            # Instruction should not exist initially
            assert not db.instruction_exists(instruction)
            
            # Add instruction
            db.add_instruction(instruction, mode="test")
            
            # Now it should exist
            assert db.instruction_exists(instruction)
            
            # Adding again should still return True (duplicate)
            assert db.instruction_exists(instruction)


def test_context_manager():
    """Test using database as context manager"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        
        with WordDatabase(db_path) as db:
            db.initialize_with_seeds(
                seed_verbs=["pick"],
                seed_nouns=["apple"],
                seed_adjectives=["red"]
            )
            stats = db.get_statistics()
            assert stats['verb']['count'] == 1
        
        # Database should be closed after exiting context


if __name__ == '__main__':
    # Run tests
    test_create_database()
    print("✓ test_create_database passed")
    
    test_initialize_with_seeds()
    print("✓ test_initialize_with_seeds passed")
    
    test_add_instruction()
    print("✓ test_add_instruction passed")
    
    test_get_words_by_type()
    print("✓ test_get_words_by_type passed")
    
    test_get_low_frequency_words()
    print("✓ test_get_low_frequency_words passed")
    
    test_instruction_exists()
    print("✓ test_instruction_exists passed")
    
    test_context_manager()
    print("✓ test_context_manager passed")
    
    print("\nAll tests passed!")
