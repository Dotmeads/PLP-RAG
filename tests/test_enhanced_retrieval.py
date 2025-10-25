#!/usr/bin/env python3
"""
Test script for enhanced retrieval system
"""

import sys
import os
sys.path.append('/Users/dotsnoise/PLP RAG')

from pet_retrieval.enhanced_retrieval import (
    parse_facets_from_text, filter_with_relaxation, 
    make_boosted_query, EnhancedBM25, safe_list_from_cell
)
import pandas as pd

def test_facet_parsing():
    """Test facet parsing from text"""
    print("🧪 Testing facet parsing...")
    
    test_cases = [
        "I want to adopt a dog in Kuala Lumpur",
        "Looking for a cat in Johor",
        "Need a male golden retriever in Selangor",
        "Female black cat in Penang",
        "Puppy in KL"
    ]
    
    for text in test_cases:
        facets = parse_facets_from_text(text)
        print(f"  '{text}' -> {facets}")
    
    print("✅ Facet parsing test completed\n")

def test_data_parsing():
    """Test robust data parsing"""
    print("🧪 Testing data parsing...")
    
    test_cases = [
        '["https://photo1.jpg", "https://photo2.jpg"]',
        "https://photo1.jpg, https://photo2.jpg",
        "https://photo1.jpg",
        None,
        "",
        '["black", "white", "brown"]',
        "black, white, brown"
    ]
    
    for case in test_cases:
        result = safe_list_from_cell(case)
        print(f"  {case} -> {result}")
    
    print("✅ Data parsing test completed\n")

def test_enhanced_bm25():
    """Test enhanced BM25 implementation"""
    print("🧪 Testing Enhanced BM25...")
    
    # Create sample documents
    doc_map = {
        0: "dog puppy canine pet adoption",
        1: "cat kitten feline pet adoption", 
        2: "dog golden retriever friendly",
        3: "cat black domestic short hair",
        4: "dog labrador retriever family"
    }
    
    bm25 = EnhancedBM25()
    bm25.fit(doc_map)
    
    # Test search
    query = "dog golden retriever"
    results = bm25.search(query, topk=3)
    
    print(f"  Query: '{query}'")
    print(f"  Results: {results}")
    
    print("✅ Enhanced BM25 test completed\n")

def test_filtering():
    """Test filtering with sample data"""
    print("🧪 Testing filtering...")
    
    # Create sample dataframe
    data = {
        'name': ['Buddy', 'Whiskers', 'Max', 'Luna', 'Charlie'],
        'animal': ['Dog', 'Cat', 'Dog', 'Cat', 'Dog'],
        'breed': ['Golden Retriever', 'Domestic Short Hair', 'Labrador', 'Persian', 'Mixed'],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'state': ['Kuala Lumpur', 'Johor', 'Selangor', 'Penang', 'Kuala Lumpur'],
        'colors_canonical': ['["golden", "brown"]', '["black", "white"]', '["brown"]', '["white"]', '["brown", "black"]']
    }
    
    df = pd.DataFrame(data)
    print(f"  Sample data:\n{df}")
    
    # Test filtering
    facets = {"animal": "dog", "state": "kuala lumpur"}
    results, used = filter_with_relaxation(df, facets, ["state", "gender"], min_floor=1)
    
    print(f"  Facets: {facets}")
    print(f"  Used: {used}")
    print(f"  Results:\n{results}")
    
    print("✅ Filtering test completed\n")

def main():
    """Run all tests"""
    print("🚀 Testing Enhanced Retrieval System\n")
    
    test_facet_parsing()
    test_data_parsing()
    test_enhanced_bm25()
    test_filtering()
    
    print("🎉 All tests completed!")

if __name__ == "__main__":
    main()
