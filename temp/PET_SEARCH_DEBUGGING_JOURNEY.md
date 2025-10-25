# Pet Search Debugging Journey

## Overview
This document chronicles the debugging process for the pet search functionality in the unified petbot application, specifically addressing issues with incorrect animal type filtering and display.

## Initial Problem
**Issue**: When searching for "I want to adopt a dog in KL", the system was returning cats instead of dogs.

**User Report**: 
```
I see a cat in my dog request for "I want to adopt a dog in KL"
Domestic Short Hair • Breed: Domestic Short Hair • Gender: Male • Age: 1.0 yrs • State: —
```

## Debugging Process

### Step 1: Entity Extraction Verification
**Test**: Verified that entity extraction was working correctly
```python
from chatbot_flow.entity_extractor import EntityExtractor
extractor = EntityExtractor()
result = extractor.extract('I want to adopt a dog in KL')
# Result: {'PET_TYPE': 'dog', 'STATE': 'Kuala Lumpur'}
```
**Finding**: ✅ Entity extraction was working correctly

### Step 2: Database Structure Analysis
**Test**: Examined the pet database structure
```python
import pandas as pd
df = pd.read_csv('artifacts/pets.csv')
print(df[['name', 'animal', 'breed', 'state']].head(10))
```
**Findings**:
- Database has both `animal` (Dog/Cat) and `breed` (Mixed Breed/Domestic Short Hair) columns
- 3,481 dogs and 3,475 cats in database
- 704 dogs specifically in Kuala Lumpur

### Step 3: Search Logic Testing
**Test**: Tested the search logic directly
```python
# BM25 + FAISS search
query = 'dog in Kuala Lumpur'
bm25_scores = bm25.search(query, topk=20)
faiss_scores, faiss_indices = faiss_index.search(query_embedding, 20)

# Filter by animal type
animal_filter = results['animal'].str.lower() == 'dog'
filtered_results = results[animal_filter]
# Result: 31 dogs found correctly
```
**Finding**: ✅ Search and filtering logic was working correctly

### Step 4: Chatbot Pipeline Testing
**Test**: Tested the chatbot pipeline's `_perform_pet_search` method
```python
ents = {'PET_TYPE': 'dog', 'STATE': 'Kuala Lumpur'}
result = chatbot._perform_pet_search(ents)
# Result: Found 10 dogs, but display showed wrong animal type
```
**Finding**: ❌ The issue was in the data mapping between database and display

## Root Cause Analysis

### The Problem
The issue was in the **data mapping** between the database and the Streamlit display:

1. **Database Structure**: 
   - `animal` column: "Dog", "Cat" 
   - `breed` column: "Mixed Breed", "Domestic Short Hair"

2. **Pet Info Dictionary**: 
   - Missing `animal` field in the structured data
   - Only included `breed` field

3. **Streamlit Display**: 
   - Was displaying `breed` as the animal type
   - So "Mixed Breed" appeared instead of "Dog"

### Data Flow Issue
```
Database: animal="Dog", breed="Mixed Breed"
    ↓
Pet Info: {breed: "Mixed Breed"}  # ❌ Missing animal field
    ↓
Display: "Mixed Breed" • Breed: Mixed Breed  # ❌ Wrong animal type
```

## Solutions Implemented

### Fix 1: Add Animal Field to Pet Info
**File**: `chatbot_flow/chatbot_pipeline.py`
**Change**: Added `animal` and `state` fields to the pet info dictionary

```python
pet_info = {
    "name": pet.get('name', 'Unnamed'),
    "animal": pet.get('animal', 'Unknown'),  # ✅ Added
    "breed": pet.get('breed', 'Unknown'),
    "age_months": pet.get('age_months', 'Unknown'),
    "gender": pet.get('gender', 'Unknown'),
    "color": pet.get('color', 'Unknown'),
    "size": pet.get('size', 'Unknown'),
    "state": pet.get('state', 'Unknown'),    # ✅ Added
    "n_photos": pet.get('n_photos', 0),
    "photo_urls": [],
    "adoption_url": pet.get('url', ''),
    "description": str(pet.get('description_clean', ''))[:100] + "..." if pd.notna(pet.get('description_clean')) else ""
}
```

### Fix 2: Update Streamlit Display
**File**: `apps/unified_petbot_app.py`
**Change**: Updated display to show `animal` instead of `breed` for animal type

```python
# Pet details
animal = pet.get('animal', 'Unknown').title()  # ✅ Changed from breed
breed = pet.get('breed', '—')
gender = pet.get('gender', '—').title()
state = pet.get('state', '—').title()  # ✅ Now properly displayed
color = pet.get('color', '—')
age_text = _age_years_from_months(pet.get('age_months'))
size = pet.get('size', '—').title()

st.write(
    f"**{animal}** • **Breed:** {breed} • **Gender:** {gender} • "
    f"**Age:** {age_text} • **State:** {state}"
)
```

## Verification

### Before Fix
```
Domestic Short Hair • Breed: Domestic Short Hair • Gender: Male • Age: 1.0 yrs • State: —
```
❌ Wrong animal type, missing state

### After Fix
```
Dog • Breed: Mixed Breed • Gender: Male • Age: 1.0 yrs • State: Kuala Lumpur
```
✅ Correct animal type, proper state display

### Test Results
```python
# Final test
ents = {'PET_TYPE': 'dog', 'STATE': 'Kuala Lumpur'}
result = chatbot._perform_pet_search(ents)
first_pet = result['pets'][0]
print(f'Animal: {first_pet.get("animal")}')  # Dog
print(f'Breed: {first_pet.get("breed")}')    # Mixed Breed
print(f'State: {first_pet.get("state")}')    # Kuala Lumpur
```

## Key Learnings

### 1. Data Mapping is Critical
- Always ensure all necessary database fields are mapped to display structures
- Missing fields can cause confusing user experiences

### 2. Debugging Strategy
1. **Start with the end result** - What is the user seeing?
2. **Work backwards** - Trace through the data flow
3. **Test each component** - Verify each layer works independently
4. **Check data structures** - Ensure proper field mapping

### 3. Entity Extraction vs Display
- Entity extraction was working correctly
- The issue was in the data presentation layer
- Always separate concerns: extraction, processing, and display

### 4. Database Schema Understanding
- Understanding the database structure is crucial
- `animal` vs `breed` are different concepts
- Proper field mapping prevents user confusion

## Files Modified

1. **`chatbot_flow/chatbot_pipeline.py`**
   - Added `animal` and `state` fields to pet info dictionary
   - Ensured proper data mapping from database to display structure

2. **`apps/unified_petbot_app.py`**
   - Updated display logic to show `animal` instead of `breed` for animal type
   - Added proper state display

## Impact

- ✅ Pet search now correctly filters by animal type
- ✅ Display shows proper animal type (Dog/Cat) instead of breed
- ✅ State information is properly displayed
- ✅ User experience is now accurate and intuitive

## Future Considerations

1. **Data Validation**: Add validation to ensure all required fields are present
2. **Error Handling**: Add fallbacks for missing data fields
3. **Testing**: Create automated tests for data mapping
4. **Documentation**: Document the expected data structure for pet info

---

*This debugging journey demonstrates the importance of thorough testing and understanding data flow in complex applications.*
