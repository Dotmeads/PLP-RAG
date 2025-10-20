# Azure Configuration Guide

This guide explains how to set up Azure Blob Storage for the Unified PetBot system.

## 🔧 **Azure Setup Requirements**

### **1. Azure Blob Storage Account**
You need an Azure Storage Account with two containers:

#### **ML Artifacts Container**
- **Purpose**: Stores ML models, FAISS indices, and other artifacts
- **Contents**:
  - NER model files (tokenizer, model weights, config)
  - Matching/Ranking model files
  - FAISS index files
  - Document embeddings

#### **Pets Container**
- **Purpose**: Stores pet data and metadata
- **Contents**:
  - Pet CSV file with adoption data
  - Pet photos and videos (optional)

### **2. Required Environment Variables**

Create a `.streamlit/secrets.toml` file in your project root:

```toml
# Azure Blob Storage Configuration
AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=your_account;AccountKey=your_key;EndpointSuffix=core.windows.net"

# Container Names
ML_ARTIFACTS_CONTAINER = "ml-artifacts"
PETS_CONTAINER = "pets-data"

# Blob Prefixes (folders within containers)
NER_PREFIX = "ner"
MR_PREFIX = "mr"

# Pet Data Blob
PETS_CSV_BLOB = "pets.csv"
```

### **3. Alternative: Environment Variables**

Instead of `secrets.toml`, you can use environment variables:

```bash
export AZURE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=your_account;AccountKey=your_key;EndpointSuffix=core.windows.net"
export ML_ARTIFACTS_CONTAINER="ml-artifacts"
export PETS_CONTAINER="pets-data"
export NER_PREFIX="ner"
export MR_PREFIX="mr"
export PETS_CSV_BLOB="pets.csv"
```

## 📁 **Expected Blob Structure**

### **ML Artifacts Container**
```
ml-artifacts/
├── ner/
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   ├── pytorch_model.bin (or model.safetensors)
│   └── special_tokens_map.json
└── mr/
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── vocab.txt
    ├── pytorch_model.bin (or model.safetensors)
    ├── doc_ids.npy
    ├── doc_embeddings.npy
    └── pets_hnsw.index (optional FAISS index)
```

### **Pets Container**
```
pets-data/
└── pets.csv
```

## 🗂️ **Pet CSV Format**

The `pets.csv` file should contain the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `pet_id` | int | Unique pet identifier |
| `name` | str | Pet name |
| `animal` | str | Animal type (dog, cat, etc.) |
| `breed` | str | Breed name |
| `gender` | str | male/female |
| `state` | str | Location state |
| `age_months` | float | Age in months |
| `color` | str | Primary color |
| `colors_canonical` | str | JSON list of colors |
| `size` | str | Size category |
| `fur_length` | str | Fur length |
| `condition` | str | Health condition |
| `vaccinated` | str | Vaccination status |
| `dewormed` | str | Deworming status |
| `neutered` | str | Neutering status |
| `spayed` | str | Spaying status |
| `description_clean` | str | Clean description |
| `photo_links` | str | JSON list of photo URLs |
| `video_links` | str | JSON list of video URLs |
| `url` | str | Pet profile URL |
| `doc` | str | Full text for search |

## 🚀 **Getting Started**

### **1. Set up Azure Storage Account**
1. Create an Azure Storage Account
2. Create the required containers
3. Upload your ML models and pet data
4. Get the connection string

### **2. Configure Secrets**
Create `.streamlit/secrets.toml` with your Azure credentials

### **3. Run the App**
```bash
source .venv/bin/activate
pip install -r requirements_stable.txt
streamlit run unified_petbot_app.py
```

## 🔒 **Security Notes**

- **Never commit** your connection string or account keys to version control
- Use **Azure Key Vault** for production deployments
- Consider using **Managed Identity** for Azure-hosted applications
- Rotate your storage account keys regularly

## 🛠️ **Troubleshooting**

### **Common Issues**

1. **"Missing secrets" error**
   - Ensure `.streamlit/secrets.toml` exists and is properly formatted
   - Check that all required keys are present

2. **"Blob not found" error**
   - Verify container names and blob paths
   - Check that files are uploaded to the correct locations

3. **"Model loading failed" error**
   - Ensure model files are complete and uncorrupted
   - Check that the model format matches expected structure

4. **"FAISS index not found" error**
   - This is not critical - the system will fall back to brute-force search
   - Upload FAISS index files for better performance

### **Debug Mode**

Enable debug mode in the app to see detailed error messages and blob listings.

## 📊 **Performance Tips**

1. **Use FAISS indices** for faster vector search
2. **Optimize pet CSV** by removing unnecessary columns
3. **Compress large files** before uploading to Azure
4. **Use Azure CDN** for faster model downloads
5. **Cache models locally** to avoid repeated downloads

## 🔄 **Fallback Mode**

If Azure components fail to load, the system will:
- Disable pet search functionality
- Continue to work with RAG system for pet care Q&A
- Show appropriate error messages in the UI

This ensures the core functionality remains available even without Azure integration.
