#!/usr/bin/env python3
"""
Test Azure Blob Storage Connection
Run this to check if Azure credentials are working
"""

import os
import sys
from azure.storage.blob import BlobServiceClient

def test_azure_connection():
    """Test Azure Blob Storage connection"""
    
    print("🔍 Testing Azure Blob Storage Connection...")
    print("=" * 50)
    
    # Try to get connection string from Streamlit secrets first, then environment
    conn_str = None
    
    # Try Streamlit secrets
    try:
        import streamlit as st
        conn_str = st.secrets.get("AZURE_CONNECTION_STRING")
        print("✅ Found Azure credentials in .streamlit/secrets.toml")
    except Exception:
        pass
    
    # Fallback to environment variable
    if not conn_str:
        conn_str = os.getenv('AZURE_CONNECTION_STRING')
        if conn_str:
            print("✅ Found Azure credentials in environment variables")
    
    if not conn_str:
        print("❌ No Azure credentials found")
        print("\n📝 To test Azure connection, you need:")
        print("1. Create .streamlit/secrets.toml with Azure credentials")
        print("2. Or set environment variable: export AZURE_CONNECTION_STRING='your_connection_string'")
        print("\nExample .streamlit/secrets.toml:")
        print("AZURE_CONNECTION_STRING = \"DefaultEndpointsProtocol=https;AccountName=...\"")
        return False
    
    try:
        # Test connection
        print(f"🔗 Testing connection to Azure Storage...")
        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        
        # List containers
        print("📦 Listing containers...")
        containers = list(blob_service_client.list_containers())
        
        if not containers:
            print("⚠️  No containers found in storage account")
            return False
            
        print(f"✅ Found {len(containers)} containers:")
        for container in containers:
            print(f"   - {container.name}")
        
        # Check for required containers
        container_names = [c.name for c in containers]
        required_containers = ['ml-artifacts', 'pets-data']
        
        print(f"\n🔍 Checking for required containers...")
        for req_container in required_containers:
            if req_container in container_names:
                print(f"✅ Found: {req_container}")
            else:
                print(f"❌ Missing: {req_container}")
        
        print(f"\n🎉 Azure connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ Azure connection failed: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Check connection string format")
        print("2. Verify storage account exists and is accessible")
        print("3. Check network connectivity")
        return False

if __name__ == "__main__":
    success = test_azure_connection()
    sys.exit(0 if success else 1)
