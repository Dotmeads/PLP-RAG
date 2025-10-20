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
    
    # Check for connection string
    conn_str = os.getenv('AZURE_CONNECTION_STRING')
    if not conn_str:
        print("❌ No AZURE_CONNECTION_STRING environment variable found")
        print("\n📝 To test Azure connection, you need:")
        print("1. Azure Storage Account connection string")
        print("2. Set environment variable: export AZURE_CONNECTION_STRING='your_connection_string'")
        print("3. Or create .streamlit/secrets.toml with Azure credentials")
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
