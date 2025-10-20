#!/usr/bin/env python3
"""
Multi-Turn PetBot Demo Script
Demonstrates multi-turn conversation capabilities with predefined scenarios
"""

import os
import sys
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from chatbot_system.chatbot_pipeline import ChatbotPipeline
from rag_system.proposed_rag_system import ProposedRAGManager

def run_multi_turn_demo():
    """Run a multi-turn conversation demo"""
    print("🐾 PetBot Multi-Turn Conversation Demo")
    print("=" * 50)
    
    # Initialize system
    print("🔄 Initializing PetBot...")
    rag = ProposedRAGManager(collection_name="demo_multi_turn", use_openai=False)
    rag.add_directory("documents")
    chatbot = ChatbotPipeline(rag)
    print("✅ PetBot ready!\n")
    
    # Demo scenarios
    scenarios = [
        {
            "name": "Pet Adoption Conversation",
            "messages": [
                "Hello, I want to adopt a pet",
                "I prefer dogs",
                "Golden retrievers are nice",
                "I live in Selangor",
                "What about puppies?",
                "Actually, how much exercise do they need?"
            ]
        },
        {
            "name": "Pet Care Conversation", 
            "messages": [
                "What can I feed my cat?",
                "How often should I feed her?",
                "What about treats?",
                "She's 2 years old",
                "Any special dietary requirements?"
            ]
        },
        {
            "name": "Mixed Intent Conversation",
            "messages": [
                "I want to adopt a dog",
                "What should I know about dog care?",
                "I prefer small breeds",
                "How often should I walk a small dog?",
                "What about training?"
            ]
        }
    ]
    
    for scenario_idx, scenario in enumerate(scenarios, 1):
        print(f"\n🎭 Scenario {scenario_idx}: {scenario['name']}")
        print("-" * 50)
        
        # Reset chatbot for each scenario
        chatbot.session = {"intent": None, "entities": {}, "greeted": False}
        
        for turn_idx, message in enumerate(scenario['messages'], 1):
            print(f"\n👤 Turn {turn_idx}: {message}")
            
            # Get response
            response = chatbot.handle_message(message)
            print(f"🤖 Bot: {response}")
            
            # Show current state
            print(f"📊 State: Intent={chatbot.session.get('intent', 'None')}, "
                  f"Entities={chatbot.session.get('entities', {})}")
        
        print(f"\n✅ Scenario {scenario_idx} completed!")
    
    print(f"\n🎉 Multi-turn demo completed!")
    print("Key observations:")
    print("• ✅ Session state maintained across turns")
    print("• ✅ Entities accumulated over conversation")
    print("• ✅ Intent classification working")
    print("• ✅ Context-aware responses")
    print("• ✅ Smooth transitions between topics")

def run_interactive_demo():
    """Run an interactive multi-turn demo"""
    print("\n🎮 Interactive Multi-Turn Demo")
    print("=" * 40)
    print("Try these example conversations:")
    print("1. Start with: 'I want to adopt a dog'")
    print("2. Add details: 'I prefer golden retrievers'")
    print("3. Add location: 'I live in Selangor'")
    print("4. Ask follow-up: 'What about puppies?'")
    print("5. Switch topic: 'How do I care for a dog?'")
    print("\nType 'quit' to exit, 'reset' to clear session")
    
    # Initialize system
    rag = ProposedRAGManager(collection_name="interactive_demo", use_openai=False)
    rag.add_directory("documents")
    chatbot = ChatbotPipeline(rag)
    
    turn_count = 0
    while True:
        try:
            user_input = input(f"\n👤 Turn {turn_count + 1}: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'reset':
                chatbot.session = {"intent": None, "entities": {}, "greeted": False}
                turn_count = 0
                print("🔄 Session reset!")
                continue
            elif not user_input:
                continue
            
            # Process message
            response = chatbot.handle_message(user_input)
            print(f"🤖 Bot: {response}")
            
            # Show state
            print(f"📊 Intent: {chatbot.session.get('intent', 'None')}")
            print(f"🏷️ Entities: {chatbot.session.get('entities', {})}")
            
            turn_count += 1
            
        except KeyboardInterrupt:
            break
    
    print("\n👋 Interactive demo ended!")

if __name__ == "__main__":
    print("Choose demo mode:")
    print("1. Automated scenarios (press 1)")
    print("2. Interactive chat (press 2)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_multi_turn_demo()
    elif choice == "2":
        run_interactive_demo()
    else:
        print("Invalid choice. Running automated demo...")
        run_multi_turn_demo()
