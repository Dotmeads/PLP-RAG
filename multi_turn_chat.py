#!/usr/bin/env python3
"""
Multi-Turn PetBot Chat Script
Interactive command-line interface for testing multi-turn conversations
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from chatbot_flow.chatbot_pipeline import ChatbotPipeline
from rag_system.proposed_rag_system import ProposedRAGManager

class MultiTurnPetBot:
    """Enhanced multi-turn conversation handler"""
    
    def __init__(self):
        """Initialize the PetBot system"""
        print("🐾 Initializing PetBot Multi-Turn Chat...")
        
        # Initialize RAG system
        self.rag = ProposedRAGManager(
            collection_name="multi_turn_chat", 
            use_openai=False
        )
        self.rag.add_directory("documents")
        
        # Initialize chatbot
        self.chatbot = ChatbotPipeline(self.rag)
        
        # Conversation history
        self.conversation_history: List[Dict[str, Any]] = []
        
        print("✅ PetBot ready! Type 'help' for commands, 'quit' to exit.\n")
    
    def display_help(self):
        """Display available commands"""
        print("\n📋 Available Commands:")
        print("  help     - Show this help message")
        print("  history  - Show conversation history")
        print("  clear    - Clear conversation history")
        print("  state    - Show current session state")
        print("  stats    - Show conversation statistics")
        print("  quit     - Exit the chat")
        print("\n💬 Just type your message to chat!")
        print("   Examples:")
        print("   • 'I want to adopt a dog'")
        print("   • 'What can I feed my cat?'")
        print("   • 'I prefer golden retrievers'")
        print("   • 'How often should I walk my dog?'")
    
    def display_history(self):
        """Display conversation history"""
        if not self.conversation_history:
            print("📝 No conversation history yet.")
            return
        
        print(f"\n📚 Conversation History ({len(self.conversation_history)} turns):")
        print("-" * 60)
        
        for i, turn in enumerate(self.conversation_history, 1):
            timestamp = turn['timestamp'].strftime("%H:%M:%S")
            print(f"{i:2d}. [{timestamp}] 👤 You: {turn['user_input']}")
            print(f"    🤖 Bot: {turn['bot_response'][:100]}{'...' if len(turn['bot_response']) > 100 else ''}")
            print(f"    🎯 Intent: {turn['intent']} | 🏷️ Entities: {turn['entities']}")
            print()
    
    def display_state(self):
        """Display current session state"""
        print(f"\n🔍 Current Session State:")
        print(f"   Intent: {self.chatbot.session.get('intent', 'None')}")
        print(f"   Entities: {self.chatbot.session.get('entities', {})}")
        print(f"   Greeted: {self.chatbot.session.get('greeted', False)}")
    
    def display_stats(self):
        """Display conversation statistics"""
        if not self.conversation_history:
            print("📊 No conversation data yet.")
            return
        
        # Count intents
        intents = [turn['intent'] for turn in self.conversation_history]
        intent_counts = {}
        for intent in intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        # Count total entities
        total_entities = 0
        for turn in self.conversation_history:
            total_entities += len(turn['entities'])
        
        print(f"\n📊 Conversation Statistics:")
        print(f"   Total turns: {len(self.conversation_history)}")
        print(f"   Intent distribution: {intent_counts}")
        print(f"   Total entities extracted: {total_entities}")
        print(f"   Average entities per turn: {total_entities/len(self.conversation_history):.1f}")
    
    def clear_history(self):
        """Clear conversation history and reset session"""
        self.conversation_history.clear()
        self.chatbot.session = {"intent": None, "entities": {}, "greeted": False}
        print("🧹 Conversation history cleared and session reset!")
    
    def process_message(self, user_input: str) -> str:
        """Process a user message and return bot response"""
        # Get bot response
        bot_response = self.chatbot.handle_message(user_input)
        
        # Extract current state
        current_intent = self.chatbot.session.get('intent', 'unknown')
        current_entities = self.chatbot.session.get('entities', {})
        
        # Store in history
        turn = {
            'timestamp': datetime.now(),
            'user_input': user_input,
            'bot_response': bot_response,
            'intent': current_intent,
            'entities': current_entities.copy()
        }
        self.conversation_history.append(turn)
        
        return bot_response
    
    def run(self):
        """Main conversation loop"""
        print("🐾 Welcome to PetBot Multi-Turn Chat!")
        print("=" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                # Handle commands
                if user_input.lower() == 'quit':
                    print("\n👋 Thanks for chatting with PetBot! Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    self.display_help()
                    continue
                elif user_input.lower() == 'history':
                    self.display_history()
                    continue
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                elif user_input.lower() == 'state':
                    self.display_state()
                    continue
                elif user_input.lower() == 'stats':
                    self.display_stats()
                    continue
                elif not user_input:
                    continue
                
                # Process message
                print("🤖 Bot: ", end="", flush=True)
                response = self.process_message(user_input)
                print(response)
                
                # Show current state (optional)
                if len(self.conversation_history) % 3 == 0:  # Every 3rd turn
                    print(f"\n💡 Current entities: {self.chatbot.session.get('entities', {})}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'help' for assistance.")

def main():
    """Main function"""
    try:
        chat = MultiTurnPetBot()
        chat.run()
    except Exception as e:
        print(f"❌ Failed to initialize PetBot: {e}")
        print("Make sure you're in the correct directory and all dependencies are installed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
