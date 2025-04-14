from agent import root_agent

def main():
    """Test the agent with a few queries."""
    
    print("\n=== Testing Weather & Time Agent with Azure OpenAI via LiteLLM ===\n")
    
    # Test queries
    test_queries = [
        "What's the weather in New York?",
        "What time is it in New York?",
        "What's the weather like in London?"
    ]
    
    # Run test queries
    for query in test_queries:
        print(f"Query: {query}")
        response = root_agent.generate_content(query)
        print(f"Response: {response}")
        print("-" * 50)
    
    # Interactive mode
    print("\nEnter 'exit' to quit interactive mode")
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
            
        if not user_input.strip():
            continue
            
        response = root_agent.generate_content(user_input)
        print(f"Agent: {response}")

if __name__ == "__main__":
    main() 