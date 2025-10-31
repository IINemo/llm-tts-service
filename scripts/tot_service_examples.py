#!/usr/bin/env python3
"""
Examples of using Tree-of-Thoughts through the service API.

This shows different ways to send requests to the ToT service.
"""

import json

import requests

# Configuration
SERVICE_URL = "http://localhost:8888"  # Adjust port if needed


def example_1_simple_request():
    """Example 1: Simple ToT request with default parameters."""
    print("=" * 80)
    print("EXAMPLE 1: Simple ToT Request")
    print("=" * 80)

    response = requests.post(
        f"{SERVICE_URL}/v1/chat/completions",
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "What are 3 ways to improve code quality?"}
            ],
            "tts_strategy": "tree_of_thoughts",  # or "tot"
        },
        headers={"Content-Type": "application/json"},
        timeout=300,
    )

    if response.status_code == 200:
        result = response.json()
        print("\n✓ Success!")
        print(f"\nAnswer:\n{result['choices'][0]['message']['content']}")
        print(
            f"\nMetadata:\n{json.dumps(result['choices'][0].get('tts_metadata', {}), indent=2)}"
        )
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(response.text)


def example_2_custom_parameters():
    """Example 2: ToT with custom parameters for more thorough search."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: ToT with Custom Parameters")
    print("=" * 80)

    response = requests.post(
        f"{SERVICE_URL}/v1/chat/completions",
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": "Design a database schema for a social media platform",
                }
            ],
            # ToT-specific parameters
            "tts_strategy": "tree_of_thoughts",
            "tot_mode": "generic",  # generic or game24
            "tot_method_generate": "propose",  # propose or sample
            "tot_beam_width": 3,  # Keep top 3 states at each step
            "tot_n_generate_sample": 5,  # Generate 5 candidates per state
            "tot_steps": 4,  # 4 reasoning steps
            "tot_max_tokens_per_step": 200,  # Max tokens per step
            "temperature": 0.7,
        },
        headers={"Content-Type": "application/json"},
        timeout=300,
    )

    if response.status_code == 200:
        result = response.json()
        metadata = result["choices"][0].get("tts_metadata", {})
        print("\n✓ Success!")
        print(f"\nAnswer:\n{result['choices'][0]['message']['content']}")
        print(f"\nAPI Calls: {metadata.get('total_api_calls')}")
        print(f"Best Score: {metadata.get('best_score')}")
        print(f"Scorer Evaluations: {metadata.get('scorer_evaluations')}")
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(response.text)


def example_5_with_reasoning_tree():
    """Example 5: ToT with full reasoning tree for visualization."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: ToT with Reasoning Tree Data")
    print("=" * 80)

    response = requests.post(
        f"{SERVICE_URL}/v1/chat/completions",
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": "What are 3 best practices for code reviews?",
                }
            ],
            "tts_strategy": "tree_of_thoughts",
            "tot_beam_width": 2,
            "tot_steps": 3,
            # Request full reasoning tree for visualization
            "include_reasoning_tree": True,
        },
        headers={"Content-Type": "application/json"},
        timeout=300,
    )

    if response.status_code == 200:
        result = response.json()
        metadata = result["choices"][0].get("tts_metadata", {})
        print("\n✓ Success!")
        print(f"\nAnswer:\n{result['choices'][0]['message']['content'][:200]}...")

        # Check if reasoning tree is included
        if "reasoning_tree" in metadata:
            tree = metadata["reasoning_tree"]
            print("\n📊 Reasoning Tree:")
            print(f"  - Total nodes: {len(tree['nodes'])}")
            print(f"  - Total edges: {len(tree['edges'])}")
            print(f"  - Question: {tree['question'][:60]}...")
            print("\n  First 3 nodes:")
            for node in tree["nodes"][:3]:
                print(
                    f"    Node {node['id']}: step={node['step']}, score={node['score']:.2f}, selected={node['is_selected']}"
                )
            print(
                "\n  💡 This data can be sent to the frontend for interactive visualization!"
            )
        else:
            print("\n⚠ No reasoning tree in response")
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(response.text)


def example_3_using_openai_sdk():
    """Example 3: Using OpenAI Python SDK (recommended)."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Using OpenAI SDK")
    print("=" * 80)

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=f"{SERVICE_URL}/v1", api_key="dummy"
        )  # API key not required

        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "What are the pros and cons of microservices?",
                }
            ],
            extra_body={
                "tts_strategy": "tree_of_thoughts",
                "tot_beam_width": 2,
                "tot_steps": 3,
            },
        )

        print("\n✓ Success!")
        print(f"\nAnswer:\n{response.choices[0].message.content}")
        if hasattr(response.choices[0], "tts_metadata"):
            print(
                f"\nMetadata:\n{json.dumps(response.choices[0].tts_metadata, indent=2)}"
            )
    except ImportError:
        print("\n⚠ OpenAI SDK not installed. Install with: pip install openai")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def example_4_curl_command():
    """Example 4: Show equivalent curl command."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Equivalent curl command")
    print("=" * 80)

    curl_command = f"""
curl -X POST {SERVICE_URL}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "openai/gpt-4o-mini",
    "messages": [
      {{"role": "user", "content": "What are best practices for REST API design?"}}
    ],
    "tts_strategy": "tree_of_thoughts",
    "tot_beam_width": 3,
    "tot_steps": 3,
    "tot_method_generate": "propose"
  }}'
"""
    print(curl_command)


if __name__ == "__main__":
    print("\nTree-of-Thoughts Service API Examples")
    print("=" * 80)
    print(f"Service URL: {SERVICE_URL}")
    print("\nMake sure the service is running:")
    print(
        "  python -m uvicorn service_app.main:app --host 0.0.0.0 --port 8888 --reload"
    )
    print("=" * 80)

    # Check if service is running
    try:
        health = requests.get(f"{SERVICE_URL}/health", timeout=5)
        if health.status_code == 200:
            print("\n✓ Service is running!")
        else:
            print(f"\n⚠ Service returned status: {health.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Service is not running at {SERVICE_URL}")
        print("Please start it first!")
        exit(1)

    # Run examples
    try:
        example_1_simple_request()
        # Uncomment to run other examples:
        # example_2_custom_parameters()
        # example_3_using_openai_sdk()
        example_4_curl_command()
        # example_5_with_reasoning_tree()  # Uncomment to see full tree data
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
