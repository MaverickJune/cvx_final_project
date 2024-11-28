import asyncio
import argparse
from batch_scheduling_engine import BatchSchedulingEngine

def argparser():
    parser = argparse.ArgumentParser(description="Batch scheduling engine")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pad_token", type=str, default="<|pad|>", help="Pad token")
    parser.add_argument("--strategy", type=str, default="OPTIMAL", choices=["OPTIMAL", "RANDOM", "SRTF", "FIFO"], help="Scheduling strategy")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum new tokens")
    parser.add_argument("--penalty", type=str, default="quadratic", help="Penalty function")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    return parser

async def main():
    arparser = argparser()
    args = arparser.parse_args()
    
    # Define requests and SLOs
    requests = [
        {"question": "What’s your favorite color?"},
        {"question": "Do you prefer mornings or evenings?"},
        {"question": "What’s the most exciting place you’ve visited?"},
        {"question": "If you could learn a new language instantly, which one would it be and why?"},
        {"question": "What’s a habit or routine you’ve developed that makes your day better?"},
        {"question": "If you could redesign your city’s public spaces, what would you change to make them more inclusive and functional?"},
        {"question": "Imagine you were given the opportunity to create a documentary. What topic would you choose, and why do you think it’s important to share with the world?"},
        {"question": "If you had the chance to bring back one extinct species, which would it be, and how would its return benefit the planet?"},
        {"question": "Suppose you were tasked with developing a program to teach empathy and kindness in schools. What activities or lessons would you include, and how would you measure its success?"},
        {"question": "If you could collaborate with a global organization to tackle a major issue like climate change or poverty, what strategies would you propose, and how would you ensure long-term positive outcomes?"}
    ]
    slo_list = [23, 18, 3, 13, 28, 5, 10, 12, 19, 20] 
    
    # Run the batch scheduling engine
    engine = BatchSchedulingEngine(requests=requests, slo_list=slo_list, strategy=args.strategy, debug=False)
    await engine.process_queue()
    
    # profile and save the results
    engine.analyze_and_record()
    print("Recorded the results")

if __name__ == "__main__":
    asyncio.run(main())
    
# slo reference
"""
# total latency measured 
2.88212266998291,
2.8948785522460936,
2.9123132751464844,
2.951351406860352,
2.9471540191650387,
3.0605486785888676,
3.0861006866455076,
3.070599656677246,
3.08922907409668,
3.0943182128906246
"""

"""
# shuffled, cumsum, and reordered
26.902515545654296,
20.894812847900393,
3.08922907409668,
15.087621020507815,
29.988616232299805,
6.159828730773926,
9.041951400756837,
11.993302807617189,
17.98249957275391,
23.84196686706543
"""