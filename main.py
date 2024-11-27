import asyncio
from batch_scheduling_engine import BatchSchedulingEngine

async def main():
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
    slo_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # should be modified to realistic values later !!
    
    # Run the batch scheduling engine
    engine = BatchSchedulingEngine(requests=requests, slo_list=slo_list)
    await engine.process_queue()
    
    # profile and save the results
    engine.analyze_and_record()

if __name__ == "__main__":
    asyncio.run(main())