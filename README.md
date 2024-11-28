# Convex Optimization Final Project

## About the project
This project is about running request batches arrived to the LLM server regarding the SLOs provided
Run the command below to run the engine
```
python main.py --strategy OPTIMAL # Can select other strategy if you want
```
I also provided some files to measure the prefill and decode latencies of LLM requests, and build a profiler based on the results 

## File to measure actual latency
Please look at "generate_profile.ipynb"

## File to build a profiler
Please look at "profile_analysis.ipynb"