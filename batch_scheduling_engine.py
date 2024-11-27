import torch
import json
import asyncio
import numpy as np
import cvxpy as cp
from queue import Queue
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import format_example, count_tokens, prepare_dataset_for_speed_eval, seed_everything, quadratic, linear

class BatchSchedulingEngine:
    def __init__(self, model_name, batch_size=4, seed=42, pad_token="<|pad|>", strategy="OPTIMAL", max_new_tokens=50, 
                 requests: dict={}, slo_list: list=[], penalty="quadratic"):
        seed_everything(seed)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer.add_special_tokens({"pad_token": pad_token})
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer

        if model_name != "meta-llama/Llama-3.1-8B-Instruct":
            raise NotImplementedError("Only Llama-3.1-8B-Instruct is supported for now")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map = "auto"
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
        self.model = model
        
        self.batch_size = batch_size
        self.datasets = []
        self.dataset_lengths_in_tokens = []
        self.num_requests = len(requests)
        self.run_queue = Queue()
        self.processing = False # ?
        self.register_requests(requests)
        self.max_new_tokens = max_new_tokens
        
        if strategy not in ["OPTIMAL", "RANDOM", "SRTF", "FIFO"]:
            raise ValueError("Invalid scheduling strategy")
        self.strategy = strategy
        
        with open("parameters.json") as f:
            self.parameters = json.load(f)
            self.decode_params = self.parameters["linear"]
            self.prefill_params = self.parameters["quadratic"]
            
        self.prefill_latency_pred = []
        self.decode_latency_pred = []
        self.run_profile()
        
        self.slo_list = slo_list # should be represented in seconds (not ms)
        if penalty in ["quadratic"]:
            self.penalty = penalty
        else:
            raise NotImplementedError("Only quadratic penalty function is supported for now")
        self.generate_schedule()
            
        
    def register_requests(self, questions: dict):
        datasets = [prepare_dataset_for_speed_eval(question, self.batch_size, self.tokenizer).to("cuda") for question in questions]
        dataset_lengths_in_tokens = [count_tokens(format_example(question, self.tokenizer), self.tokenizer) for question in questions]
        self.datasets.extend(datasets)
        self.dataset_lengths_in_tokens.extend(dataset_lengths_in_tokens)
    
    
    def run_profile(self):
        prefill_latencies_pred = quadratic(np.array(self.dataset_lengths_in_tokens), *self.prefill_params).tolist()
        decoding_latencies_pred = [item * self.max_new_tokens for item in linear(np.array(self.dataset_lengths_in_tokens), *self.decode_params).tolist()]
        
        # convert to seconds
        prefill_latencies_pred = [item/1000 for item in prefill_latencies_pred]
        decoding_latencies_pred = [item/1000 for item in decoding_latencies_pred]
        
        self.prefill_latency_pred.extend(prefill_latencies_pred)
        self.decode_latency_pred.extend(decoding_latencies_pred)
    
        
    def generate_schedule(self) -> list[int]:
        if self.strategy == "OPTIMAL":
            process_time = np.array([sum(item) for item in zip(self.prefill_latency_pred, self.decode_latency_pred)])
            T = np.array(self.slo_list)
            
            n = len(self.datasets)
            s = cp.Variable(n, nonneg=True)
            f = s + process_time
            y = cp.Variable((n, n), boolean=True)
            
            if self.penalty == "quadratic":
                penalties = cp.sum(cp.pos(f - T)**2)
            else:
                raise NotImplementedError("Only quadratic penalty function is supported for now")
            
            objective = cp.Minimize(penalties) # minimize the total penalty
            
            constraints = []
            M = 1e5
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        constraints += [
                            s[i] + process_time[i] <= s[j] + M * (1 - y[i, j]),
                            s[j] + process_time[j] <= s[i] + M * y[i, j],
                            y[i, j] + y[j, i] == 1,
                        ]
            
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.MOSEK)
            expected_start_times = s.value.tolist()
            
            return [index for index, value in sorted(enumerate(expected_start_times), key=lambda x: x[1])]
        
        elif self.strategy == "RANDOM":
            pass
        elif self.strategy == "SRTF":
            pass
        elif self.strategy == "FIFO":
            pass
        else:
            raise ValueError("Invalid scheduling strategy")
        