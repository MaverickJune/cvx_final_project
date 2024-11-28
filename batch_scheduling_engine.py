import torch
import json
import os
import asyncio
import numpy as np
import cvxpy as cp
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import format_example, count_tokens, prepare_dataset_for_speed_eval, seed_everything, quadratic, linear, get_profile_penalty

class BatchSchedulingEngine:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", batch_size=4, seed=42, pad_token="<|pad|>", strategy="OPTIMAL", max_new_tokens=50, 
                 requests: dict={}, slo_list: list=[], penalty="quadratic", debug=False):
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
        self.data_queue = asyncio.Queue()
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
        self.process_time_pred = [] # prefill + decode, expected
        self.actual_process_time = [] # actual process time
        self.actual_finish_time = [] # actual finish time
        self.run_profile()
        
        self.slo_list = slo_list # should be represented in seconds (not ms)
        self.profile_penalty = get_profile_penalty(penalty)
        if penalty in ["quadratic"]:
            self.penalty = penalty
        else:
            raise NotImplementedError("Only quadratic penalty function is supported for now")
        
        self.running_order = self.generate_schedule()
        self.add_items_to_queue()
        
        if debug:
            self.init_debug()
        
    def init_debug(self):
        print("Debug mode is on")
        print("Prefill parameters: ", self.prefill_params)
        print("Decode parameters: ", self.decode_params)
        print("Prefill latency predictions: ", self.prefill_latency_pred)
        print("Decode latency predictions: ", self.decode_latency_pred)
        print("Process time predictions: ", self.process_time_pred)
        print("Actual process time: ", self.actual_process_time)
        print("SLO list: ", self.slo_list)
        print("Running order: ", self.running_order)
        print("Strategy: ", self.strategy)
        
        
    def register_requests(self, questions: dict):
        datasets = [prepare_dataset_for_speed_eval(question, self.batch_size, self.tokenizer).to("cuda") for question in questions]
        dataset_lengths_in_tokens = [count_tokens(format_example(question, self.tokenizer), self.tokenizer) for question in questions]
        self.datasets.extend(datasets)
        self.dataset_lengths_in_tokens.extend(dataset_lengths_in_tokens)
    
    
    def run_profile(self):
        prefill_latencies_pred = quadratic(np.array(self.dataset_lengths_in_tokens), *self.prefill_params).tolist()
        decoding_latencies_pred = [item * (self.max_new_tokens - 1) for item in linear(np.array(self.dataset_lengths_in_tokens), *self.decode_params).tolist()]
        
        # convert to seconds
        prefill_latencies_pred = [item/1000 for item in prefill_latencies_pred]
        decoding_latencies_pred = [item/1000 for item in decoding_latencies_pred]
        
        self.prefill_latency_pred.extend(prefill_latencies_pred)
        self.decode_latency_pred.extend(decoding_latencies_pred)
        
        self.actual_process_time = [-1] * self.num_requests
        self.actual_finish_time = [-1] * self.num_requests
    
        
    def generate_schedule(self) -> list[int]:
        self.process_time_pred.extend([sum(item) for item in zip(self.prefill_latency_pred, self.decode_latency_pred)])
        
        if self.strategy == "OPTIMAL": #MILP problem
            process_time = np.array(self.process_time_pred)
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
            return np.random.permutation(self.num_requests).tolist()
        
        elif self.strategy == "SRTF":
            return np.argsort(self.process_time_pred).tolist()
        
        elif self.strategy == "FIFO":
            return list(range(self.num_requests))
        
        else:
            raise ValueError("Invalid scheduling strategy")
        
        
    def add_items_to_queue(self):
        for index in self.running_order:
            data = self.datasets[index]
            item = (index, data)
            asyncio.get_event_loop().call_soon_threadsafe(self.data_queue.put_nowait, item)
            
            
    def analyze_and_record(self):
        log_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
        log_file_path = os.path.join(log_dir_path, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
        os.makedirs(log_dir_path, exist_ok=True)
        
        temp_cumsum = 0
        for idx in self.running_order:
            self.actual_finish_time[idx] = temp_cumsum + self.actual_process_time[idx]
            temp_cumsum = self.actual_finish_time[idx]
            
        slo_violation = self.profile_penalty(self.actual_finish_time, self.slo_list)
        individual_slo_violation = [self.profile_penalty(self.actual_finish_time[index], self.slo_list[index]) for index in range(self.num_requests)]
        
        log_results = {
            "predicted_prefill_latencies": self.prefill_latency_pred,
            "predicted_decode_latencies": self.decode_latency_pred,
            "predicted_latencies": self.process_time_pred,
            "actual_latencies": self.actual_process_time,
            "actual_finish_time": self.actual_finish_time,
            "slo_list": self.slo_list,
            "running_order": self.running_order,
            "individual_slo_violation": individual_slo_violation,
            "total_slo_violation": slo_violation,
            "strategy": self.strategy,
        }
        
        with open(log_file_path, "w") as f:
            json.dump(log_results, f, indent=4)
        
        
    async def llm_inference(self, data, index, warmup=False):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs = self.model.generate(**data, max_new_tokens = self.max_new_tokens)
        end.record()
        torch.cuda.synchronize()
        if not warmup:
            self.actual_process_time[index] = start.elapsed_time(end) / 1000 # convert to seconds
        
    
    async def process_queue(self):
        WARMUP_STEP = 50
        
        # warmup
        print("Warming up...")
        warmup_data = self.datasets[0]
        index = -1
        torch.cuda.synchronize()
        for _ in range(WARMUP_STEP):
            await self.llm_inference(warmup_data, index=index, warmup=True)
        torch.cuda.synchronize()
        print("Warming up done \n")
        
        self.processing = True
        print("Processing queue...")
        while True:
            try:
                index, data = await asyncio.wait_for(self.data_queue.get(), timeout=1)
                await self.llm_inference(data, index)
                self.data_queue.task_done()
                
            except asyncio.TimeoutError:
                if self.data_queue.empty():
                    self.processing = False
                    break
        print("Processing done \n")
        
    