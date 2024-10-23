from collections import defaultdict

class memory_allocator:
    def __init__(self, model):
        self.model = model
        self.SRAM_MAX_SIZE = 1 << 63
        self.live_range = defaultdict(dict)
        self.memory_allocate()

    def compute_tensor_live_range(self):
        operators = self.model['subgraphs'][0]['operators']

        # Record each input/output tensor's first and last time used
        for opid, op_info in enumerate(operators):
            for tensor_id in op_info['inputs'] + op_info['outputs']:
                # Update this tensor's first time used
                if opid < self.live_range[tensor_id].get('first_time_used', len(operators)):
                    self.live_range[tensor_id]['first_time_used'] = opid
                # Update this tensor's last time used
                if opid > self.live_range[tensor_id].get('last_time_used', -1):
                    self.live_range[tensor_id]['last_time_used'] = opid

    def greedy_by_size(self):
        pass

    def memory_allocate(self):
        self.compute_tensor_live_range()
        self.greedy_by_size()