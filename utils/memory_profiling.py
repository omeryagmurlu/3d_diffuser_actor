import torch
import torch.distributed as dist

MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100_000_000

class MemoryProfiler:
    def __init__(self, log_dir):
        self.log_dir = log_dir

        if dist.get_rank() == 0:
            torch.cuda.memory._record_memory_history(
                max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
            )

            def oom_observer(device, alloc, device_alloc, device_free):
                # snapshot right after an OOM happened
                print('saving allocated state during OOM')
                torch.cuda.memory._dump_snapshot(str(self.log_dir / f"memory_1_oom_snapshot.pickle"))


            torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    def dump_memory_profile(self, step):
        if dist.get_rank() == 0 and (step+1) % 10 == 0:
            torch.cuda.memory._dump_snapshot(str(self.log_dir / f"memory_at_{step}.pickle"))