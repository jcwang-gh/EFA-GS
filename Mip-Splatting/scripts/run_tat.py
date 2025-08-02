# single-scale trainig and single-scale testing as in the original mipnerf-360

import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import queue
import time

# scenes = ["Auditorium", "Ballroom", "Courtroom", "Museum", "Palace", "Temple", "Family", "Francis", "Horse", "Lighthouse", "M60", "Panther", "Playground", "Train", "Barn", "Caterpillar", "Church", "Courthouse", "Ignatius", "Meetingroom", "Truck"]
scenes = ["Auditorium", "Ballroom", "Courtroom", "Museum", "Palace", "Temple"]
# factors = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
factors = [1, 1, 1, 1, 1, 1]
subfolders = ["advanced", "intermediate", "trainingdata"]
# indices = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
indices = [0, 0, 0, 0, 0, 0]
excluded_gpus = set([])

output_dir_prefix = "EFA_Mip_tat"
dataset_dir = "TanksandTemples"
dry_run = False

jobs = list(zip(scenes, factors, indices))

def train_scene(gpu, scene, factor, indice):
    scaler_max = 1.5
    interval_time = 2
    scaler_min = 1.0
    diffscale = 'True'
    output_dir = output_dir_prefix + "_s" + str(scaler_max)+"_t" + str(interval_time)
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s data/{dataset_dir}/{subfolders[indice]}/{scene} -m {output_dir}/{scene} --eval -r {factor} --port {6009+int(gpu)} --kernel_size 0.1 --init_scaling_multiplier_max {scaler_max} --init_scaling_multiplier_min {scaler_min} --interval_times {interval_time} --diffscale {diffscale}"
    print(cmd)
    if not dry_run:
        os.system(cmd)

    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {output_dir}/{scene} --data_device cpu --skip_train"
    print(cmd)
    if not dry_run:
        os.system(cmd)
        
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {output_dir}/{scene} -r {factor}"
    print(cmd)
    if not dry_run:
        os.system(cmd)
    return True


def worker(gpu, scene, factor, indice):
    print(f"Starting job on GPU {gpu} with scene {subfolders[indice]}/{scene}\n")
    train_scene(gpu, scene, factor, indice)
    print(f"Finished job on GPU {gpu} with scene {subfolders[indice]}/{scene}\n")
    # This worker function starts a job and returns when it's done.
    
def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)
        
        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)
        
    print("All jobs have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)

