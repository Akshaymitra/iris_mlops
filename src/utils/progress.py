from tqdm import tqdm

class PipelineProgress:
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.progress = tqdm(total=total_steps, desc="Pipeline Progress")

    def update(self, step_name):
        self.progress.update(1)
        self.progress.set_description(f"Running: {step_name}")

    def close(self):
        self.progress.close()
