from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

class IsaacEmptyRunner(BaseImageRunner):
    def __init__(self, output_dir):
        super().__init__(output_dir)

    def run(self, policy):
        # 欺骗训练器：直接返回空日志，什么都不做
        return {}