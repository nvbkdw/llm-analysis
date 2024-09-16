import sys
sys.path.append('/home/ruoyu.huang/workspace/llm-analysis')
from llm_analysis.analysis import ActivationRecomputation, DSZeRO, LLMAnalysis
from llm_analysis.config import (ParallelismConfig, get_dtype_config_by_name,
                                 get_gpu_config_by_name,
                                 get_model_config_by_name)
from llm_analysis.utils import _latency_to_string, _num_to_string, within_range

TOLERANCE = 0.05

# megatron-lm paper https://arxiv.org/abs/2104.04473 Table 2
def test_training_megatron_lm_1():
    model_name = "megatron-lm-175b"
    dtype_name = "w16a16e16"
    gpu_name = "h100-sxm-80gb"
    total_num_tokens = 300e9

    activation_recomputation = ActivationRecomputation.FULL
    tp_size = 8
    pp_size = 12
    total_num_gpus = 384
    dp_size = total_num_gpus // (tp_size * pp_size)
    batch_size_per_gpu = 8


    model_config = get_model_config_by_name(model_name)
    gpu_config = get_gpu_config_by_name(gpu_name)
    dtype_config = get_dtype_config_by_name(dtype_name)
    parallel_config = ParallelismConfig(tp_size=tp_size,
                                        pp_size=pp_size,
                                        dp_size=dp_size)

    # achieved_tflops = 153  # reported in the paper
    achieved_tflops = gpu_config.peak_fp16_TFLOPS * 0.6  # 60% of peak

    analysis = LLMAnalysis(
        model_config,
        gpu_config,
        dtype_config,
        parallel_config,
        achieved_tflops=achieved_tflops,
    )

    summary_dict = analysis.training(
        batch_size_per_gpu=batch_size_per_gpu,
        total_num_tokens=total_num_tokens,
        activation_recomputation=activation_recomputation,
    )

    breakpoint()

    assert within_range(
        summary_dict["total_training_latency_using_flops"] / 3600 / 24, 84,
        TOLERANCE)

    assert (_latency_to_string(
        summary_dict["total_training_latency_using_flops"]) == "84.82 days")

    assert _num_to_string(summary_dict["num_params_total"]) == "162.58 G"

if __name__ == '__main__':
    test_training_megatron_lm_1()
