import json

from LaughLM.analysis.metrics import (
    load_metrics,
    summarize_metrics,
)


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_summarize_metrics_selects_rows_after_skip(tmp_path):
    path = tmp_path / "metrics.jsonl"

    rows = [
        {
            "step": 1,
            "tokens_per_sec": 100.0,
            "device_tokens_per_sec": 110.0,
            "total_step_time": 10.0,
            "device_step_time": 9.0,
            "data_wait_time": 1.0,
            "host_batch_prepare_time": 1.0,
            "device_put_time": 1.0,
            "host_overhead_time": 1.0,
            "input_pipeline_time": 3.0,
            "mfu_non_embedding": 1.0,
            "mfu_with_logits_estimate": 2.0,
            "loss": 5.0,
            "ppl": 100.0,
            "tokens_in_step": 8192,
            "seq_len": 2048,
            "global_batch": 4,
            "micro_global_batch": 4,
            "effective_global_batch": 64,
            "gradient_accumulation": 16,
            "num_devices": 4,
        },
        {
            "step": 2,
            "tokens_per_sec": 200.0,
            "device_tokens_per_sec": 220.0,
            "total_step_time": 5.0,
            "device_step_time": 4.0,
            "data_wait_time": 0.5,
            "host_batch_prepare_time": 0.5,
            "device_put_time": 0.5,
            "host_overhead_time": 1.0,
            "input_pipeline_time": 1.5,
            "mfu_non_embedding": 3.0,
            "mfu_with_logits_estimate": 4.0,
            "loss": 4.0,
            "ppl": 50.0,
            "tokens_in_step": 8192,
            "seq_len": 2048,
            "global_batch": 4,
            "micro_global_batch": 4,
            "effective_global_batch": 64,
            "gradient_accumulation": 16,
            "num_devices": 4,
        },
    ]

    _write_jsonl(path, rows)

    summary = summarize_metrics(
        path,
        skip_steps=1,
    )

    assert summary["rows_total"] == 2
    assert summary["rows_selected"] == 1
    assert summary["first_step"] == 2
    assert summary["last_step"] == 2
    assert summary["tokens_per_sec_mean"] == 200.0
    assert summary["loss_last"] == 4.0
    assert summary["effective_global_batch"] == 64.0


def test_summarize_metrics_detects_device_step_bottleneck(tmp_path):
    path = tmp_path / "metrics.jsonl"

    rows = [
        {
            "step": 10,
            "tokens_per_sec": 100.0,
            "device_tokens_per_sec": 120.0,
            "total_step_time": 10.0,
            "device_step_time": 8.0,
            "data_wait_time": 0.2,
            "host_batch_prepare_time": 0.2,
            "device_put_time": 0.2,
            "host_overhead_time": 1.0,
            "input_pipeline_time": 0.6,
            "mfu_non_embedding": 2.0,
            "mfu_with_logits_estimate": 3.0,
        }
    ]

    _write_jsonl(path, rows)

    summary = summarize_metrics(path)

    assert summary["bottleneck"] == "device_step"


def test_load_metrics_skips_malformed_lines(tmp_path):
    path = tmp_path / "metrics.jsonl"

    with open(path, "w") as f:
        f.write(json.dumps({"step": 1}) + "\n")
        f.write("{bad json\n")
        f.write(json.dumps({"step": 2}) + "\n")

    rows = load_metrics(path)

    assert [row["step"] for row in rows] == [1, 2]
