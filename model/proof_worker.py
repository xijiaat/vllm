"""Same startup-only metadata worker in both arms; no forward hooks."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import torch
from vllm.v1.worker.gpu_worker import Worker


def source(obj):
    kind = type(obj)
    path = inspect.getsourcefile(kind)
    return {'class': kind.__module__ + '.' + kind.__qualname__,
            'source_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None}


class ProofWorker(Worker):
    def load_model(self, *, load_dummy_weights=False):
        assert not load_dummy_weights
        super().load_model(load_dummy_weights=load_dummy_weights)
        spec = self.vllm_config.speculative_config
        assert spec.method == 'dspark' and spec.num_speculative_tokens == 5
        assert spec.rejection_sample_method == 'standard'
        assert spec.synthetic_acceptance_rates is None and spec.synthetic_acceptance_length is None
        models = {'target': self.model_runner.get_model(), 'draft': self.model_runner.get_draft_model()}
        assert models['draft'] is not None
        assert models['draft'] is self.model_runner.speculator.model
        result = {'rank': self.rank, 'pid': os.getpid(), 'arm': os.environ['C1C2_ARM'],
                  'gpu': str(torch.cuda.get_device_properties(self.device)),
                  'gpu_uuid': str(torch.cuda.get_device_properties(self.device).uuid), 'models': {},
                  'synthetic_acceptance': False, 'forward_hooks': False,
                  'allocated_after_load': torch.cuda.memory_allocated(),
                  'reserved_after_load': torch.cuda.memory_reserved()}
        for role, model in models.items():
            item = {'source': source(model), 'compressors': [], 'linear_kernels': {}, 'moe_methods': {}}
            for name, module in model.named_modules():
                if hasattr(module, 'compressor') and module.compressor is not None:
                    active = getattr(module, '_fuse_compressor_cache', False)
                    assert active == (result['arm'] == 'B'), (role, name, active)
                    assert module.compressed_bytes_per_token == 584
                    item['compressors'].append({'name': name, 'ratio': module.compressor.compress_ratio,
                        'fusion': active, 'dtype': str(module.kv_cache_torch_dtype),
                        'record_bytes': module.compressed_bytes_per_token, 'source': source(module)})
                quant = getattr(module, 'quant_method', None)
                kernel = getattr(quant, 'kernel', None)
                if kernel is not None:
                    item['linear_kernels'][name] = source(kernel)
                if quant is not None and 'MoE' in type(module).__name__:
                    item['moe_methods'][name] = source(quant)
            if role == 'target':
                assert {x['ratio'] for x in item['compressors']} == {1, 2}
            else:
                item['note'] = 'DSpark draft shares target context KV; it may have no compressor of its own'
            result['models'][role] = item
        path = Path('/results/inventory')
        path.mkdir(exist_ok=True)
        (path / f'rank-{self.rank}.json').write_text(json.dumps(result, indent=2))
        print('C1C2_MODEL_PROOF_PASS', self.rank, result['arm'], flush=True)
